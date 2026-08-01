from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.build_credible_dataset import Label, Sample, apply_reviews, stratified_group_split
from scripts.apply_pseudo_labels import accepted_rows
from scripts.compare_evaluations import compare_reports
from scripts.evaluate_detector import Box as EvalBox
from scripts.evaluate_detector import (
    coco_payload,
    fuse_external_predictions,
    fuse_prediction_sets,
    run_coco_eval,
    validate_args,
    weighted_box_fusion,
)
from scripts.make_crop_dataset import write_efficient_view
from scripts.make_class_projection_dataset import project_label
from scripts.make_sliced_dataset import Box as SliceBox
from scripts.make_sliced_dataset import remap_box, tile_origins
from scripts.mine_missing_labels import consensus_candidates
from train import AUGMENT_PRESETS, BestMap50Checkpoint, build_train_kwargs
from gf_insuyolo.modules import ContextGuidedEnhance
from gf_insuyolo.boxes import Detection
from infer import combine_two_stage_defects


class DatasetTests(unittest.TestCase):
    def test_pseudo_labels_can_require_human_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proposals = Path(directory) / "proposals.csv"
            proposals.write_text(
                "score,agreement_iou,gt_any_iou,review_status\n"
                "0.9,0.9,0.0,approved\n"
                "0.1,0.1,0.9,approved\n"
                "0.9,0.9,0.0,rejected\n"
                "0.9,0.9,0.0,pending\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                proposals=proposals,
                min_score=0.65,
                min_agreement_iou=0.7,
                max_gt_iou=0.1,
                require_review=True,
            )
            self.assertEqual(len(accepted_rows(args)), 2)

    def test_class_projection_reindexes_and_drops_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source.txt", root / "projected.txt"
            source.write_text(
                "0 0.5 0.5 0.1 0.1\n1 0.4 0.4 0.2 0.2\n3 0.6 0.6 0.3 0.3\n",
                encoding="utf-8",
            )
            self.assertEqual(project_label(source, destination, {1: 0, 3: 2}), 2)
            self.assertEqual(
                destination.read_text(encoding="utf-8").splitlines(),
                ["0 0.4 0.4 0.2 0.2", "2 0.6 0.6 0.3 0.3"],
            )

    def test_efficient_crop_view_keeps_base_negatives_and_positive_jitter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images" / "train"
            label_root = root / "labels" / "train"
            image_root.mkdir(parents=True)
            label_root.mkdir(parents=True)
            for stem, label in {
                "negative_crop00": "",
                "negative_crop00_jitter01": "",
                "positive_crop00": "0 0.5 0.5 0.1 0.1\n",
                "positive_crop00_jitter01": "0 0.5 0.5 0.1 0.1\n",
                "positive_crop00_jitter02": "0 0.5 0.5 0.1 0.1\n",
            }.items():
                (image_root / f"{stem}.jpg").touch()
                (label_root / f"{stem}.txt").write_text(label, encoding="utf-8")
            self.assertEqual(write_efficient_view(root, 1), 3)
            selected = (root / "train_efficient.txt").read_text(encoding="utf-8")
            self.assertIn("negative_crop00.jpg", selected)
            self.assertIn("positive_crop00_jitter01.jpg", selected)
            self.assertNotIn("negative_crop00_jitter01.jpg", selected)
            self.assertNotIn("positive_crop00_jitter02.jpg", selected)

    def test_group_split_never_leaks(self) -> None:
        groups = {}
        for index in range(40):
            labels = tuple(Label(class_id, 0.5, 0.5, 0.1, 0.1) for class_id in range(4))
            groups[f"g{index}"] = [Sample("source", __file__, f"s{index}", f"f{index}", labels, group_id=f"g{index}")]
        result = stratified_group_split(groups, (0.8, 0.1, 0.1), 7)
        assigned = {}
        for split, samples in result.items():
            for sample in samples:
                self.assertNotIn(sample.group_id, assigned)
                assigned[sample.group_id] = split
        self.assertEqual(len(assigned), 40)

    def test_review_can_relabel_a_defect(self) -> None:
        sample = Sample(
            "CPLID_Defective", __file__, "cplid_defective_1", "family",
            (Label(0, 0.5, 0.5, 0.5, 0.5), Label(3, 0.4, 0.4, 0.1, 0.1)),
        )
        result = apply_reviews(
            [sample],
            {sample.stem: {"review_status": "corrected", "corrected_defect_class": "broken_shell", "corrected_yolo": ""}},
        )
        self.assertEqual([label.class_id for label in result[0].labels], [0, 1])

    def test_missing_label_miner_requires_cross_model_agreement(self) -> None:
        first = EvalBox(1, 0.9, (10, 10, 50, 50))
        second = EvalBox(1, 0.8, (12, 12, 52, 52))
        candidates = consensus_candidates([[first], [second]], [], 0.5, 0.2)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].class_id, 1)
        labeled = [EvalBox(1, 1.0, (9, 9, 51, 51))]
        self.assertEqual(consensus_candidates([[first], [second]], labeled, 0.5, 0.2), [])


class SliceTests(unittest.TestCase):
    def test_tile_origins_cover_last_pixel(self) -> None:
        self.assertEqual(tile_origins(2000, 768, 0.2)[-1], 1232)

    def test_box_coordinate_round_trip(self) -> None:
        box = SliceBox(2, 850, 300, 950, 400)
        row = remap_box(box, (700, 100, 1468, 868), 0.6)
        self.assertIsNotNone(row)
        _, x, y, width, height = row
        self.assertAlmostEqual(x * 768 + 700, 900)
        self.assertAlmostEqual(y * 768 + 100, 350)
        self.assertAlmostEqual(width * 768, 100)
        self.assertAlmostEqual(height * 768, 100)


class EvaluationTests(unittest.TestCase):
    def test_local_preferred_fusion_replaces_global_defects_inside_crop(self) -> None:
        global_inside = Detection(1, "broken_shell", 0.9, (10, 10, 20, 20), "global")
        global_outside = Detection(1, "broken_shell", 0.8, (80, 80, 90, 90), "global")
        local = Detection(1, "broken_shell", 0.7, (11, 11, 21, 21), "local_crop")
        fused = combine_two_stage_defects(
            [global_inside, global_outside], [local], [(0, 0, 50, 50)], "local-preferred"
        )
        self.assertEqual(fused, [global_outside, local])

    def test_promotion_uses_map50(self) -> None:
        baseline = {
            "dataset_fingerprint": "same",
            "split": "val",
            "metrics": {"map50": 0.80, "map50_95": 0.50, "map75": 0.60},
            "per_class": {
                name: {"map50": 0.80, "map50_95": 0.50}
                for name in ["broken_shell", "flashover_pollution", "missing_disc_drop"]
            },
        }
        candidate = {
            "dataset_fingerprint": "same",
            "split": "val",
            "metrics": {"map50": 0.83, "map50_95": 0.45, "map75": 0.61},
            "per_class": {
                name: {"map50": 0.83, "map50_95": 0.45}
                for name in ["broken_shell", "flashover_pollution", "missing_disc_drop"]
            },
        }
        result = compare_reports(baseline, candidate)
        self.assertTrue(result["promoted"])
        self.assertEqual(result["promotion_metric"], "map50")

    def test_fusion_stays_in_image(self) -> None:
        boxes = [EvalBox(1, 0.9, (-5, 5, 110, 90)), EvalBox(1, 0.8, (0, 0, 100, 100))]
        fused = weighted_box_fusion(boxes, 0.5, 100, 100)
        self.assertEqual(len(fused), 1)
        self.assertTrue(all(0 <= value <= 100 for value in fused[0].xyxy))

    def test_standard_mode_rejects_ignored_ensemble_weights(self) -> None:
        args = argparse.Namespace(
            mode="standard",
            weights="primary.pt",
            ensemble_weights=["second.pt"],
            ensemble_class_offsets=[],
            external_predictions=None,
        )
        with self.assertRaisesRegex(ValueError, "requires --mode ensemble"):
            validate_args(args)

    def test_external_fusion_keeps_all_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            from PIL import Image

            path = Path(directory) / "image.jpg"
            Image.new("RGB", (100, 80)).save(path)
            primary = {path.resolve(): [EvalBox(1, 0.8, (10, 10, 30, 30))]}
            external = {path.resolve(): [EvalBox(2, 0.7, (50, 40, 70, 60))]}
            fused = fuse_external_predictions(primary, external, [path], 0.55)
            self.assertEqual(len(fused[path.resolve()]), 2)

    def test_multiple_external_prediction_sets_are_fused_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            from PIL import Image

            path = Path(directory) / "image.jpg"
            Image.new("RGB", (100, 100)).save(path)
            first = {
                path.resolve(): [EvalBox(1, 0.9, (10.0, 10.0, 40.0, 40.0))]
            }
            second = {
                path.resolve(): [EvalBox(1, 0.8, (12.0, 12.0, 42.0, 42.0))]
            }
            fused = fuse_prediction_sets([first, second], [path], 0.55)
            self.assertEqual(len(fused[path.resolve()]), 1)
            self.assertGreater(fused[path.resolve()][0].confidence, 0.8)

    def test_perfect_coco_map(self) -> None:
        target = EvalBox(0, 1.0, (10, 20, 50, 80))
        dataset, predictions, _ = coco_payload({"a.jpg": [target]}, {"a.jpg": [target]})
        self.assertAlmostEqual(run_coco_eval(dataset, predictions)[0], 1.0)


class TrainTests(unittest.TestCase):
    def test_context_guided_module_preserves_shape(self) -> None:
        import torch

        module = ContextGuidedEnhance(32)
        tensor = torch.randn(2, 32, 20, 24)
        output = module(tensor)
        self.assertEqual(output.shape, tensor.shape)
        self.assertTrue(torch.equal(output, tensor))

    def test_map50_checkpoint_tracks_primary_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = root / "last.pt"
            last.write_bytes(b"first")
            trainer = SimpleNamespace(
                metrics={"metrics/mAP50(B)": 0.7}, wdir=root, last=last, epoch=3
            )
            callback = BestMap50Checkpoint()
            callback(trainer)
            self.assertEqual((root / "best_map50.pt").read_bytes(), b"first")
            last.write_bytes(b"worse")
            trainer.metrics["metrics/mAP50(B)"] = 0.6
            callback(trainer)
            self.assertEqual((root / "best_map50.pt").read_bytes(), b"first")

    def test_defect_safe_augmentation_is_less_aggressive(self) -> None:
        safe = AUGMENT_PRESETS["defect_safe"]
        moderate = AUGMENT_PRESETS["moderate"]
        self.assertLess(safe["hsv_s"], moderate["hsv_s"])
        self.assertLess(safe["scale"], moderate["scale"])
        self.assertLess(safe["mosaic"], moderate["mosaic"])
        self.assertEqual(safe["flipud"], 0.0)
        self.assertEqual(safe["copy_paste"], 0.0)
        self.assertEqual(AUGMENT_PRESETS["none"]["augmentations"], [])

    def test_training_overrides_are_forwarded(self) -> None:
        args = argparse.Namespace(
            data="data.yaml", epochs=2, imgsz=640, batch=2, project="runs", name="test",
            workers=0, resume=False, close_mosaic=3, cos_lr=True, patience=4, seed=9,
            deterministic=True, optimizer="AdamW", cache=False, exist_ok=True, device="0",
            lr0=0.001, lrf=0.01, weight_decay=0.0005, freeze=5, augment_preset="none",
            warmup_epochs=0.0, warmup_bias_lr=0.0, warmup_momentum=0.8,
            context_only=False,
        )
        kwargs = build_train_kwargs(args)
        self.assertEqual(kwargs["seed"], 9)
        self.assertEqual(kwargs["optimizer"], "AdamW")
        self.assertEqual(kwargs["lr0"], 0.001)
        self.assertEqual(kwargs["warmup_epochs"], 0.0)
        self.assertEqual(kwargs["warmup_bias_lr"], 0.0)
        self.assertEqual(kwargs["mosaic"], 0.0)

    def test_context_only_freezes_all_non_context_layers(self) -> None:
        args = argparse.Namespace(
            data="data.yaml", epochs=2, imgsz=640, batch=2, project="runs", name="test",
            workers=0, resume=False, close_mosaic=0, cos_lr=True, patience=2, seed=9,
            deterministic=True, optimizer="AdamW", cache=False, exist_ok=True, device="0",
            lr0=0.001, lrf=0.01, weight_decay=0.0005, freeze=None, augment_preset="none",
            warmup_epochs=0.0, warmup_bias_lr=0.0, warmup_momentum=0.8,
            context_only=True,
        )
        self.assertEqual(build_train_kwargs(args)["freeze"], [*range(23), 26])


if __name__ == "__main__":
    unittest.main()
