"""MMPose-powered ViTPose inference utilities."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .vitpose_service import (
    KEYPOINT_NAMES,
    KeypointPrediction,
    PersonPrediction,
    PosePrediction,
    _ensure_image,  # reuse helper for path/PIL conversion
)

from grepl import constants

try:  # pragma: no cover - import guard for optional dependency
    import torch
    from mmpose.apis import inference_topdown, init_model
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "MMPoseViTPoseService requires 'torch' and 'mmpose' to be installed. "
        "Install project dependencies via 'uv sync' before using this service."
    ) from exc


@dataclass(frozen=True)
class _BBoxRequest:
    bbox: Tuple[float, float, float, float]


class MMPoseViTPoseService:
    """Run ViTPose inference using the MMPose runtime."""

    def __init__(
        self,
        *,
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        confidence_threshold: float = 0.3,
    ) -> None:
        self.config_path = Path(config_path or constants.MMPPOSE_VITPOSE_CONFIG)
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else (Path(constants.MMPPOSE_VITPOSE_CHECKPOINT) if constants.MMPPOSE_VITPOSE_CHECKPOINT else None)
        )

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"MMPose config not found at {self.config_path}. Update constants.MMPPOSE_VITPOSE_CONFIG or pass a path explicitly."
            )
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"MMPose checkpoint not found at {self.checkpoint_path}. Update constants.MMPPOSE_VITPOSE_CHECKPOINT or pass a path explicitly."
            )

        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = init_model(str(self.config_path),
                                 str(self.checkpoint_path) if self.checkpoint_path is not None else None,
                                 device=target_device)
        self.device = target_device
        self.confidence_threshold = confidence_threshold
        self.model_id = f"mmpose::{self.config_path.name}"

    # ------------------------------------------------------------------
    def predict(
        self,
        frame: Union[Path, str, Image.Image, np.ndarray],
        *,
        boxes: Optional[Sequence[Sequence[float]]] = None,
        occlusion_threshold: Optional[float] = None,
        max_people: Optional[int] = None,
    ) -> PosePrediction:
        """Run MMPose ViTPose inference on the provided frame."""

        image, maybe_path = _ensure_image(frame)
        width, height = image.size

        if boxes is None:
            boxes = [[0.0, 0.0, float(width), float(height)]]
        if max_people is not None and len(boxes) > max_people:
            boxes = boxes[:max_people]
        if not boxes:
            raise ValueError("At least one bounding box is required for MMPose inference.")

        # MMPose expects BGR numpy arrays
        img_array = np.array(image)[:, :, ::-1]

        bbox_requests = [_BBoxRequest(tuple(map(float, box))) for box in boxes]
        person_results = [dict(bbox=np.array(bbox.bbox, dtype=np.float32)) for bbox in bbox_requests]

        data_samples = inference_topdown(
            self.model,
            img_array,
            person_results,
            bbox_format="xyxy",
        )

        people, raw_keypoints, raw_scores = self._format_people(
            boxes=bbox_requests,
            data_samples=data_samples,
            occlusion_threshold=occlusion_threshold,
        )

        raw_keypoints_array = (
            np.stack(raw_keypoints)
            if raw_keypoints
            else np.empty((0, len(KEYPOINT_NAMES), 3), dtype=np.float32)
        )
        raw_scores_array = (
            np.stack(raw_scores)
            if raw_scores
            else np.empty((0, len(KEYPOINT_NAMES)), dtype=np.float32)
        )

        return PosePrediction(
            frame_path=maybe_path,
            image_size=(width, height),
            model_id=self.model_id,
            people=people,
            raw_keypoints=raw_keypoints_array,
            raw_scores=raw_scores_array,
        )

    # ------------------------------------------------------------------
    def _format_people(
        self,
        *,
        boxes: Sequence[_BBoxRequest],
        data_samples,
        occlusion_threshold: Optional[float],
    ) -> Tuple[Tuple[PersonPrediction, ...], List[np.ndarray], List[np.ndarray]]:
        threshold = (
            occlusion_threshold if occlusion_threshold is not None else self.confidence_threshold
        )

        people: List[PersonPrediction] = []
        raw_keypoints: List[np.ndarray] = []
        raw_scores: List[np.ndarray] = []

        for bbox_req, sample in zip(boxes, data_samples):
            pred_instances = getattr(sample, "pred_instances", None)
            if pred_instances is None or not hasattr(pred_instances, "keypoints"):
                continue
            if len(pred_instances.keypoints) == 0:
                continue

            keypoints_xy = pred_instances.keypoints.detach().cpu().numpy()
            if keypoints_xy.ndim == 3:
                keypoints_xy = keypoints_xy[0]

            if keypoints_xy.shape[0] != len(KEYPOINT_NAMES):
                raise ValueError(
                    f"Expected {len(KEYPOINT_NAMES)} keypoints, got {keypoints_xy.shape[0]}"
                )

            if hasattr(pred_instances, "keypoint_scores") and pred_instances.keypoint_scores is not None:
                kp_scores = pred_instances.keypoint_scores.detach().cpu().numpy()
                if kp_scores.ndim == 2:
                    kp_scores = kp_scores[0]
            else:
                kp_scores = np.ones(len(KEYPOINT_NAMES), dtype=np.float32)

            combined = np.concatenate([keypoints_xy, kp_scores[..., None]], axis=1)
            raw_keypoints.append(combined.astype(np.float32))
            raw_scores.append(kp_scores.astype(np.float32))

            keypoint_preds = []
            for name, (x, y, score) in zip(KEYPOINT_NAMES, combined):
                confidence = float(score)
                keypoint_preds.append(
                    KeypointPrediction(
                        name=name,
                        x=float(x),
                        y=float(y),
                        confidence=confidence,
                        occluded=confidence < threshold,
                    )
                )

            if hasattr(pred_instances, "bbox_scores") and pred_instances.bbox_scores is not None:
                person_score = float(pred_instances.bbox_scores.detach().cpu().numpy()[0])
            else:
                person_score = float(np.mean(kp_scores))

            if hasattr(pred_instances, "bboxes") and pred_instances.bboxes is not None:
                pred_bbox = pred_instances.bboxes.detach().cpu().numpy()
                if pred_bbox.ndim == 2:
                    pred_bbox = pred_bbox[0]
                bbox_tuple = tuple(float(v) for v in pred_bbox[:4])
            else:
                bbox_tuple = bbox_req.bbox

            people.append(
                PersonPrediction(
                    bbox=bbox_tuple,
                    score=person_score,
                    keypoints=tuple(keypoint_preds),
                )
            )

        return tuple(people), raw_keypoints, raw_scores


@functools.lru_cache(maxsize=1)
def get_mmpose_vitpose_service() -> MMPoseViTPoseService:
    """Return a cached instance of `MMPoseViTPoseService`."""

    return MMPoseViTPoseService()


__all__ = [
    "MMPoseViTPoseService",
    "get_mmpose_vitpose_service",
]
