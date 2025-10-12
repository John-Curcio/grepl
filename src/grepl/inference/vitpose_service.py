"""ViTPose inference utilities.

This module centralises model loading, preprocessing, and prediction formatting
for ViTPose. It relies on Hugging Face Transformers to load
`VitPoseForKeypointDetection` and caches the model + processor in memory for
repeated use.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from grepl import constants

# Order aligns with the user's ViTPose keypoint specification.
KEYPOINT_NAMES: Tuple[str, ...] = (
    "Nose",
    "L_Eye",
    "R_Eye",
    "L_Ear",
    "R_Ear",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
)


# Delayed heavy imports so that simply importing the module doesn't require
# `transformers`/`torch` until the service is instantiated. We keep the module
# importable even if the optional dependencies are absent so that other
# inference backends (e.g. MMPose) can still reuse the keypoint metadata.
try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover
    from transformers import AutoImageProcessor, VitPoseForKeypointDetection
except ImportError:  # pragma: no cover
    AutoImageProcessor = None  # type: ignore[assignment]
    VitPoseForKeypointDetection = None  # type: ignore[assignment]


@dataclass(frozen=True)
class KeypointPrediction:
    """Single keypoint prediction in image pixel space."""

    name: str
    x: float
    y: float
    confidence: float
    occluded: bool


@dataclass(frozen=True)
class PersonPrediction:
    """Pose prediction for one person (bbox + skeleton)."""

    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    score: float
    keypoints: Tuple[KeypointPrediction, ...]


@dataclass(frozen=True)
class PosePrediction:
    """Structured output for a frame-level inference request."""

    frame_path: Optional[Path]
    image_size: Tuple[int, int]  # (width, height)
    model_id: str
    people: Tuple[PersonPrediction, ...]
    raw_keypoints: np.ndarray
    raw_scores: Optional[np.ndarray]


def _ensure_image(frame: Union[Path, str, Image.Image, np.ndarray]) -> Tuple[Image.Image, Optional[Path]]:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB"), None
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame).convert("RGB"), None
    path = Path(frame)
    return Image.open(path).convert("RGB"), path


class ViTPoseInferenceService:
    """Loads ViTPose once and serves pose predictions for frames."""

    def __init__(
        self,
        *,
        model_id: str = constants.VITPOSE_MODEL_ID,
        cache_dir: Optional[Union[str, Path]] = constants.VITPOSE_CACHE_DIR,
        device: Optional[Union[str, torch.device]] = None,
        confidence_threshold: float = 0.3,
    ) -> None:
        if AutoImageProcessor is None or VitPoseForKeypointDetection is None or torch is None:
            raise ImportError(
                "ViTPoseInferenceService requires the 'torch' and 'transformers' packages. "
                "Install the project dependencies before instantiating this service."
            )

        self.model_id = model_id
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.confidence_threshold = confidence_threshold

        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.image_processor = AutoImageProcessor.from_pretrained(
            model_id,
            cache_dir=self.cache_dir,
        )
        self.model = VitPoseForKeypointDetection.from_pretrained(
            model_id,
            cache_dir=self.cache_dir,
            torch_dtype=torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(  # pylint: disable=too-many-arguments
        self,
        frame: Union[Path, str, Image.Image, np.ndarray],
        *,
        boxes: Optional[Sequence[Sequence[float]]] = None,
        occlusion_threshold: Optional[float] = None,
        max_people: Optional[int] = None,
    ) -> PosePrediction:
        """Run ViTPose on `frame` and return structured predictions.

        Args:
            frame: Image input (path, PIL image, or numpy array).
            boxes: Optional iterable of bounding boxes (xmin, ymin, xmax, ymax).
                If omitted, the entire image is treated as a single box.
            occlusion_threshold: Override the default confidence threshold for
                marking joints as occluded.
            max_people: If set, truncate predictions to the first `max_people` boxes.
        """

        image, maybe_path = _ensure_image(frame)
        width, height = image.size

        if boxes is None:
            boxes = [[0.0, 0.0, float(width), float(height)]]

        if max_people is not None and len(boxes) > max_people:
            boxes = boxes[:max_people]

        if not boxes:
            raise ValueError("At least one bounding box is required for ViTPose inference.")

        model_inputs = self.image_processor(
            images=image,
            boxes=boxes,
            return_tensors="pt",
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        # `outputs.keypoints` has shape (num_people, num_keypoints, 3)
        keypoints = outputs.keypoints.detach().cpu().numpy()
        scores = None
        if hasattr(outputs, "keypoint_scores") and outputs.keypoint_scores is not None:
            scores = outputs.keypoint_scores.detach().cpu().numpy()

        people = self._format_people(
            boxes=boxes,
            keypoints=keypoints,
            scores=scores,
            occlusion_threshold=occlusion_threshold,
        )

        return PosePrediction(
            frame_path=maybe_path,
            image_size=(width, height),
            model_id=self.model_id,
            people=people,
            raw_keypoints=keypoints,
            raw_scores=scores,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _format_people(
        self,
        *,
        boxes: Sequence[Sequence[float]],
        keypoints: np.ndarray,
        scores: Optional[np.ndarray],
        occlusion_threshold: Optional[float],
    ) -> Tuple[PersonPrediction, ...]:
        threshold = (
            occlusion_threshold if occlusion_threshold is not None else self.confidence_threshold
        )

        people: List[PersonPrediction] = []
        for idx, (bbox, kp_array) in enumerate(zip(boxes, keypoints)):
            # Some models return (x, y, score) in the last dimension.
            if kp_array.shape[-1] < 3:
                raise ValueError(
                    "Expected keypoints with (x, y, score); got shape {}".format(kp_array.shape)
                )
            keypoint_scores = kp_array[:, 2]
            if scores is not None and idx < len(scores):
                # Use the explicit keypoint_scores tensor if available.
                keypoint_scores = np.asarray(scores[idx]).reshape(-1)
            else:
                keypoint_scores = np.asarray(keypoint_scores).reshape(-1)

            keypoint_preds: List[KeypointPrediction] = []
            for name, (x, y, score) in zip(KEYPOINT_NAMES, kp_array):
                conf = float(score)
                keypoint_preds.append(
                    KeypointPrediction(
                        name=name,
                        x=float(x),
                        y=float(y),
                        confidence=conf,
                        occluded=conf < threshold,
                    )
                )

            person_score = float(np.mean(keypoint_scores)) if len(keypoint_scores) else 0.0
            people.append(
                PersonPrediction(
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    score=person_score,
                    keypoints=tuple(keypoint_preds),
                )
            )

        return tuple(people)


@functools.lru_cache(maxsize=1)
def get_vitpose_service() -> ViTPoseInferenceService:
    """Return a process-wide singleton for ViTPose inference."""

    return ViTPoseInferenceService()
