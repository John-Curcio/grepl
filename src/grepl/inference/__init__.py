"""Inference services (ViTPose, etc.)."""

from .vitpose_service import (
    KEYPOINT_NAMES,
    KeypointPrediction,
    PersonPrediction,
    PosePrediction,
    ViTPoseInferenceService,
    get_vitpose_service,
)
from .mmpose_vitpose_service import (
    MMPoseViTPoseService,
    get_mmpose_vitpose_service,
)

__all__ = [
    "KEYPOINT_NAMES",
    "KeypointPrediction",
    "PersonPrediction",
    "PosePrediction",
    "ViTPoseInferenceService",
    "get_vitpose_service",
    "MMPoseViTPoseService",
    "get_mmpose_vitpose_service",
]
