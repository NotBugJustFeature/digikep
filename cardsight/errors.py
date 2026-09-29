class CardSightError(RuntimeError):
    """Base exception for actionable CardSight failures."""


class DatasetDownloadError(CardSightError):
    """Raised when the selected dataset cannot be downloaded."""


class DatasetConfigurationError(CardSightError):
    """Raised when the dataset does not contain the expected YOLO structure."""


class ModelNotFoundError(CardSightError):
    """Raised when inference is requested before a model has been trained."""


class MediaProcessingError(CardSightError):
    """Raised when input media cannot be decoded or output media cannot be written."""
