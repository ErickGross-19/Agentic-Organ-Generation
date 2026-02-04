"""Parameter presets and validation for vascular network design."""

from .presets import (
    liver_arterial_dense,
    liver_venous_sparse,
    kidney_arterial,
    sparse_debug,
    lung_arterial,
    brain_arterial,
    get_preset,
    list_presets,
    PRESETS,
)

from .validation import (
    validate_params,
    validate_and_warn,
    PARAM_BOUNDS,
)

__all__ = [
    # Presets
    "liver_arterial_dense",
    "liver_venous_sparse",
    "kidney_arterial",
    "sparse_debug",
    "lung_arterial",
    "brain_arterial",
    "get_preset",
    "list_presets",
    "PRESETS",
    # Validation
    "validate_params",
    "validate_and_warn",
    "PARAM_BOUNDS",
]
