"""Encoders package - Multi-modal encoders for cardiac data"""

from .ecg_encoder import (
    ECGEncoder,
    vit_tiny_patchX,
    vit_small_patchX,
    vit_base_patchX,
    build_ecg_encoder
)

from .localizer_encoder import (
    LocalizerEncoder,
    vit_tiny_localizer,
    vit_small_localizer,
    vit_base_localizer,
    build_localizer_encoder
)

from .tabular_encoder import (
    TabularEncoder,
    build_tabular_encoder,
    UKBB_NUMERICAL_FEATURES,
    UKBB_BINARY_FEATURES,
    UKBB_MULTI_CATEGORICAL_FEATURES
)

__all__ = [
    # ECG
    'ECGEncoder',
    'vit_tiny_patchX',
    'vit_small_patchX',
    'vit_base_patchX',
    'build_ecg_encoder',
    
    # Localizer
    'LocalizerEncoder',
    'vit_tiny_localizer',
    'vit_small_localizer',
    'vit_base_localizer',
    'build_localizer_encoder',
    
    # Tabular
    'TabularEncoder',
    'build_tabular_encoder',
    'UKBB_NUMERICAL_FEATURES',
    'UKBB_BINARY_FEATURES',
    'UKBB_MULTI_CATEGORICAL_FEATURES',
]
