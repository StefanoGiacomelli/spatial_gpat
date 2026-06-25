"""
datasets — Public Dataset API
=============================
Dataset classes, augmentation utilities, projection wrappers, and Lightning
DataModules used by the NAS-SELD and standalone SELD training pipelines.

The package exposes dataset implementations for STARSS23, TAU-NIGENS2020,
TAU-NIGENS2021, TAU2019, and BalancedSTARSS23.

Author
------
Stefano Giacomelli, Ph.D. candidate
DISIM Department — University of L'Aquila
"""

from .utils         import SELDDataset, SELDAugment
from .starss23      import STARSS23Dataset
from .tau_nigens    import TAUNIGENSDataset
from .tau_nigens2020 import TAUNIGENS2020Dataset
from .tau2019       import TAU2019Dataset
from .balanced   import (BalancedSTARSS23Dataset, ProjectedSELDDataset,
                          NIGENS_TO_STARSS13, TAU2019_TO_STARSS13)
from .datamodule import (SELDDataModule, BalancedSELDDataModule,
                          FoldCyclingDataModule)

__all__ = [
    'SELDDataset',
    'SELDAugment',
    'STARSS23Dataset',
    'TAUNIGENSDataset',
    'TAUNIGENS2020Dataset',
    'TAU2019Dataset',
    'BalancedSTARSS23Dataset',
    'ProjectedSELDDataset',
    'NIGENS_TO_STARSS13',
    'TAU2019_TO_STARSS13',
    'SELDDataModule',
    'BalancedSELDDataModule',
    'FoldCyclingDataModule',
]
