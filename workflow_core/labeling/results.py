from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


AtomsList = List[object]
GroupedTrainVal = Dict[str, Tuple[AtomsList, AtomsList]]


@dataclass
class LabelingOutputs:
    labeled_structures: AtomsList
    uncertainties: np.ndarray
    valid_structures: AtomsList
    train_structures: AtomsList
    dft_structures: AtomsList
    discard_structures: AtomsList
    grouped_train_val: GroupedTrainVal
