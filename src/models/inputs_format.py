import numpy as np
from typing import NamedTuple

class InputsFormat(NamedTuple):
    """Format specification for model inputs."""

    dtype: np.dtype
    shape: tuple[int, ...]
