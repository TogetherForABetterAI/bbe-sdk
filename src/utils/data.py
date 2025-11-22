import numpy as np
from typing import Optional, NamedTuple, Union


class InputsFormat(NamedTuple):
    dtype: np.dtype
    shape: tuple[int, ...]


def parse_inputs_format(inputs_format_str: str) -> Optional[InputsFormat]:
    if not inputs_format_str or not inputs_format_str.strip():
        return None

    if not (inputs_format_str.startswith("(") and inputs_format_str.endswith(")")):
        raise ValueError(f"Invalid shape format in: {inputs_format_str}")

    try:
        shape_str = inputs_format_str.strip("()")
        shape_parts = [part.strip() for part in shape_str.split(",") if part.strip()]

        if not shape_parts:
            raise ValueError("Empty shape not allowed")

        shape = tuple(map(int, shape_parts))

        if any(dim <= 0 for dim in shape):
            raise ValueError("All dimensions must be positive")

    except ValueError as e:
        raise ValueError(f"Invalid shape format in: {inputs_format_str}") from e

    return InputsFormat(shape=shape, dtype=np.float32)
