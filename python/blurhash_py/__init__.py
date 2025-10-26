import enum
from contextlib import nullcontext

from PIL import Image

from ._lib_name import blurhash_for_pixels_py, decode_blurhash_py, is_valid_blurhash_py

__version__ = "0.2.2"

__all__ = (
    "BlurhashDecodeError",
    "PixelMode",
    "__version__",
    "decode",
    "encode",
    "is_valid_blurhash",
)


class PixelMode(enum.IntEnum):
    RGB = 3
    RGBA = 4


class BlurhashDecodeError(Exception):
    def __init__(self, blurhash: str) -> None:
        self.blurhash = blurhash

    def __str__(self) -> str:
        return f"Failed to decode blurhash {self.blurhash}"


def encode(image, x_components: int, y_components: int) -> str:
    """Encode an image to a blurhash string."""
    if isinstance(image, Image.Image):
        image_context = nullcontext()
    else:
        image = Image.open(image)
        image_context = image
    with image_context:
        if image.mode != "RGB":
            image = image.convert("RGB")
        rgb_data = image.tobytes()
        width, height = image.size

    result = blurhash_for_pixels_py(
        x_components,
        y_components,
        width,
        height,
        rgb_data,
        width * 3,
    )

    if result is None:
        msg = "Invalid x_components or y_components"
        raise ValueError(msg)
    return result


def decode(
    blurhash: str,
    width: int,
    height: int,
    punch: float = 1,
    mode: PixelMode = PixelMode.RGB,
):
    """Decode a blurhash string to an image."""
    if width <= 0 or not isinstance(width, int):
        msg = f"Argument width={width} is not a valid positive integer (must be > 0)."
        raise ValueError(msg)

    if height <= 0 or not isinstance(height, int):
        msg = f"Argument height={height} is not a valid positive integer (must be > 0)."
        raise ValueError(msg)

    if punch < 1 or not isinstance(punch, (int, float)):
        msg = f"Argument punch={punch} is not a valid positive number (must be >= 1)."
        raise ValueError(msg)

    if not isinstance(mode, PixelMode):
        msg = f"Argument 'mode' must be of type {PixelMode} but got {type(mode)}"
        raise TypeError(msg)

    channels = mode.value

    # Call the Rust decode wrapper: returns bytes, rgb order, row-major
    try:
        bytes_data = decode_blurhash_py(blurhash, width, height, punch)
    except Exception:
        raise BlurhashDecodeError(blurhash)

    # decode_blurhash_py returns a Python bytes object.
    # NB: While mode=RGBA might be supported, our Rust code outputs only RGB,
    # so we only support RGB at the moment. If/when RGBA is supported,
    # handle accordingly.
    if not bytes_data or len(bytes_data) != width * height * channels:
        raise BlurhashDecodeError(blurhash)

    return Image.frombuffer(
        mode.name,
        (width, height),
        bytes_data,
        "raw",
        mode.name,
        0,
        1,
    )


def is_valid_blurhash(blurhash: str) -> bool:
    """Check if a blurhash string is valid."""
    return is_valid_blurhash_py(blurhash)
