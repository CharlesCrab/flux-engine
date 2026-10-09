import base64
import binascii
import warnings
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError


def decode_reference(encoded: str) -> Image.Image:
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("reference image exceeds 8 MiB")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("reference must be PNG, JPEG, or WebP")
                if image.width * image.height > 4096 * 4096:
                    raise ValueError("reference image exceeds 16 megapixels")
                image.load()
                result = ImageOps.exif_transpose(image).convert("RGB")
                result.thumbnail((1024, 1024))
                return result
    except (
        binascii.Error,
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("invalid reference image") from exc
