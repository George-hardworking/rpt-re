"""Storage for binary OHLC images (pixel values 0/255).

Current format is bit-packed (``IMAGES_EXT = .bin``): one byte holds 8
pixels, losslessly. Every supported window has ``height * width`` divisible
by 8, so one image row in a file is exactly ``(height * width) // 8`` bytes
with no per-row padding.

The reader also accepts legacy unpacked ``.dat`` files (one uint8 byte per
pixel) so that in-place migration of existing bundles does not break jobs that
spawn fresh processes mid-run. Once all bundles are migrated to ``.bin`` the
``.dat`` branch can be removed.

Information cutoff: storage only; no look-ahead concern here. Images are
written in as_of order by the generation pipeline; this module just (un)packs
bytes and exposes row-wise random access for training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import image_bundle_dir
from data.images import image_shape

IMAGES_EXT = ".bin"
LEGACY_IMAGES_EXT = ".dat"


def packed_bytes_per_image(window_days: int) -> int:
    height, width = image_shape(window_days)
    pixels = height * width
    if pixels % 8 != 0:
        raise ValueError(
            f"window={window_days} image has {pixels} pixels, not divisible by 8"
        )
    return pixels // 8


def bytes_per_image_for(path: Path, window_days: int) -> int:
    if path.suffix == IMAGES_EXT:
        return packed_bytes_per_image(window_days)
    height, width = image_shape(window_days)
    return height * width


def pack_image(image: np.ndarray) -> bytes:
    """Pack a uint8 binary image (values 0/255) to packed bit bytes."""
    return np.packbits(image.ravel()).tobytes()


def resolve_image_file(
    images_root: Path,
    window_days: int,
    year: int,
    *,
    sample_freq: str,
) -> Path:
    """Path of the on-disk image file for a year, preferring .bin over .dat."""
    freq_dir = Path(images_root) / image_bundle_dir(window_days, sample_freq)
    bin_path = freq_dir / f"{window_days}d_{year}_images{IMAGES_EXT}"
    if bin_path.is_file():
        return bin_path
    dat_path = freq_dir / f"{window_days}d_{year}_images{LEGACY_IMAGES_EXT}"
    if dat_path.is_file():
        return dat_path
    raise FileNotFoundError(
        f"no image file for window={window_days} freq={sample_freq} year={year} in {freq_dir}"
    )


def list_image_years(
    images_root: Path,
    window_days: int,
    sample_freq: str,
) -> list[int]:
    """Sorted years present on disk, deduped across .bin and .dat."""
    freq_dir = Path(images_root) / image_bundle_dir(window_days, sample_freq)
    years: set[int] = set()
    for ext in (IMAGES_EXT, LEGACY_IMAGES_EXT):
        for path in freq_dir.glob(f"{window_days}d_*_images{ext}"):
            years.add(int(path.stem.split("_")[1]))
    return sorted(years)


def load_image_memmap(path: Path, window_days: int):
    """Read-only row-indexable view over an image file (.bin or legacy .dat)."""
    path = Path(path)
    if path.suffix == IMAGES_EXT:
        return BitPackedImageMemmap(path, window_days)
    height, width = image_shape(window_days)
    flat = np.memmap(path, dtype=np.uint8, mode="r")
    n_images = flat.size // (height * width)
    return flat.reshape(n_images, height, width)


class BitPackedImageMemmap:
    """Read-only memmap over a bit-packed image file with row fancy indexing."""

    def __init__(self, path: Path, window_days: int) -> None:
        self.path = Path(path)
        self.window_days = window_days
        self.height, self.width = image_shape(window_days)
        self.row_bytes = packed_bytes_per_image(window_days)
        flat = np.memmap(self.path, dtype=np.uint8, mode="r")
        if flat.size % self.row_bytes != 0:
            raise ValueError(
                f"{self.path}: size {flat.size} not a multiple of "
                f"{self.row_bytes} bytes/image (window={window_days})"
            )
        self.n_images = flat.size // self.row_bytes
        self._rows = flat.reshape(self.n_images, self.row_bytes)

    def __len__(self) -> int:
        return self.n_images

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_images, self.height, self.width)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            bits = np.unpackbits(self._rows[idx])
            return bits.reshape(self.height, self.width) * np.uint8(255)
        if isinstance(idx, slice):
            idx_arr = np.arange(*idx.indices(self.n_images), dtype=np.int64)
        else:
            idx_arr = np.asarray(idx, dtype=np.int64)
        gathered = self._rows[idx_arr]  # (n, row_bytes) copy
        bits = np.unpackbits(gathered, axis=1)  # (n, height*width)
        return bits.reshape(len(idx_arr), self.height, self.width) * np.uint8(255)
