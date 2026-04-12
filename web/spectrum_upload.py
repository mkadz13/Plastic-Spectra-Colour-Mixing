from __future__ import annotations

import io
import numpy as np
import pandas as pd


def parse_spectrum_csv(
    content: bytes,
    expected_wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Parse a two-column CSV: wavelength, reflectance (paper-style 10 nm grid).
    Column names are case-insensitive; extra columns are rejected.
    Returns a 1D reflectance array aligned to ``expected_wavelengths``.
    """
    df = pd.read_csv(io.BytesIO(content))
    df.columns = [str(c).strip().lower() for c in df.columns]

    wl_col = None
    ref_col = None
    for c in df.columns:
        if c in ("wavelength", "wl", "lambda"):
            wl_col = c
        elif c in ("reflectance", "r", "refl"):
            ref_col = c

    if wl_col is None or ref_col is None:
        raise ValueError("CSV must contain wavelength and reflectance columns (e.g. wavelength,reflectance).")

    extra = [c for c in df.columns if c not in (wl_col, ref_col)]
    if extra:
        raise ValueError(f"Unexpected columns: {', '.join(extra)}. Use only wavelength and reflectance.")

    wl = df[wl_col].to_numpy(dtype=float)
    ref = df[ref_col].to_numpy(dtype=float)

    if len(wl) != len(expected_wavelengths):
        raise ValueError(
            f"Expected {len(expected_wavelengths)} wavelength rows (same as app grid); got {len(wl)}."
        )

    if not np.allclose(wl, expected_wavelengths, rtol=0, atol=0.51):
        raise ValueError("Wavelength column does not match the app's fixed 10 nm grid. Download the template CSV.")

    ref = np.clip(np.nan_to_num(ref, nan=0.0), 0.0, 1.0)
    return ref.astype(float)
