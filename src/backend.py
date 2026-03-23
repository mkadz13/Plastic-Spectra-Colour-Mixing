from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import deltae


SolverName = Literal["SLSQP", "L-BFGS-B", "Nelder-Mead"]
ModeName = Literal["paper", "recipe"]
WeightsName = Literal["uniform", "notebook"]


NUM_RESTARTS = 10

# The paper's specific starting point for 8 ingredients
_PAPER_INITIAL_8 = np.array([0.23, 0.13, 0.02, 0.08, 0.15, 0.3, 0.07, 0.04])


@dataclass(frozen=True)
class XYZCache:
    wavelengths: np.ndarray
    E: np.ndarray
    xbar: np.ndarray
    ybar: np.ndarray
    zbar: np.ndarray


def load_xyz(xyz_path: str) -> XYZCache:
    wl = np.array(pd.read_excel(xyz_path, usecols="A")).astype(float).flatten()
    E = np.array(pd.read_excel(xyz_path, usecols="B")).astype(float).flatten()
    xyzbar = np.array(pd.read_excel(xyz_path, usecols="C:E")).astype(float)
    xbar, ybar, zbar = xyzbar.T
    return XYZCache(wavelengths=wl, E=E, xbar=xbar, ybar=ybar, zbar=zbar)


def ref2lab(R: np.ndarray, xyz: XYZCache) -> np.ndarray:
    R = np.asarray(R, dtype=float).flatten()
    E, xbar, ybar, zbar = xyz.E, xyz.xbar, xyz.ybar, xyz.zbar

    K = 100.0 / np.dot(E, ybar)

    Xn = K * np.dot(E, xbar)
    Yn = K * np.dot(E, ybar)
    Zn = K * np.dot(E, zbar)

    X = K * np.dot(E * R, xbar)
    Y = K * np.dot(E * R, ybar)
    Z = K * np.dot(E * R, zbar)

    def f(t: np.ndarray) -> np.ndarray:
        delta = 6 / 29
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)

    L = 116 * f(Y / Yn) - 16
    a = 500 * (f(X / Xn) - f(Y / Yn))
    b = 200 * (f(Y / Yn) - f(Z / Zn))
    return np.array([L, a, b], dtype=float)


def cal_delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    lab1 = np.asarray(lab1, dtype=float).flatten()
    lab2 = np.asarray(lab2, dtype=float).flatten()

    lab1_dict = {"L": lab1[0], "a": lab1[1], "b": lab1[2]}
    lab2_dict = {"L": lab2[0], "a": lab2[1], "b": lab2[2]}

    de = deltae.delta_e_2000(lab1_dict, lab2_dict)
    de = np.nan_to_num(de, nan=0.0)
    return float(np.round(np.mean(de), 5))


def weighted_rms(pred: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float).flatten()
    target = np.asarray(target, dtype=float).flatten()
    weights = np.asarray(weights, dtype=float).flatten()
    diffs = pred - target
    return float(np.sqrt(np.mean((diffs * diffs) * weights)))


def notebook_weights(wavelengths: np.ndarray) -> np.ndarray:
    w = np.ones_like(wavelengths, dtype=float)
    w[(wavelengths >= 400) & (wavelengths <= 450)] += 15
    w[(wavelengths > 450) & (wavelengths <= 550)] -= 0.25
    w[(wavelengths >= 600) & (wavelengths <= 700)] *= 1.5
    return w


def uniform_weights(wavelengths: np.ndarray) -> np.ndarray:
    return np.ones_like(wavelengths, dtype=float)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    seen = set()
    for col in df.columns:
        name = str(col).strip()
        if name.lower().startswith("unnamed"):
            continue
        base = name.split(".")[0]
        if base == "wavelength":
            keep.append(col)
            continue
        if base in seen:
            continue
        seen.add(base)
        keep.append(col)
    return df[keep].rename(columns=lambda c: str(c).strip())


def load_cmyk_10nm(cmyk_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    df = pd.read_excel(cmyk_path)
    df = _clean_columns(df)

    if "wavelength" not in df.columns:
        raise ValueError("cmyk-10nm.xlsx must contain a 'wavelength' column")

    wavelengths = df["wavelength"].to_numpy(dtype=float)
    spectra: Dict[str, np.ndarray] = {}
    for col in df.columns:
        if col == "wavelength":
            continue
        spectra[col] = df[col].to_numpy(dtype=float)
    return wavelengths, spectra


def load_pink_interpolated(pink_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    df = pd.read_excel(pink_path)
    df = _clean_columns(df)

    if "wavelength" not in df.columns:
        raise ValueError("pink-interpolated.xlsx must contain a 'wavelength' column")

    wavelengths = df["wavelength"].to_numpy(dtype=float)
    spectra: Dict[str, np.ndarray] = {}
    for col in df.columns:
        if col == "wavelength":
            continue
        spectra[col] = df[col].to_numpy(dtype=float)
    return wavelengths, spectra


def _objective(
    concentrations: np.ndarray,
    target_spectrum: np.ndarray,
    target_lab: np.ndarray,
    colorant_spectra: np.ndarray,
    weights: np.ndarray,
    xyz: XYZCache,
    rms_weight: float,
    delta_e_weight: float,
) -> float:

    predicted_spectrum = np.dot(concentrations, colorant_spectra)

    rms_error = weighted_rms(predicted_spectrum, target_spectrum, weights)

    predicted_lab = ref2lab(predicted_spectrum, xyz)
    delta_e_error = cal_delta_e(predicted_lab, target_lab)

    return float(rms_weight * rms_error + delta_e_weight * delta_e_error)


def _random_starts(n: int, num_starts: int, rng: np.random.Generator) -> List[np.ndarray]:
    starts = []
    if n == 8:
        starts.append(_PAPER_INITIAL_8.copy())
    else:
        starts.append(np.full(n, 0.5))
    for _ in range(num_starts - 1):
        x = rng.dirichlet(np.ones(n))
        x *= rng.uniform(0.5, 1.5)
        x = np.clip(x, 0.0, 1.0)
        starts.append(x)
    return starts


def optimize_mix(
    target_spectrum: np.ndarray,
    ingredient_spectra: List[np.ndarray],
    xyz: XYZCache,
    wavelengths: np.ndarray,
    solver: SolverName = "Nelder-Mead",
    mode: ModeName = "paper",
    weights_mode: WeightsName = "notebook",
    initial: Optional[np.ndarray] = None,
) -> Dict[str, object]:

    target_spectrum = np.asarray(target_spectrum, dtype=float).flatten()
    n = len(ingredient_spectra)

    colorant_spectra = np.array(
        [np.asarray(s, dtype=float).flatten() for s in ingredient_spectra]
    )

    weights = notebook_weights(wavelengths) if weights_mode == "notebook" else uniform_weights(wavelengths)

    rms_w, de_w = 0.1, 0.9

    target_lab = ref2lab(target_spectrum, xyz)
    bounds = [(0, 1) for _ in range(n)]

    if initial is not None:
        starts = [np.asarray(initial, dtype=float)]
    else:
        rng = np.random.default_rng(42)
        starts = _random_starts(n, NUM_RESTARTS, rng)

    best_res = None
    best_fun = np.inf

    for x0 in starts:
        res = minimize(
            _objective,
            x0,
            args=(target_spectrum, target_lab, colorant_spectra, weights, xyz, rms_w, de_w),
            method=solver,
            bounds=bounds,
        )
        if res.fun < best_fun:
            best_fun = res.fun
            best_res = res

    x = np.clip(best_res.x, 0.0, 1.0)

    if mode == "recipe":
        s = np.sum(x)
        if s > 0:
            x = x / s
        else:
            x = np.full_like(x, 1.0 / n)

    pred = np.dot(colorant_spectra.T, x)
    rms = weighted_rms(pred, target_spectrum, weights)
    de = cal_delta_e(ref2lab(pred, xyz), target_lab)

    return {
        "success": bool(best_res.success),
        "message": str(best_res.message),
        "mix": x,
        "predicted_spectrum": pred,
        "rms": rms,
        "deltaE2000": de,
        "fun": float(best_res.fun),
        "solver": solver,
        "mode": mode,
        "weights_mode": weights_mode,
    }


PAPER_8 = ["cyan", "magenta", "green", "blue", "black", "Red petg", "blue-pla", "cream-pla"]
PAPER_4 = ["cyan", "magenta", "green", "black"]
