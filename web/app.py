from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backend import (
    load_xyz, load_cmyk_10nm, load_pink_interpolated,
    optimize_mix, ref2lab,
)

XYZ_PATH  = os.path.join(PROJECT_ROOT, "data", "XYZ.xlsx")
CMYK_PATH = os.path.join(PROJECT_ROOT, "data", "cmyk-10nm.xlsx")
PINK_PATH = os.path.join(PROJECT_ROOT, "Dataset", "pink-interpolated.xlsx")

xyz_cache = load_xyz(XYZ_PATH)
wavelengths, cmyk_spectra = load_cmyk_10nm(CMYK_PATH)

all_spectra: dict[str, np.ndarray] = dict(cmyk_spectra)
if os.path.exists(PINK_PATH):
    pink_wl, pink_sp = load_pink_interpolated(PINK_PATH)
    if np.allclose(pink_wl, wavelengths):
        all_spectra.update(pink_sp)

COLOR_NAMES: List[str] = list(all_spectra.keys())

app = FastAPI(title="SpectOptiBlend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mkadz13.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def lab_to_rgb(L: float, a: float, b: float) -> List[int]:
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def finv(t):
        return t**3 if t > 6.0 / 29.0 else 3.0 * (6.0 / 29.0) ** 2 * (t - 4.0 / 29.0)

    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X = Xn * finv(fx)
    Y = Yn * finv(fy)
    Z = Zn * finv(fz)

    r_lin =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_lin =  0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def gamma(u):
        u = max(0.0, min(1.0, u))
        return 12.92 * u if u <= 0.0031308 else 1.055 * u ** (1.0 / 2.4) - 0.055

    return [
        max(0, min(255, int(round(gamma(r_lin) * 255)))),
        max(0, min(255, int(round(gamma(g_lin) * 255)))),
        max(0, min(255, int(round(gamma(b_lin) * 255)))),
    ]


class OptimizeRequest(BaseModel):
    target: str = Field(..., description="Name of target color")
    ingredients: List[str] = Field(..., min_length=2, description="Names of ingredient colors")
    solver: str = Field("Nelder-Mead", description="Solver: Nelder-Mead | SLSQP | L-BFGS-B")
    total_grams: float = Field(200.0, gt=0)


class MixEntry(BaseModel):
    color: str
    fraction: float
    grams: float


class LabValues(BaseModel):
    L: float
    a: float
    b: float


class OptimizeResponse(BaseModel):
    success: bool
    message: str
    solver: str
    target_name: str
    rms: float
    deltaE2000: float
    target_lab: LabValues
    predicted_lab: LabValues
    target_rgb: List[int]
    predicted_rgb: List[int]
    mix: List[MixEntry]
    wavelengths: List[float]
    target_spectrum: List[float]
    predicted_spectrum: List[float]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/colors")
async def get_colors():
    return {"colors": COLOR_NAMES}


@app.post("/api/optimize", response_model=OptimizeResponse)
async def api_optimize(req: OptimizeRequest):
    if req.target not in all_spectra:
        raise HTTPException(400, f"Unknown target color: {req.target}")

    ingredient_names = [n for n in req.ingredients if n != req.target]
    for name in ingredient_names:
        if name not in all_spectra:
            raise HTTPException(400, f"Unknown ingredient color: {name}")
    if len(ingredient_names) < 2:
        raise HTTPException(400, "Need at least 2 ingredients (excluding the target).")

    valid_solvers = {"Nelder-Mead", "SLSQP", "L-BFGS-B"}
    if req.solver not in valid_solvers:
        raise HTTPException(400, f"Solver must be one of {valid_solvers}")

    target_spec = all_spectra[req.target]
    ingredient_specs = [all_spectra[n] for n in ingredient_names]

    result = optimize_mix(
        target_spectrum=target_spec,
        ingredient_spectra=ingredient_specs,
        xyz=xyz_cache,
        wavelengths=wavelengths,
        solver=req.solver,
        mode="paper",
        weights_mode="notebook",
    )

    mix_arr = np.asarray(result["mix"], dtype=float)
    pred_arr = np.asarray(result["predicted_spectrum"], dtype=float)

    # Normalize fractions so they sum to 1 for gram calculation
    mix_sum = float(np.sum(mix_arr))
    recipe = mix_arr / mix_sum if mix_sum > 0 else np.full_like(mix_arr, 1.0 / len(mix_arr))

    target_lab = ref2lab(target_spec, xyz_cache)
    pred_lab = ref2lab(pred_arr, xyz_cache)

    return OptimizeResponse(
        success=result["success"],
        message=result["message"],
        solver=req.solver,
        target_name=req.target,
        rms=round(float(result["rms"]), 6),
        deltaE2000=round(float(result["deltaE2000"]), 5),
        target_lab=LabValues(L=round(float(target_lab[0]), 2),
                             a=round(float(target_lab[1]), 2),
                             b=round(float(target_lab[2]), 2)),
        predicted_lab=LabValues(L=round(float(pred_lab[0]), 2),
                                a=round(float(pred_lab[1]), 2),
                                b=round(float(pred_lab[2]), 2)),
        target_rgb=lab_to_rgb(float(target_lab[0]), float(target_lab[1]), float(target_lab[2])),
        predicted_rgb=lab_to_rgb(float(pred_lab[0]), float(pred_lab[1]), float(pred_lab[2])),
        mix=[
            MixEntry(color=name, fraction=round(float(r), 4), grams=round(float(r) * req.total_grams, 2))
            for name, r in zip(ingredient_names, recipe)
        ],
        wavelengths=wavelengths.tolist(),
        target_spectrum=target_spec.tolist(),
        predicted_spectrum=pred_arr.tolist(),
    )
