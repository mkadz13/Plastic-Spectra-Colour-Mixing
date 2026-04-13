from __future__ import annotations

import os
import re
import sys
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backend import (
    load_xyz,
    load_cmyk_10nm,
    load_pink_interpolated,
    optimize_mix,
    ref2lab,
)

from web import db_spectra
from web.spectrum_upload import parse_spectrum_csv

XYZ_PATH = os.path.join(PROJECT_ROOT, "data", "XYZ.xlsx")
CMYK_PATH = os.path.join(PROJECT_ROOT, "data", "cmyk-10nm.xlsx")
PINK_PATH = os.path.join(PROJECT_ROOT, "Dataset", "pink-interpolated.xlsx")

xyz_cache = load_xyz(XYZ_PATH)
wavelengths, cmyk_spectra = load_cmyk_10nm(CMYK_PATH)

base_spectra: dict[str, np.ndarray] = dict(cmyk_spectra)
if os.path.exists(PINK_PATH):
    pink_wl, pink_sp = load_pink_interpolated(PINK_PATH)
    if np.allclose(pink_wl, wavelengths):
        base_spectra.update(pink_sp)

all_spectra: dict[str, np.ndarray] = {}
COLOR_NAMES: List[str] = []
color_rgb_cache: dict[str, List[int]] = {}


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

    r_lin = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_lin = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def gamma(u):
        u = max(0.0, min(1.0, u))
        return 12.92 * u if u <= 0.0031308 else 1.055 * u ** (1.0 / 2.4) - 0.055

    return [
        max(0, min(255, int(round(gamma(r_lin) * 255)))),
        max(0, min(255, int(round(gamma(g_lin) * 255)))),
        max(0, min(255, int(round(gamma(b_lin) * 255)))),
    ]


def rebuild_spectrum_cache() -> None:
    global all_spectra, COLOR_NAMES, color_rgb_cache
    merged: dict[str, np.ndarray] = dict(base_spectra)
    if db_spectra.database_url():
        try:
            merged.update(db_spectra.fetch_approved_spectra())
        except Exception:
            pass
    all_spectra.clear()
    all_spectra.update(merged)
    COLOR_NAMES[:] = sorted(all_spectra.keys(), key=str.lower)
    color_rgb_cache.clear()
    for name, spectrum in all_spectra.items():
        lab = ref2lab(spectrum, xyz_cache)
        color_rgb_cache[name] = lab_to_rgb(float(lab[0]), float(lab[1]), float(lab[2]))


try:
    db_spectra.ensure_schema()
except Exception:
    pass
rebuild_spectrum_cache()

_DEFAULT_CORS = [
    "https://mkadz13.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
_extra = os.environ.get("CORS_EXTRA_ORIGINS", "")
CORS_ORIGINS = list(_DEFAULT_CORS)
for part in _extra.split(","):
    p = part.strip()
    if p and p not in CORS_ORIGINS:
        CORS_ORIGINS.append(p)

app = FastAPI(title="SpectOptiBlend", version="1.0.0")


def _client_ip_for_rate_limit(request: Request) -> str:
    xff = request.headers.get("X-Forwarded-For") or request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip() or get_remote_address(request)
    return get_remote_address(request)


_limiter = Limiter(key_func=_client_ip_for_rate_limit)
app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

RATE_LIMIT_SUBMIT = os.environ.get("RATE_LIMIT_SUBMIT", "8/hour")
RATE_LIMIT_OPTIMIZE = os.environ.get("RATE_LIMIT_OPTIMIZE", "30/minute")
RATE_LIMIT_COLORS = os.environ.get("RATE_LIMIT_COLORS", "120/minute")
RATE_LIMIT_TEMPLATE = os.environ.get("RATE_LIMIT_TEMPLATE", "60/minute")
RATE_LIMIT_ADMIN = os.environ.get("RATE_LIMIT_ADMIN", "45/minute")
MAX_SPECTRUM_CSV_BYTES = int(os.environ.get("MAX_SPECTRUM_CSV_BYTES", str(3 * 1024 * 1024)))

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _require_admin(authorization: Optional[str]) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization: Bearer <ADMIN_SECRET> header.")
    token = authorization[7:].strip()
    if not db_spectra.admin_token_ok(token):
        raise HTTPException(403, "Invalid admin token.")


_COLOR_NAME_RE = re.compile(r"^[\w .\-]{1,80}$")


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


class SubmitSpectrumResponse(BaseModel):
    id: int
    message: str


class PendingItem(BaseModel):
    id: int
    color_name: str
    submitter_email: str
    created_at: str


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/admin.html")
async def admin_page():
    return FileResponse(os.path.join(STATIC_DIR, "admin.html"))


@app.get("/api/colors")
@_limiter.limit(RATE_LIMIT_COLORS)
async def get_colors(request: Request):
    return {
        "colors": [
            {"name": c, "rgb": color_rgb_cache[c]}
            for c in COLOR_NAMES
        ]
    }


@app.get("/api/spectrum-template.csv")
@_limiter.limit(RATE_LIMIT_TEMPLATE)
async def spectrum_template_csv(request: Request):
    lines = ["wavelength,reflectance"]
    for w in wavelengths:
        lines.append(f"{float(w):g},0")
    body = ("\n".join(lines) + "\n").encode("utf-8")
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="spectoptiblend_spectrum_template.csv"'},
    )


@app.post("/api/submit-spectrum", response_model=SubmitSpectrumResponse)
@_limiter.limit(RATE_LIMIT_SUBMIT)
async def submit_spectrum(
    request: Request,
    color_name: str = Form(...),
    submitter_email: str = Form(...),
    file: UploadFile = File(...),
):
    if not db_spectra.database_url():
        raise HTTPException(503, "This server is not configured with DATABASE_URL yet.")

    name = color_name.strip()
    if not name or not _COLOR_NAME_RE.match(name):
        raise HTTPException(
            400,
            "Invalid color name (1–80 chars: letters, numbers, spaces, underscore, hyphen, period).",
        )

    email = submitter_email.strip()
    if not email or "@" not in email:
        raise HTTPException(400, "Please enter a valid email.")
    if not db_spectra.allowed_submitter_email(email):
        suffix = os.environ.get("ALLOWED_EMAIL_SUFFIX", "@uwo.ca").strip()
        raise HTTPException(400, f"Only {suffix} addresses can submit for now.")

    raw = b""
    while True:
        chunk = await file.read(65536)
        if not chunk:
            break
        raw += chunk
        if len(raw) > MAX_SPECTRUM_CSV_BYTES:
            raise HTTPException(413, f"CSV too large (max {MAX_SPECTRUM_CSV_BYTES} bytes).")
    if not raw:
        raise HTTPException(400, "Empty file.")

    try:
        reflectance = parse_spectrum_csv(raw, wavelengths)
    except Exception as e:
        raise HTTPException(400, str(e))

    if name in all_spectra:
        raise HTTPException(400, "A color with this name already exists in the live library.")

    try:
        sid = db_spectra.insert_pending(name, reflectance, email)
    except Exception:
        raise HTTPException(500, "Could not save submission. Check DATABASE_URL / database status.")

    db_spectra.send_new_submission_email(
        submission_id=sid,
        color_name=name,
        submitter_email=email.lower(),
        uploaded_filename=(file.filename or "").strip(),

    )

    return SubmitSpectrumResponse(
        id=sid,
        message="Submitted for approval. It will appear in the menus after your instructor approves it.",
    )


@app.get("/api/admin/pending", response_model=List[PendingItem])
@_limiter.limit(RATE_LIMIT_ADMIN)
async def admin_list_pending(request: Request, authorization: Optional[str] = Header(default=None)):
    if not db_spectra.database_url():
        raise HTTPException(503, "DATABASE_URL is not set.")
    _require_admin(authorization)
    rows = db_spectra.list_pending()
    out: List[PendingItem] = []
    for r in rows:
        ts = r["created_at"]
        if hasattr(ts, "isoformat"):
            ts_s = ts.isoformat()
        else:
            ts_s = str(ts)
        out.append(
            PendingItem(
                id=int(r["id"]),
                color_name=str(r["color_name"]),
                submitter_email=str(r["submitter_email"]),
                created_at=ts_s,
            )
        )
    return out


@app.post("/api/admin/approve/{row_id}")
@_limiter.limit(RATE_LIMIT_ADMIN)
async def admin_approve(request: Request, row_id: int, authorization: Optional[str] = Header(default=None)):
    if not db_spectra.database_url():
        raise HTTPException(503, "DATABASE_URL is not set.")
    _require_admin(authorization)
    row = db_spectra.get_pending_row(row_id)
    if not row:
        raise HTTPException(404, "Submission not found.")
    if row["status"] != "pending":
        raise HTTPException(400, "Submission is not pending.")

    nm = str(row["color_name"]).strip()
    if nm in base_spectra:
        raise HTTPException(400, "This name matches a built-in catalog color; reject this submission or rename it in the database.")
    if nm in all_spectra:
        raise HTTPException(400, "A color with this name already exists in the live library.")

    ok = db_spectra.set_status(row_id, "approved")
    if not ok:
        raise HTTPException(409, "Could not approve (maybe already processed).")
    rebuild_spectrum_cache()
    return {"ok": True, "message": f"Approved '{nm}'. Menus updated."}


@app.post("/api/admin/reject/{row_id}")
@_limiter.limit(RATE_LIMIT_ADMIN)
async def admin_reject(request: Request, row_id: int, authorization: Optional[str] = Header(default=None)):
    if not db_spectra.database_url():
        raise HTTPException(503, "DATABASE_URL is not set.")
    _require_admin(authorization)
    row = db_spectra.get_pending_row(row_id)
    if not row:
        raise HTTPException(404, "Submission not found.")
    ok = db_spectra.set_status(row_id, "rejected")
    if not ok:
        raise HTTPException(409, "Could not reject (maybe already processed).")
    return {"ok": True, "message": "Rejected."}


@app.post("/api/optimize", response_model=OptimizeResponse)
@_limiter.limit(RATE_LIMIT_OPTIMIZE)
async def api_optimize(request: Request, req: OptimizeRequest):
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
        target_lab=LabValues(
            L=round(float(target_lab[0]), 2),
            a=round(float(target_lab[1]), 2),
            b=round(float(target_lab[2]), 2),
        ),
        predicted_lab=LabValues(
            L=round(float(pred_lab[0]), 2),
            a=round(float(pred_lab[1]), 2),
            b=round(float(pred_lab[2]), 2),
        ),
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
