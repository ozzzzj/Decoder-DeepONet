from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.ddon_inference import EXPECTED_POINTS, load_ddon_model, predict_ddon


ddon_model = None
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ddon_model
    ddon_model = load_ddon_model()
    yield
    ddon_model = None


app = FastAPI(
    title="EFISH Reconstruction API",
    version="0.1.0",
    description="Web API for DDON electric-field reconstruction from EFISH data.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class DDONRequest(BaseModel):
    model: str = Field(default="ddon")
    values: list[list[float]]
    u: float


class DDONResponse(BaseModel):
    model: str
    z: list[float]
    efield: list[float]


@app.get("/", include_in_schema=False)
def web_interface():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": "ddon", "loaded": ddon_model is not None}


@app.post("/predict", response_model=DDONResponse)
def predict(request: DDONRequest):
    if request.model.lower() != "ddon":
        raise HTTPException(status_code=400, detail="Only model='ddon' is available in this API version.")
    if len(request.values) != EXPECTED_POINTS:
        raise HTTPException(status_code=422, detail=f"DDON requires exactly {EXPECTED_POINTS} input points.")
    try:
        prediction = predict_ddon(ddon_model, request.values, request.u)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    z = [float(row[0]) for row in request.values]
    return DDONResponse(model="ddon", z=z, efield=prediction.astype(float).tolist())
