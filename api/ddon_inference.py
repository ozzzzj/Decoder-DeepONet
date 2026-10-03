import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from self_layers import ResidualBlock_dense, Jitter


U_SCALE = -1.0
EXPECTED_POINTS = 109

DEFAULT_MODEL_NAME = (
    "20260520_09-39_AM+model.Epoch-27_Loss-0.000404+MSE-0.000244+"
    "Batsize-[512].h5"
)


def _default_model_path() -> Path:
    env_path = os.getenv("DDON_MODEL_PATH")
    if env_path:
        return Path(env_path)
    return Path("model log") / DEFAULT_MODEL_NAME


def load_ddon_model(model_path=None):
    path = Path(model_path) if model_path is not None else _default_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"DDON model file not found: {path}. "
            "Set DDON_MODEL_PATH or place the .h5 file under 'model log/'."
        )

    return tf.keras.models.load_model(
        str(path),
        custom_objects={
            "ResidualBlock": ResidualBlock_dense,
            "ResidualBlock_dense": ResidualBlock_dense,
            "Jitter": Jitter,
        },
        compile=False,
    )


def predict_ddon(model, values, u):
    """
    Run DDON inference only.

    Parameters
    ----------
    model
        Loaded TensorFlow/Keras DDON model.
    values
        Array with shape (109, 2). Column 0 is the normalized z coordinate
        used by DDON; column 1 is the normalized EFISH profile.
    u
        Raw DDON u value. The current DDON model uses U_SCALE = -1.

    Returns
    -------
    numpy.ndarray
        Predicted normalized electric-field profile.
    """
    values = np.asarray(values, dtype=np.float32)

    if values.shape != (EXPECTED_POINTS, 2):
        raise ValueError(
            f"values must have shape ({EXPECTED_POINTS}, 2); "
            f"received {values.shape}"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("values contains NaN or infinite values")

    u = float(u)
    if not np.isfinite(u):
        raise ValueError("u must be finite")

    xinput = values.reshape(1, EXPECTED_POINTS, 2)
    input_u = np.asarray([[u / U_SCALE]], dtype=np.float32)

    y_pred = model.predict([xinput, input_u], verbose=0)
    y_pred = np.asarray(y_pred).squeeze()

    if y_pred.size != EXPECTED_POINTS:
        raise RuntimeError(
            f"Unexpected DDON output size: {y_pred.size}; "
            f"expected {EXPECTED_POINTS}"
        )

    return y_pred.reshape(EXPECTED_POINTS)
