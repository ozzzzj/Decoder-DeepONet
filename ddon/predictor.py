from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import onnxruntime as ort


MODEL_URL = "https://github.com/ozzzzj/Decoder-DeepONet/releases/download/DDON-WEB/DDON.onnx"
U_SCALE = -1.0
EXPECTED_POINTS = 109


class DDON:
    """Lightweight local DDON inference using ONNX Runtime."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path.home() / ".ddon" / "DDON.onnx"
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading DDON model to {self.model_path}")
            urlretrieve(MODEL_URL, self.model_path)

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

    def predict(self, values, u):
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (EXPECTED_POINTS, 2):
            raise ValueError(
                f"values must have shape ({EXPECTED_POINTS}, 2); got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("values contains NaN or infinite values")

        raw_u = float(np.asarray(u).reshape(-1)[0])
        if not np.isfinite(raw_u):
            raise ValueError("u must be finite")

        values_input = values.reshape(1, EXPECTED_POINTS, 2)
        u_input = np.asarray([[raw_u / U_SCALE]], dtype=np.float32)

        inputs = self.session.get_inputs()
        outputs = self.session.run(
            None,
            {
                inputs[0].name: values_input,
                inputs[1].name: u_input,
            },
        )
        return np.asarray(outputs[0]).squeeze().reshape(EXPECTED_POINTS)
