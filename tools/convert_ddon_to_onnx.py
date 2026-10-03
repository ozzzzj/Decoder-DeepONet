"""One-time DDON Keras -> ONNX conversion and numerical verification.

This does not change the original DDON model or sample scripts.
"""

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

from self_layers import ResidualBlock_dense, Jitter


MODEL_URL = (
    "https://github.com/ozzzzj/Decoder-DeepONet/releases/download/DDON/"
    "20260520_09-39_AM%2Bmodel.Epoch-27_Loss-0.000404%2BMSE-0.000244%2BBatsize-.512.h5"
)


def get_model(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading DDON model to {path}")
        urlretrieve(MODEL_URL, path)
    return path


def load_model(path: Path):
    return tf.keras.models.load_model(
        str(path),
        custom_objects={
            "ResidualBlock": ResidualBlock_dense,
            "ResidualBlock_dense": ResidualBlock_dense,
            "Jitter": Jitter,
        },
        compile=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="model/DDON.h5")
    parser.add_argument("--onnx", default="model/DDON.onnx")
    parser.add_argument("--opset", type=int, default=15)
    args = parser.parse_args()

    h5_path = get_model(Path(args.h5))
    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(h5_path)
    print("Keras inputs:")
    for inp in model.inputs:
        print(" ", inp.name, inp.shape, inp.dtype)
    print("Keras outputs:")
    for out in model.outputs:
        print(" ", out.name, out.shape, out.dtype)

    # DDON has two inputs: EFISH values (batch, 109, 2) and u (batch, 1).
    signature = (
        tf.TensorSpec((None, 109, 2), tf.float32, name="values"),
        tf.TensorSpec((None, 1), tf.float32, name="u"),
    )

    tf2onnx.convert.from_keras(
        model,
        input_signature=signature,
        opset=args.opset,
        output_path=str(onnx_path),
    )
    print(f"Saved ONNX model: {onnx_path}")

    # Deterministic synthetic input for numerical equivalence checking.
    z = np.linspace(-1.0, 1.0, 109, dtype=np.float32)
    efish = np.exp(-((z / 0.25) ** 2)).astype(np.float32)
    values = np.stack([z, efish], axis=-1)[None, :, :]
    # raw u=-0.35 and DDON U_SCALE=-1 -> normalized model input = +0.35
    u_norm = np.asarray([[0.35]], dtype=np.float32)

    tf_out = np.asarray(model.predict([values, u_norm], verbose=0)).squeeze()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    print("ONNX inputs:")
    for inp in inputs:
        print(" ", inp.name, inp.shape, inp.type)

    if len(inputs) != 2:
        raise RuntimeError(f"Expected 2 ONNX inputs, got {len(inputs)}")

    # The explicit signature normally preserves this order.
    ort_out = np.asarray(
        session.run(None, {inputs[0].name: values, inputs[1].name: u_norm})[0]
    ).squeeze()

    abs_err = np.abs(tf_out - ort_out)
    print("max_abs_error:", float(abs_err.max()))
    print("mean_abs_error:", float(abs_err.mean()))

    if not np.allclose(tf_out, ort_out, rtol=1e-4, atol=1e-5):
        raise RuntimeError("ONNX output does not match TensorFlow within tolerance.")

    print("PASS: TensorFlow and ONNX predictions match.")


if __name__ == "__main__":
    main()
