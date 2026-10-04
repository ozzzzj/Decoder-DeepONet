import argparse

from .io import load_input, save_output
from .predictor import DDON


def main():
    parser = argparse.ArgumentParser(
        prog="ddon",
        description="Local DDON electric-field reconstruction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="Run DDON inference.")
    predict.add_argument("input", help="Input .mat, .csv, or .txt file.")
    predict.add_argument(
        "--u",
        type=float,
        default=None,
        help="Raw u value. Required for CSV/TXT; MAT reads Profile_Px.u automatically.",
    )
    predict.add_argument(
        "-o", "--output", default="ddon_prediction.csv",
        help="Output .csv or .mat file.",
    )
    predict.add_argument("--model", default=None, help="Optional local DDON.onnx path.")

    args = parser.parse_args()

    values, file_u, y_true = load_input(args.input)
    u = args.u if args.u is not None else file_u
    if u is None:
        parser.error("--u is required when input does not contain Profile_Px.u")

    model = DDON(model_path=args.model)
    efield = model.predict(values, u)
    save_output(args.output, values[:, 0], efield, y_true)
    print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
