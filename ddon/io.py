from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def load_input(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".mat":
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        if "Profile_Px" not in mat:
            raise ValueError("MAT file must contain Profile_Px.")
        profile = mat["Profile_Px"]
        values = np.asarray(profile.Px, dtype=np.float32)
        u = float(np.asarray(profile.u).reshape(-1)[0])
        y_true = None
        if hasattr(profile, "Ex"):
            ex = np.asarray(profile.Ex).squeeze()
            if ex.size == 109:
                y_true = ex.reshape(109)
        return values, u, y_true

    if suffix in {".csv", ".txt"}:
        values = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
        return np.asarray(values, dtype=np.float32), None, None

    raise ValueError("Supported input formats are .mat, .csv, and .txt.")


def save_output(path, z, efield, y_true=None):
    path = Path(path)
    suffix = path.suffix.lower()
    z = np.asarray(z).reshape(-1)
    efield = np.asarray(efield).reshape(-1)

    if suffix == ".mat":
        data = {"z": z, "efield": efield}
        if y_true is not None:
            data["Ex"] = np.asarray(y_true).reshape(-1)
        savemat(path, data)
        return

    if suffix == ".csv":
        cols = [z, efield]
        header = "z,efield"
        if y_true is not None:
            cols.append(np.asarray(y_true).reshape(-1))
            header += ",Ex"
        np.savetxt(path, np.column_stack(cols), delimiter=",", header=header, comments="")
        return

    raise ValueError("Output file must end in .csv or .mat.")
