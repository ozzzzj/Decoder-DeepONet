# Decoder-DeepONet (DDON)

Model and code of Decoder DeepONet (DDON) for electric-field reconstruction from EFISH measurements. This model is specifically designed for vertically polarized EFISH signals (for a vertically polarized probe beam).

## Citation

If you use DDON, the web tool, or results generated with this model, please cite:

> Yang, Z., Sugeng, E. S., Alicherif, M. and Chng, T. L. (2026). An interpretable operator-learning model for electric field profile reconstruction in discharges based on the EFISH method. *Plasma Sources Science and Technology*, **35**(2), 025035.

DOI: [10.1088/1361-6595/ae413f](https://doi.org/10.1088/1361-6595/ae413f)

## Choose how to use DDON

### 1. Online Web App — no installation

Run DDON directly in a web browser:

**[Launch the DDON E-field Reconstruction Web App](https://ozzzzj.github.io/Decoder-DeepONet/)**

The web app supports preprocessed CSV and MATLAB MAT inputs. Inference is performed locally in the browser using ONNX Runtime Web.

For DDON, the EFISH polarization is fixed to **vertical**.

### 2. Local Packaged Inference — lightweight ONNX Runtime

This mode is recommended if you want to run DDON locally, especially for predictions of multiple EFISH profiles, without installing TensorFlow:

```bash
pip install ddon-efish
```

**[View `ddon-efish` on PyPI](https://pypi.org/project/ddon-efish/)**


#### MATLAB MAT input (Recommended)

The recommended input is a MATLAB MAT file containing:

- `Profile_Px.Px`: $[z,P_x]$, size $(109,2)$
- `Profile_Px.u`: phase-mismatch parameter $u$, size $(109,1)$
- `Profile_Px.Ex`: normalized benchmark electric field $E_x$, size $(109,1)$, optional

A typical MATLAB input file can be prepared as:

```matlab
Profile_Px.Px = [z(:), Px(:)];   % 109 x 2
Profile_Px.u  = u * ones(109,1); % 109 x 1
Profile_Px.Ex = Ex(:);           % 109 x 1, optional

save('Efish_vertical.mat', 'Profile_Px');
```

Python example using the MATLAB MAT file:

```python
from ddon import DDON
from scipy.io import loadmat
import numpy as np

mat = loadmat(
    "Efish_vertical.mat",
    squeeze_me=True,
    struct_as_record=False
)

Profile = mat["Profile_Px"]

values = np.asarray(Profile.Px)       # [z, Px], shape (109, 2)
u = np.asarray(Profile.u).reshape(-1)[0]

model = DDON()

E = model.predict(
    values=values,
    u=u
)

print(E)
```

#### CSV input (optional)

```python
from ddon import DDON
import numpy as np

values = np.loadtxt("input.csv", delimiter=",")

model = DDON()

E = model.predict(
    values=values,
    u=-0.35
)

print(E)
```

The verified ONNX model is downloaded automatically on first use and cached locally. This mode requires NumPy, SciPy, and ONNX Runtime.

### 3. Full Python Research Code — To be released

The full TensorFlow research implementation, including the original prediction, evaluation, and analysis workflow, is not included in the current public release and will be released separately in the future.


## Full research implementation

The original TensorFlow research scripts are currently withheld from the public repository. The public web app and lightweight ONNX package remain available for inference.


## Model versioning

DDON uses semantic model versions in the form `vMAJOR.MINOR.PATCH` (for example, `v1.0.0`).

- **Versioned releases** such as `DDON-v1.0.0` are reproducible snapshots and are not overwritten.
- **DDON-WEB** points to the current stable ONNX model used by the Web App and lightweight local package.
- Updating the stable model does not change older versioned releases.
- The current stable model is **v1.0.0**.

## Model file

The current DDON TensorFlow model is available from the DDON release:

**[Download DDON-v1.0.0.h5](https://github.com/ozzzzj/Decoder-DeepONet/releases/download/DDON-v1.0.0/DDON.h5)**

For the original TensorFlow sample, place the model under the `model log` directory.

## Input preprocessing and E-field prediction

1. Interpolate the EFISH profile to the recommended grid:

   $z/z_R = [-50:2:-24 \, -22:1:-16 \, -15:0.5:-1.5 \, -1:0.2:1 \, 1.5:0.5:15 \, 16:1:22 \, 24:2:50]$

   or

   $z/z_R = [-50:1:-2 \, -1:0.2:1 \, 2:1:50]$.

   The first grid is recommended and should be tried first.

2. Normalize the coordinate using $z_{\mathrm{scale}}=50$:

   $z' = (z/z_R)/50$,

   so that $z' \in [-1,1]$. Crop the input EFISH profile if the normalized/scaled range extends beyond this interval.

   Sampling points outside the experimental range may be set to zero. The input range should preferably cover at least $4.2\times\mathrm{FWHM}$ of the normalized EFISH profile.

3. Normalize the measured EFISH profile and, if available, the benchmark electric-field profile:

   $P_x^{\mathrm{norm}} = P_x/\max(P_x)$,

   $E_x^{\mathrm{norm}} = E_x/\max(E_x)$.

4. Estimate the physical phase-mismatch parameter:

   $u=\Delta k \times z_R$.

   **Input the physical value of $u$ directly. Do not normalize $u$ before input.**

   The current DDON model automatically applies the required internal normalization,

   $u' = u/(-1)$,

   before inference.

5. Import the preprocessed data and obtain the DDON prediction.

### MATLAB input structure

| Structure | Field | Description |
|---|---|---|
| `Profile_Px` | `Px` | $[z,\,P_x/\max(P_x)]$: normalized coordinate and normalized EFISH; shape $[109,2]$ |
| `Profile_Px` | $u$ | Physical phase-mismatch parameter $u=\Delta k \times z_R$; the web/local interfaces read the physical value and apply the required DDON normalization internally |
| `Profile_Px` | `Ex` | Optional normalized electric-field benchmark $E_x/\max(E_x)$ for comparison; shape $[109,1]$ |

## Web and ONNX model

The browser and lightweight local modes use an ONNX version of DDON that is numerically verified against the TensorFlow model during the GitHub Actions conversion workflow.

## License and copyright

© 2026 Zhijian Yang. All rights reserved.
