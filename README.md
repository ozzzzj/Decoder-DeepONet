# Decoder-DeepONet (DDON)
Model and code of Decoder DeepONet (DDON) for Electric field reconstruction from EFISH measurements. This model is specifically designed for vertically polarized EFISH signals (for a vertically
polarized probe beam).

To use the model and code, please cite:

''Yang, Z., Sugeng, E.S., Alicherif, M. and Chng, T.L., 2026. An interpretable operator-learning model for electric field profile reconstruction in discharges based on the EFISH method. Plasma Sources Science and Technology, 35(2), p.025035.''

Related paper and analysis are available at DOI 10.1088/1361-6595/ae413f.

**Note** Scripts and model will be available soon...  

## Environment recommended (model trained on):
- Python 3.10.15
- TensorFlow-gpu 2.10.1

## Main user file:
1. 'DeepONet_Resnet_Exp.py' % for script use
2. 'DeepONet_Resnet_Exp.ipynb' % for jupyter editor use

## Script files:
1. 'self_layers.py' %to import some self-defined layers
2. 'PINN_Model_Predict.py' % run the model, output MATLAB .mat file, visualize the prediction
3. 'self_Predict_ModelResult.py' % child file of the 'PINN_Model_Predict.py', including necessary code for Efield prediction

## Model file (DDON) and model description
- Please download the DDON model via the release page for use (put it under the dir model log) or via:
  https://github.com/ozzzzj/Decoder-DeepONet/releases/download/DDON/20260520_09-39_AM+model.Epoch-27_Loss-0.000404+MSE-0.000244+Batsize-.512.h5

## Instructions to use the model for Efield prediction:
1. To use the model, please first interpolate the EFISH file to the following grid via MATLAB:
   $z/z_R = [-50:2:-24 \, -22:1:-16 \, -15:0.5:-1.5 \, -1:0.2:1 \, 1.5:0.5:15 \, 16:1:22 \, 24:2:50]$;
   
   or
   
   $z/z_R = [-50:1:-2 \, -1:0.2:1 \, 2:1:50]$;

   **Note**: The first grid point is recommended and should be tried first, as it may always show good predictions; otherwise, try the second to see if better results can be gotten.

2. Then further normalize the $z/z_R$ by dividing $z_\mathrm{scale} = 50$, then the input grid should be:
   $z^\prime = z/z_R/50 = [-50:2:-24 \, -22:1:-16 \, -15:0.5:-1.5 \, -1:0.2:1 \, 1.5:0.5:15 \, 16:1:22 24:2:50]/50$;
   
   or
   
   $z^\prime = z/z_R/50 = [-50:1:-2 \, -1:0.2:1 \, 2:1:50]/50$;
   
   **Note 1**: $z^\prime \in [-1,1]$; crop the input EFISH profile if the normalized and scaled range (z/z_R/50) goes beyond this range.

   **Note 2**: The sampling grid outside your experiment range could be set to zero, as the DDON accepts zero input outside the key feature range. For how to quantify the key range, please refer to our paper. Please ensure the input range is no less than 4.2*FWHM of your input EFISH profile (normalized), although sometimes a smaller sampling range than the criterion also works.

3. Normalize the measured EFISH profile (along the laser propagation axis, $z$) by its maximum:
   $P_\mathrm{norm}(z) = P(z)/P_\mathrm{max}$

4. Estimate the phase mismatch value $u$ through the wave-factor mismatch $\Delta k$ and Rayleigh range $z_\mathrm{R}$, and normalize it as input:
   
   $u^\prime$ = $\Delta k \cdot z_\mathrm{R}$/-0.068.
   
   **Note**: -0.068 is the max $u$ value from the training dataset.

6. Import the MAT file as structure files and obtain the prediction. Or you can modify the code to fit your data structure as well.
   
   The MATLAB file structure is:
   
    		            ----- $P_x$ ---> $[z,P_x]$	(experiment-measured EFISH and normalized z^\prime; dim: [109,2])
   
    Profile_Px -->> ----- $u$             	(the phase mismatch value along $z$; dim: [109,1])
   
    		            ----- $E_x$			(the phase mismatch value along $z$; dim: [109,1])

