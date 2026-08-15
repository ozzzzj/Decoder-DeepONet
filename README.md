# Decoder-DeepONet (DDON)
Model and code of Decoder DeepONet (DDON) for Electric field reconstruction from EFISH measurements. This model is specifically designed for vertical polarizationed EFISH signals (for a vertically
polarized probe beam).

Related paper and analysis are available at DOI 10.1088/1361-6595/ae413f:

To use the model and code, please cite:
''Yang, Z., Sugeng, E.S., Alicherif, M. and Chng, T.L., 2026. An interpretable operator-learning model for electric field profile reconstruction in discharges based on the EFISH method. Plasma Sources Science and Technology, 35(2), p.025035.''


## Environment recommended (model trained on):
Python 3.10.15
TensorFlow-gpu 2.10.1


## Main user file:
1. 'DeepONet_Resnet_Exp.py' % for script use
2. 'DeepONet_Resnet_Exp.ipynb' % for jupyter editor use

## Script files:
1. 'self_layers.py' %to import some self-defined layers
2. 'PINN_Model_Predict.py' % run the model, output MATLAB .mat file, visualize the prediction
3. 'self_Predict_ModelResult.py' % child file of the 'PINN_Model_Predict.py', including necessary code for Efield prediction
