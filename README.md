## DILPA-DTA
**DILPA-DTA: A Dilated-convolution Interaction and Laplacian Position-Aware method for Predicting Drug-Target Binding Affinity**

## Description  
DILPA-DTA is a novel deep learning framework designed for Drug-Target Binding Affinity (DTA) prediction. It integrates dilated convolutional modules, Laplacian position-aware encoding, and interaction-aware attention mechanisms to enhance multimodal representation learning for both drugs and proteins.

## System Requirements  
```bash
torch>=1.10.0  
dgl>=0.8.2  
dgllife>=0.2.9  
numpy>=1.21.0  
scikit-learn>=1.0.2  
pandas>=1.3.3  
rdkit~=2021.09.4  
yacs~=0.1.8  
prettytable>=3.0.0  
tqdm>=4.62.3  
```

## Training  
```bash
python main.py 
```


## Citation  
If you use DILPA-DTI in your research, please cite the following paper:  
```
@article{your2025dilpa,
  title={DILPA-DTI: A Dilated and Interaction-Laplacian Position-Aware Network for Drug–Target Interaction Prediction},
  author={Your Name and Collaborators},
  journal={Journal of Biomedical Informatics},
  year={2025},
  publisher={Elsevier}
}
```
