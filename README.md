# GBDT-HSM: A novel heterogeneous data classification approach combining gradient boosting decision trees and hybrid structure model	

## Training

```
python train.py
```

Modify the main function in `train.py` to select the dataset and model.

```
if __name__ == '__main__':
    RunModel().run('slap', 'gb_hsm', task='classification')
```

## Supported models and datasets

Models:
 - catboost
 - gnn
 - hgnn
 - resgnn
 - bgnn
 - gb_hsm

Datasets:
 - vk_class
 - house_class
 - slap
 - dblp
 - NTU_mvcnn
 - NTU_gvcnn
 - NTU_mvgv
 - ModeNet40_mvcnn
 - ModeNet40_gvcnn
 - ModeNet40_mvgv

## Dependencies

```
catboost==1.0.6
category_encoders==2.2.2
dgl==1.0.2+cu118
dhg==0.9.3
fire==0.5.0
matplotlib==3.7.1
networkx==2.8.4
numpy==1.23.5
omegaconf==2.3.0
pandas==1.5.3
pandas==1.4.4
plotly==5.13.1
scikit_learn==1.2.2
torch==1.13.1
torch_geometric==2.3.1
tqdm==4.64.1
```