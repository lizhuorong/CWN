# Intelligent Grading of Diabetic Retinopathy Based on Category Weighted Network

-------

We provide the PyTorch implements.
![cwn](https://github.com/lizhuorong/CWN/assets/29669415/7defa746-8dd7-412e-8346-5fcad0098cc3)
      
## Getting Started

### Install requirments
``` bash
torch==0.4.1
torchvision==0.2.0
numpy==1.21.5
Pillow==9.5.0
prefetch-generator==1.0.3
opencv_python==4.5.5.62
matplotlib==3.3.0
```
### Running Training/Testing

Remember to check/change the data path and weight path

```bash
python train.py efficientnet ModelName
python test.py efficientnet ModelName.pkl
```

### Citation
```
@article{HAN2023106408,
title = {Category weighted network and relation weighted label for diabetic retinopathy screening},
journal = {Computers in Biology and Medicine},
volume = {152},
pages = {106408},
year = {2023},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2022.106408},
url = {https://www.sciencedirect.com/science/article/pii/S0010482522011167},
author = {Zhike Han and Bin Yang and Shuiguang Deng and Zhuorong Li and Zhou Tong}
}
```  
