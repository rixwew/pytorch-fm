# Factorization Machine models in PyTorch
  
This package provides a PyTorch implementation of factorization machine models and common datasets in CTR prediction.


## Available Datasets

* [MovieLens Dataset](https://grouplens.org/datasets/movielens)
* [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)
* [Avazu Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)


## Available Models

| Model | Reference |
|-------|-----------|
| Logistic Regression | |
| Factorization Machine | [S Rendle, Factorization Machines, 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
| Field-aware Factorization Machine | [Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) |
| Factorization-Supported Neural Network | [W Zhang, et al. Deep Learning over Multi-field Categorical Data - A Case Study on User Response Prediction, 2016.](https://arxiv.org/abs/1601.02376) |
| Wide&Deep | [HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.](https://arxiv.org/abs/1606.07792) |
| Attentional Factorization Machine | [J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.](https://arxiv.org/abs/1708.04617) |
| Neural Factorization Machine | [X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.](https://arxiv.org/abs/1708.05027) |
| Neural Collaborative Filtering | [X He, et al. Neural Collaborative Filtering, 2017.](https://arxiv.org/abs/1708.05031) |
| Field-aware Neural Factorization Machine | [L Zhang, et al. Field-aware Neural Factorization Machine for Click-Through Rate Prediction, 2019.](https://arxiv.org/abs/1902.09096) |
| Product Neural Network | [Y Qu, et al. Product-based Neural Networks for User Response Prediction, 2016.](https://arxiv.org/abs/1611.00144) |
| Deep Cross Network | [R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.](https://arxiv.org/abs/1708.05123) |
| DeepFM | [H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.](https://arxiv.org/abs/1703.04247) |
| xDeepFM | [J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.](https://arxiv.org/abs/1803.05170) |
| AutoInt (Automatic Feature Interaction Model) | [W Song, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks, 2018.](https://arxiv.org/abs/1810.11921) |
| AFN(AdaptiveFactorizationNetwork Model) | [Cheng W, et al. Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions, AAAI'20.](https://arxiv.org/pdf/1909.03276.pdf) |

Each model's AUC values are about 0.80 for criteo dataset, and about 0.78 for avazu dataset. (please see [example code](examples/main.py))


## Installation

    pip install torchfm


## API Documentation

https://rixwew.github.io/pytorch-fm


## Licence

MIT
