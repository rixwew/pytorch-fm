# Factorization Machine models in PyTorch
  
This package provides a PyTorch implementation of factorization machine models and common datasets in CTR prediction.


## Available Datasets

* [MovieLens Dataset](https://grouplens.org/datasets/movielens)
* [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)
* [Avazu Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction)


## Models

| Model | Reference |
|-------|-----------|
| Logistic Regression | |
| Factorization Machine | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
| Field-aware Factorization Machine | [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) |
| Factorization-Supported Neural Network | [Deep Learning over Multi-field Categorical Data - A Case Study on User Response Prediction](https://arxiv.org/abs/1601.02376) |
| Wide&Deep | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) |
| Attentional Factorization Machine | [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/abs/1708.04617) |
| Neural Factorization Machine | [Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/abs/1708.05027) |
| Field-aware Neural Factorization Machine | [Field-aware Neural Factorization Machine for Click-Through Rate Prediction](https://arxiv.org/abs/1902.09096) |
| Product Neural Network | [Product-based Neural Networks for User Response Prediction](https://arxiv.org/abs/1611.00144) |
| Deep Cross Network | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) |
| DeepFM | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) |
| xDeepFM | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170) |
| AutoInt (Automatic Feature Interaction Model) | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) |

Each model's AUC values are about 0.80 for criteo dataset, and about 0.78 for avazu dataset. (please see [example code](examples/main.py))


## Installation

    pip install torchfm


## API Documentation

https://pytorch-fm.readthedocs.io/


## Licence

MIT
