import math
import torch
import torch.nn.functional as F

from torchfm.layer import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron

class LNN(torch.nn.Module):
    """
    A pytorch implementation of LNN layer
    Input shape
        - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
    Output shape
        - 2D tensor with shape:``(batch_size,LNN_dim*embedding_size)``.
    Arguments
        - **in_features** : Embedding of feature.
        - **num_fields**: int.The field size of feature.
        - **LNN_dim**: int.The number of Logarithmic neuron.
        - **bias**: bool.Whether or not use bias in LNN.
    """
    def __init__(self, num_fields, embed_dim, LNN_dim, bias=False):
        super(LNN, self).__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim
        self.LNN_dim = LNN_dim
        self.lnn_output_dim = LNN_dim * embed_dim
        self.weight = torch.nn.Parameter(torch.Tensor(LNN_dim, num_fields))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(LNN_dim, embed_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields, embedding_size)``
        """
        embed_x_abs = torch.abs(x) # Computes the element-wise absolute value of the given input tensor.
        embed_x_afn = torch.add(embed_x_abs, 1e-7)
        # Logarithmic Transformation
        embed_x_log = torch.log1p(embed_x_afn) # torch.log1p and torch.expm1
        lnn_out = torch.matmul(self.weight, embed_x_log)
        if self.bias is not None:
            lnn_out += self.bias
        lnn_exp = torch.expm1(lnn_out)
        output = F.relu(lnn_exp).contiguous().view(-1, self.lnn_output_dim)
        return output






class AdaptiveFactorizationNetwork(torch.nn.Module):
    """
    A pytorch implementation of AFN.

    Reference:
        Cheng W, et al. Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions, 2019.
    """
    def __init__(self, field_dims, embed_dim, LNN_dim, mlp_dims, dropouts):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)    # Linear
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)   # Embedding
        self.LNN_dim = LNN_dim
        self.LNN_output_dim = self.LNN_dim * embed_dim
        self.LNN = LNN(self.num_fields, embed_dim, LNN_dim)
        self.mlp = MultiLayerPerceptron(self.LNN_output_dim, mlp_dims, dropouts[0])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        lnn_out = self.LNN(embed_x)
        x = self.linear(x) + self.mlp(lnn_out)
        return torch.sigmoid(x.squeeze(1))

