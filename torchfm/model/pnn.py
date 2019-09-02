import torch

from torchfm.layer import FeaturesEmbedding, FeaturesLinear, InnerProductNetwork, \
    OuterProductNetwork, MultiLayerPerceptron


class ProductNeuralNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of inner/outer Product Neural Network.

    Reference:
        Y Qu, et al. Product-based Neural Networks for User Response Prediction, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, method='inner'):
        super().__init__()
        num_fields = len(field_dims)
        self.embed_output_dim = num_fields * embed_dim
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        # linear_signals_weights
        self.linear_signals_weights = torch.nn.ModuleList([
            torch.nn.ParameterList([
                torch.nn.Parameter(torch.randn(embed_dim), requires_grad=True) for _ in range(num_fields)
            ]) for _ in range(mlp_dims[0])
        ])

        # quadratic_signals_weights
        if method == "inner":
            self.quadratic_signals_weights = torch.nn.ModuleList([
                InnerProductNetwork() for _ in range(mlp_dims[0])
            ])
        elif method == "outer": 
            self.quadratic_signals_weights = torch.nn.ModuleList([
                OuterProductNetwork(num_fields, embed_dim) for _ in range(mlp_dims[0])
            ])
        else:
            raise ValueError('unknow product type: ' + method) 

        # MLP layer
        self.bias = torch.nn.Parameter(torch.randn(mlp_dims[0]), requires_grad=True)
        self.mlp = MultiLayerPerceptron(mlp_dims[0], mlp_dims[1:], dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        
        linear_signals_arr = []
        for _, weight_layer in enumerate(self.linear_signals_weights):
            tmp_arr = []
            for j, weight_field in enumerate(weight_layer):
                tmp_arr.append(torch.sum(embed_x[:,j]*weight_field, 1))
            linear_signals_arr.append(sum(tmp_arr).view([-1, 1]))
        linear_signals = torch.cat(linear_signals_arr, 1)

        quadratic_signals_arr = []
        for pn in self.quadratic_signals_weights:
            tmp_pn = pn(embed_x)
            quadratic_signals_arr.append(tmp_pn)
        quadratic_signals = torch.cat(quadratic_signals_arr,1)

        deep_x = linear_signals + quadratic_signals + self.bias
        deep_x = self.mlp(deep_x)
        return torch.sigmoid(deep_x.squeeze(1))
