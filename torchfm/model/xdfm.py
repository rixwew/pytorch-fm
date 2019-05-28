import torch

from torchfm.layer import CompressedInteractionNetwork, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class ExtremeDeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.

    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))
