import torch

from torchfm.layer import FeaturesLinear, FactorizationMachine, AnovaKernel, FeaturesEmbedding


class HighOrderFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Higher-Order Factorization Machines.

    Reference:
        M Blondel, et al. Higher-Order Factorization Machines, 2016.
    """

    def __init__(self, field_dims, order, embed_dim):
        super().__init__()
        if order < 1:
            raise ValueError(f'invalid order: {order}')
        self.order = order
        self.embed_dim = embed_dim
        self.linear = FeaturesLinear(field_dims)
        if order >= 2:
            self.embedding = FeaturesEmbedding(field_dims, embed_dim * (order - 1))
            self.fm = FactorizationMachine(reduce_sum=True)
        if order >= 3:
            self.kernels = torch.nn.ModuleList([
                AnovaKernel(order=i, reduce_sum=True) for i in range(3, order + 1)
            ])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        y = self.linear(x).squeeze(1)
        if self.order >= 2:
            x = self.embedding(x)
            x_part = x[:, :, :self.embed_dim]
            y += self.fm(x_part).squeeze(1)
            for i in range(self.order - 2):
                x_part = x[:, :, (i + 1) * self.embed_dim: (i + 2) * self.embed_dim]
                y += self.kernels[i](x_part).squeeze(1)
        return torch.sigmoid(y)
