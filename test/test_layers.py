import unittest

import numpy as np
import torch

from torchfm.layer import AnovaKernel


class TestAnovaKernel(unittest.TestCase):

    def test_forward_order_2(self):
        batch_size, num_fields, embed_dim = 32, 16, 16
        kernel = AnovaKernel(order=2, reduce_sum=True).eval()
        with torch.no_grad():
            x = torch.FloatTensor(np.random.randn(batch_size, num_fields, embed_dim))
            y_true = 0
            for i in range(num_fields - 1):
                for j in range(i + 1, num_fields):
                    y_true = x[:, i, :] * x[:, j, :] + y_true
            y_true = torch.sum(y_true, dim=1, keepdim=True).numpy()
            y_pred = kernel(x).numpy()
        np.testing.assert_almost_equal(y_pred, y_true, 3)

    def test_forward_order_3(self):
        batch_size, num_fields, embed_dim = 32, 16, 16
        kernel = AnovaKernel(order=3, reduce_sum=True).eval()
        with torch.no_grad():
            x = torch.FloatTensor(np.random.randn(batch_size, num_fields, embed_dim))
            y_true = 0
            for i in range(num_fields - 2):
                for j in range(i + 1, num_fields - 1):
                    for k in range(j + 1, num_fields):
                        y_true = x[:, i, :] * x[:, j, :] * x[:, k, :] + y_true
            y_true = torch.sum(y_true, dim=1, keepdim=True).numpy()
            y_pred = kernel(x).numpy()
        np.testing.assert_almost_equal(y_pred, y_true, 3)

    def test_forward_order_4(self):
        batch_size, num_fields, embed_dim = 32, 16, 16
        kernel = AnovaKernel(order=4, reduce_sum=True).eval()
        with torch.no_grad():
            x = torch.FloatTensor(np.random.randn(batch_size, num_fields, embed_dim))
            y_true = 0
            for i in range(num_fields - 3):
                for j in range(i + 1, num_fields - 2):
                    for k in range(j + 1, num_fields - 1):
                        for l in range(k + 1, num_fields):
                            y_true = x[:, i, :] * x[:, j, :] * x[:, k, :] * x[:, l, :] + y_true
            y_true = torch.sum(y_true, dim=1, keepdim=True).numpy()
            y_pred = kernel(x).numpy()
        np.testing.assert_almost_equal(y_pred, y_true, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
