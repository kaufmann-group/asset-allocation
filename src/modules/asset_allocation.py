import numpy as np
from .qubo import *

class AssetAllocation(Qubo):
    """
    takes returns and covariance
    """
    def __init__(self, returns, covariance, lambda_1=1.0, lambda_2=50.0, lambda_3=10.0, bits_per_asset=6):
        super().__init__()
        self.returns = np.array(returns)
        self.covariance = np.array(covariance)
        self.lambda_1 = lambda_1 # return reward strength - makes the optimizer chase return more aggressively
        self.lambda_2 = lambda_2 # budget constraint strength - forces the raw decoded weights to sum close to 1
        self.lambda_3 = lambda_3 # risk penalty strength - makes the optimizer avoid risky or high variance/covariance allocations
        self.bits_per_asset = bits_per_asset

    """
    builds asset allocation upper triangular qubo
    """
    def build_qubo(self):  
        mu = self.returns
        C = self.covariance

        n = len(mu)
        k = self.bits_per_asset
        size = n * k

        Q = np.zeros((size, size))

        asset_idx = np.zeros(size, dtype=int)
        bit_weights = np.zeros(size)

        for u in range(size):
            asset_idx[u] = u // k
            bit_power = (u % k) + 1
            bit_weights[u] = 2.0 ** (-bit_power)

        for u in range(size):
            i = asset_idx[u]
            a_u = bit_weights[u]

            for v in range(u, size):
                j = asset_idx[v]
                a_v = bit_weights[v]

                variance_quad = self.lambda_3 * C[i, j] * a_u * a_v

                budget_quad = self.lambda_2 * a_u * a_v

                coeff = variance_quad + budget_quad

                if u == v:
                    return_lin = -self.lambda_1 * mu[i] * a_u

                    budget_lin = -2.0 * self.lambda_2 * a_u

                    Q[u, u] = coeff + return_lin + budget_lin

                else:
                    Q[u, v] = 2.0 * coeff

        return Q
    
    """
    decodes solution with fixed point binary encoding
    """
    def decode_solution(self, x):
        n = len(self.returns)
        allocations = []

        for i in range(n):
            bits = x[i * self.bits_per_asset : (i + 1) * self.bits_per_asset]

            value = sum(bit * 2 ** (-j) for j, bit in enumerate(bits, start=1))

            allocations.append(value)

        allocations = np.array(allocations)

        if allocations.sum() != 0:
            allocations = allocations / allocations.sum()

        return allocations