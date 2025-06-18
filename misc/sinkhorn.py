#Basically alternative to IPFP

import torch

def sinkhorn(M, a, b, num_iters=10, epsilon=1e-8):
    assert torch.all(M > 0), "M must have strictly positive entries."
    n, m = M.shape
    u = torch.ones(n, dtype=M.dtype, device=M.device)
    v = torch.ones(m, dtype=M.dtype, device=M.device)

    for _ in range(num_iters):
        u = a / (M @ v + epsilon)
        v = b / (M.T @ u + epsilon)

    print("u is: ")
    print(u)
    print("v is: ")
    print(v)
    P = torch.diag(u) @ M @ torch.diag(v)
    return P

# Example usage:
n, m = 4, 5

# Random positive matrix M (e.g. exp of random normal)
torch.manual_seed(0)
M = torch.exp(torch.randn(n, m))
print("M starts as: ")
print(M)
# Define valid marginals a and b (non-negative, sum to 1)
a = torch.tensor([0.05, 0.05, 0.85, 0.05])
b = torch.tensor([0.1, 0.1, 0.2, 0.3, 0.3])

# Run Sinkhorn to get P with marginals a and b
P = sinkhorn(M, a, b, num_iters=6)

print("Resulting matrix P:")
print(P)

print("Row sums (shld equal a):", P.sum(dim=1))
print("Column sums (shld equal b):", P.sum(dim=0))

