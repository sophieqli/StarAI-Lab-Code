import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-7


class LearnableJointCategorical(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()  
        self.l = num_classes
        self.lambdas = nn.Parameter(torch.randn(self.l - 1)) #from 2 to self.l inclusive 

    def getjoints(self, p, lambdas, method = "none"):
        #p is n x l 
        #lambdas is n x n x l
        n, l = p.shape

        #p_u: row marginals for all (i, j) pairs
        #p_v: col marginals for all (i, j) pairs
        pu = p.unsqueeze(1).expand(n,n,l)  #i, j, k same as p(X_i = k)
        pv = p.unsqueeze(0).expand(n,n,l) #i, j, k same as p(X_j = k)
        pu_cumsum = pu.cumsum(dim = -1)
        pv_cumsum = pv.cumsum(dim = -1)

        Pl_prev = None
        for level in range(2, l + 1):
            pu_sum_prev = pu_cumsum[:, :, level-2]
            pv_sum_prev = pv_cumsum[:, :, level-2]

            pu_sum = pu_sum_prev + pu[: , :, level-1] #shape: n x 1
            pv_sum = pv_sum_prev + pv[: , :, level-1] #shape: n x 1

            #acceptable range for lambda
            lower = torch.maximum(torch.full_like(pu_sum_prev, EPS), pu_sum_prev / pu_sum + pv_sum_prev / pv_sum - 1)
            upper = torch.minimum(pu_sum_prev/pu_sum, pv_sum_prev/pv_sum)
            lambda_sc = lambdas[:, :, level-2]
            
            def bounded_param(x, a, b):
                return a + (b - a + EPS) * torch.sigmoid(x) 

            lambda_scaled = bounded_param(lambda_sc, lower, upper)

            #Choose fixed lambda in range (ignore if pre-determined, passed in)
            if method == "midpoint":
                lambda_scaled = (lower + upper) / 2
            elif method == "random":
                lambda_scaled = lower + (upper - lower) * torch.rand(())

            #create the new distribution
            Pl = torch.zeros((n, n, level, level), device=lambdas.device)

            if level == 2: 
                # Base case
                pu0, pu1 = pu[:,:,0], pu[:, :, 1]
                pv0, pv1 = pv[:,:,0], pv[:, :, 1]

                Pl[:, :, 0, 0] = lambda_scaled
                Pl[:, :, 0, 1] = pu0 / (pu0 + pu1) - lambda_scaled
                Pl[:, :, 1, 0] = pv0 / (pv0 + pv1) - lambda_scaled
                Pl[:, :, 1, 1] = 1 -  Pl[:, :, 0, 0] - Pl[:, :, 0, 1] - Pl[:, :, 1, 0]
            else: 
                Pl[:, :, :level-1, :level-1] = Pl_prev * lambda_scaled.unsqueeze(-1).unsqueeze(-1)

                numer_pu = pu[:, :, :level-1]  # (n, n, level-1)
                Pl[:, :, :level-1, -1] = numer_pu/pu_sum.unsqueeze(-1) - lambda_scaled.unsqueeze(-1) * (numer_pu / pu_sum_prev.unsqueeze(-1))

                numer_pv = pv[:, :, :level-1]
                Pl[:, :, -1, :level-1] = numer_pv/pv_sum.unsqueeze(-1) - lambda_scaled.unsqueeze(-1) * (numer_pv / pv_sum_prev.unsqueeze(-1))


                Pl[:, :, -1, -1] = 1 - pu_sum_prev/pu_sum -  pv_sum_prev/pv_sum + lambda_scaled
            
            Pl_prev = Pl
        return Pl_prev

'''
Sample use case
n = 3
l = 4

# Create LearnableJointCategorical model with 4 categories
model = LearnableJointCategorical(num_classes=4)

# Define marginals for 3 variables over 4 categories
p = torch.tensor([
    [0.3, 0.2, 0.1, 0.4],
    [0.4, 0.3, 0.25, 0.05],
    [0.25, 0.25, 0.25, 0.25]
])  # shape (3, 4)

# Define lambdas for each (i, j) pair and each level (l - 1 = 3)
lambdas = torch.tensor([
    [[0.215, 0.1, 0.4],
     [0.3,   0.5, 0.6],
     [0.12,  0.7, 0.1]],

    [[0.2, 0.3, 0.4],
     [0.0, 0.0, 0.0],
     [0.15, 0.6, 0.45]],

    [[0.3, 0.2, 0.1],
     [0.11, 0.5, 0.8],
     [0.0,  0.0, 0.0]]
])  # shape (3, 3, 3)

# Get joint distributions for each variable pair (i, j)
joint = model.getjoints(p, lambdas, method="none")
print(joint)
'''
