import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableJointCategorical(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()  
        self.l = num_classes
        self.lambdas = nn.Parameter(torch.randn(self.l - 1)) #from 2 to self.l inclusive 

    def getjoints(self, p_u, p_v, lambdas, method = "none"):
        assert p_u.shape == p_v.shape
        assert p_u.shape[0] == self.l

        Pl_prev = None
        for level in range(2, self.l + 1):
            
            pu_sum_prev = p_u[:level-1].sum()
            pv_sum_prev = p_v[:level-1].sum()

            pu_sum = pu_sum_prev + p_u[level-1]
            pv_sum = pv_sum_prev + p_v[level-1]

            #acceptable range for lambda
            lower = max(0, pu_sum_prev / pu_sum + pv_sum_prev / pv_sum - 1)
            upper = min(pu_sum_prev / pu_sum, pv_sum_prev / pv_sum)

            lambda_scaled = lambdas[level-2]
            lower = torch.tensor(lower, dtype=lambdas.dtype, device=lambdas.device)
            upper = torch.tensor(upper, dtype=lambdas.dtype, device=lambdas.device)
            
            def bounded_param(x, a, b):
                #return a + (b - a) * torch.sigmoid(x) 
                #empirically, it seems tanh makes loss go down much faster
                return a + (b-a) * 0.5 * (torch.tanh(x) + 1)

            lambda_scaled = bounded_param(lambda_scaled, lower, upper)

            #Choose fixed lambda in range (ignore if pre-determined, passed in)
            if method == "midpoint":
                lambda_scaled = (lower + upper) / 2
            elif method == "random":
                lambda_scaled = lower + (upper - lower) * torch.rand(())

            #create the new distribution
            Pl = torch.zeros((level, level), device=lambdas.device)

            if level == 2: 
                # Base case
                Pl[0,0] = lambda_scaled
                Pl[0,1] = p_u[0] / (p_u[0] + p_u[1]) - lambda_scaled
                Pl[1,0] = p_v[0] / (p_v[0] + p_v[1]) - lambda_scaled
                Pl[1,1] = 1 - Pl[0,0] - Pl[0,1] - Pl[1,0]
            else: 
                Pl[:level-1, :level-1] = Pl_prev * lambda_scaled 

                for i in range(level-1): 
                    Pl[i, -1] = p_u[i]/pu_sum - lambda_scaled * (p_u[i] / pu_sum_prev)
                for j in range(level-1): 
                    Pl[-1, j] = p_v[j]/pv_sum - lambda_scaled * (p_v[j] / pv_sum_prev)
                Pl[-1, -1] = 1 - pu_sum_prev/pu_sum -  pv_sum_prev/pv_sum + lambda_scaled
            
            Pl_prev = Pl
        return Pl_prev



#SET WHATEVER MARGINALS U WANT HERE
'''
model = LearnableJointCategorical(num_classes=4)
p_u = torch.tensor([0.3, 0.2, 0.1, 0.4])
p_v = torch.tensor([0.4, 0.3, 0.25, 0.05])
lambdas = torch.tensor([0.215, 0.68, 0.56])
print("u(row)-marginals: ", p_u)
print("v(col)-marginals: ", p_v)
joint = model.getjoints(p_u, p_v, lambdas, method="none")

print(joint)
'''
