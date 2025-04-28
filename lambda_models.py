import torch
from torch import nn
import networkx as nx
import math
import utils
import random

#hello :) 

from tqdm import tqdm

EPS=1e-7

from lambda_param import LearnableJointCategorical

class MoAT(nn.Module):

    ########################################
    ###     Parameter Initialization     ###
    ########################################

    def __init__(self, n, x, num_classes=2, device='cpu'):
        super().__init__()

        self.n = n
        self.l = num_classes
        self.lambdas = torch.zeros((n, n, self.l - 1))
        self.catmodel = LearnableJointCategorical(num_classes=num_classes)

        print('initializing params ...')
        with torch.no_grad():
            m = x.shape[0]  # samples

            # estimate marginals from data
            x = x.to(device)

            # pairwise marginals
            E = torch.zeros(n, n, self.l, self.l).to(device)  # SL
            block_size = (2 ** 30) // (n * n * self.l * self.l)
            for block_idx in tqdm(range(0, m, block_size)):
                block_size_ = min(block_size, m - block_idx)
                x_block = x[block_idx:block_idx + block_size_]
                x_2d = torch.zeros(block_size_, n, n, self.l, self.l).to(device)
                for k in range(block_size_):
                    for i in range(n):
                        for j in range(n):
                            a = x_block[k, i].item()
                            b = x_block[k, j].item()
                            x_2d[k, i, j, a, b] = 1  # X_i = a, X_j = b on this sample

                E += torch.sum(x_2d, dim=0)  # shape: [n, n, l, l]

            E = (E + 1.0) / float(m + 2) 

            E = E.to('cpu')
            E_compress = E.clone()

            # univariate marginals initialization
            V = torch.zeros(n, self.l)

            for i in range(self.l):
                cnt = torch.sum(x == i, dim=0)  # count how many times i appears in each column
                V[:, i] = (cnt + 1) / (float(m) + self.l) #CHANGE INITIALIZATION BACK LATER
                #V[:, i] = (cnt) / (float(m))
            V_compress = V.clone()

            E_compress = torch.clamp(E_compress, min=EPS)  # to avoid log(0) or div by 0
            #E_compress = E_compress / (E_compress.sum(dim=(-2, -1), keepdim=True) + EPS)  # normalize joint by basically 1

            print("initial marginals, based on frequencies: ")
            print(V_compress)

            print("initial joints, based on frequencies: ")
            print(E_compress)

            #lambda initialization (ratio within lower to upper bound interval)

            for i in range(self.l): 
                for j in range(self.l):
                    for k in range(2, self.l + 1): 
                        
                        numer = torch.sum(E_compress[i, j, :k-1, :k-1], dim=(-2, -1))
                        denom = (torch.sum(E_compress[i, j, :k, :k], dim=(-2, -1)))
                        val = numer / (denom + EPS)
                        
                        pi_sum_prev = V_compress[i][:k-1].sum()
                        pj_sum_prev = V_compress[j][:k-1].sum() 
                        
                        pi_sum = pi_sum_prev + V_compress[i][k-1]
                        pj_sum = pj_sum_prev + V_compress[j][k-1]

                        lower = max(EPS, pi_sum_prev / pi_sum + pj_sum_prev / pj_sum - 1)
                        upper = min(pi_sum_prev / pi_sum, pj_sum_prev / pj_sum)
                        print(i, j, upper, lower)

                        self.lambdas[i, j, k-2] = (val-lower)/(upper-lower)

            print("the ratios are ")
            print(self.lambdas)

            # logit for unconstrained parameter learning (inverse sigmoid)
            V_compress = torch.special.logit(V_compress)
            E_compress = torch.special.logit(E_compress)
            #not learnable rn b/c of lambdas: E_compress = torch.special.logit(E_compress)

            print('computing MI ...')
            E_new = torch.maximum(E, torch.ones(1) * EPS).to(device) #Pairwise joints
            left = V.unsqueeze(1).unsqueeze(-1)  # shape: [n, 1, l, 1]
            right = V.unsqueeze(1).unsqueeze(0)  # shape: [1, n, 1, l]
            # gives tensor n,n,l,l -> pairwise mutual info distributions (assuming independence for baseline comparison)
            V_new = torch.maximum(left * right, torch.ones(1) * EPS).to(device)

            print(E_new)
            print(V_new)
            MI = torch.sum(torch.sum(E_new * torch.log(E_new / V_new), dim=-1), dim=-1)
            MI += EPS

            #ENSURE IN RANGE OF 0-1
            MI_max = (MI.max()+EPS).unsqueeze(0).unsqueeze(0)
            if MI_max >= 1: MI = MI / (MI_max)

            MI = torch.special.logit(MI)

        # W stores the edge weights
        self.W = nn.Parameter(MI, requires_grad=True)
        # E_compress are no longer parameters -- they're determined by marginals and lambdas

        #make lambdas unconstrained 
        self.lambdas = torch.special.logit(self.lambdas) 
        self.lambdas = nn.Parameter(self.lambdas, requires_grad=True)
        self.E_compress = E_compress
        self.V_compress = nn.Parameter(V_compress, requires_grad=True)

        print("Weights init to")
        print(self.W)
        print("Lambdas init to ")
        print(self.lambdas)
        print("E compress init to ")
        print(self.E_compress)
        print("V compress init to ")
        print(self.V_compress)


    ########################################
    ###             Inference            ###
    ########################################

    def forward(self, x):
        batch_size, d = x.shape
        n, W, V_compress, E_compress = self.n, self.W, torch.sigmoid(self.V_compress),torch.sigmoid(self.E_compress) #convert back to raw probabilities

        #must normalize V_compress so all margs add to 1
        print("before norm ")
        print(V_compress)
        V_comp_norm = torch.sum(V_compress, dim = 1, keepdim = True) #sum each row 
        V_compress = V_compress / V_comp_norm
        print("after norm ")
        print(V_compress)

        #both V_compress and E_compress are fine here (recover initial param)
        E_computed = torch.zeros_like(E_compress)  # no gradients attached here

        for i in range(n):
            for j in range(n):
                E_computed[i, j, :, :] = self.catmodel.getjoints(
                    V_compress[i], V_compress[j], self.lambdas[i, j, :], method="none"
                )

        self.E_compress = E_computed.detach()  # avoid storing something that needs gradients
        print("E compress inside fwd loop ")
        print(self.E_compress)

        E = E_computed  # this one will keep the graph for backprop
        V  = V_compress.clone()

        E_mask = (1.0 - torch.diag(torch.ones(n)).unsqueeze(-1).unsqueeze(-1)).to(E.device) #broadcasts to 1, 1, n, n diag matrix (zeroes out Xi, Xi)
        E = E * E_mask
        E=torch.clamp(E,0,1)
        print("E is ")
        print(E)

        W = torch.sigmoid(W)
        W = torch.tril(W, diagonal=-1)
        W = torch.transpose(W, 0, 1) + W

        # det(principal minor of L_0) gives the normalizing factor over the spanning trees
        L_0 = -W + torch.diag_embed(torch.sum(W, dim=1))

        Pr = V[torch.arange(n).unsqueeze(0), x]

        P = E[torch.arange(n).unsqueeze(0).unsqueeze(-1),
                torch.arange(n).unsqueeze(0).unsqueeze(0),
                x.unsqueeze(-1),
                x.unsqueeze(1)] # E[i, j, x[idx, i], x[idx, j]]

        print("PROB P")
        print(P)

        P = P / torch.matmul(Pr.unsqueeze(2), Pr.unsqueeze(1)) # P: bath_size * n * n

        W = W.unsqueeze(0) # W: 1 * n * n; W * P: batch_size * n * n
        L = -W * P + torch.diag_embed(torch.sum(W * P, dim=2))  # L: batch_size * n * n

        y = torch.sum(torch.log(Pr), dim=1) + torch.logdet(L[:, 1:, 1:]) - torch.logdet(L_0[1:, 1:])
        print("3 components ")
        print(torch.sum(torch.log(Pr), dim=1) )
        print(torch.logdet(L[:, 1:, 1:]), torch.logdet(L_0[1:, 1:]))

        if y[y != y].shape[0] != 0:
            print("NaN!")
            exit(0)
        #likelihood for each sample
        print(" --> forward, returning ", y)
        return y

    ########################################
    ### Methods for Sampling Experiments ###
    ########################################

    # can be repurposed to just return samples!

    # spanning tree distribution normalization constant
    def log_Z(self):
        W = torch.sigmoid(self.W)
        W = torch.tril(W, diagonal=-1)
        W = torch.transpose(W, 0, 1) + W
        L_0 = -W + torch.diag_embed(torch.sum(W, dim=1))
        return torch.logdet(L_0[1:, 1:]).item()

    def sample_spanning_tree(self,G,W):
        st=nx.algorithms.random_spanning_tree(G,weight='weight').edges
        parents=utils.get_parents(st,self.n)

        # unnormalized weight of sampled spanning tree
        # log_w=1
        # for (i,j) in st:
        #     log_w+=math.log(W[i][j])
        # normalized weight of spanning tree
        # log_wst=log_w - Z

        return st, parents #, log_wst

    # returns V,E,W after projecting to right space
    def get_processed_parameters(self):
        n, V_compress, E_compress = self.n, torch.sigmoid(self.V_compress),torch.sigmoid(self.E_compress)
        upper_bound = torch.minimum(V_compress.unsqueeze(0), V_compress.unsqueeze(-1))
        lower_bound = torch.maximum(V_compress.unsqueeze(-1) + V_compress.unsqueeze(0) - 1.0,
                        torch.zeros(E_compress.shape).to(V_compress.device)+EPS)

        E_compress = E_compress * ((upper_bound - lower_bound)+EPS) + lower_bound

        V1 = V_compress # n
        V0 = 1 - V_compress
        V = torch.stack((V0, V1), dim=1) # n * 2
        V=V.cpu().detach().numpy()

        E11 = E_compress # n * n
        E01 = V1.unsqueeze(0) - E11
        E10 = V1.unsqueeze(-1) - E11
        E00 = 1 - E01 - E10 - E11
        E0 = torch.stack((E00, E01), dim=-1) # n * n * 2
        E1 = torch.stack((E10, E11), dim=-1) # n * n * 2
        E = torch.stack((E0, E1), dim=2)
        E_mask = (1.0 - torch.diag(torch.ones(n)).unsqueeze(-1).unsqueeze(-1)).to(E.device)
        E = E * E_mask
        E=E.cpu().detach().numpy()

        W = torch.sigmoid(self.W)
        W = torch.tril(W, diagonal=-1)
        W = torch.transpose(W, 0, 1) + W

        return V,E,W


    def get_true_marginals(self,evidence):
        n=self.n
        true_marginals=[1 for i in range(n)]
        for i in range(n):
            if evidence[i]!=-1:
                true_marginals[i]=evidence[i]
                continue
            evidence[i]=0
            data=utils.generate_marginal_terms(n,evidence)
            p_0=torch.sum(torch.exp(self.forward(data))).item()
            evidence[i]=1
            data=utils.generate_marginal_terms(n,evidence)
            p_1=torch.sum(torch.exp(self.forward(data))).item()

            true_marginals[i]=p_1/(p_0+p_1)
            evidence[i]=-1
        return true_marginals

    def get_importance_samples(self,evidence,num_samples=1):
        num_missing=evidence.count(-1)
        n=self.n
        V,E,W=self.get_processed_parameters()

        true_marginals=self.get_true_marginals(evidence)

        marginals=[0 for i in range(n)]
        norm=0

        klds=[]
        wts=[]
        G = nx.Graph()
        for i in range(self.n):
            for j in range(i+1,self.n):
                G.add_edge(i,j,weight=W[i][j].item())

        data=utils.generate_marginal_terms(n,evidence)
        m_e=torch.sum(torch.exp(self.forward(data))).item()

        for it in range(num_samples):
            st, parents = self.sample_spanning_tree(G,W)
            res,wt=utils.sample_from_tree_factor_autoregressive(n,parents,E,V,evidence)
            for i in range(n):
                if res[i]:
                    marginals[i]+=wt
            norm+=wt
            wts.append(wt/m_e)
            approximate_marginals=[marginals[i]/norm for i in range(n)]
            kld=utils.kld(true_marginals,approximate_marginals)/num_missing
            klds.append(kld)

        return klds,wts

    def get_collapsed_importance_samples(self,evidence,num_samples=1):
        num_missing=evidence.count(-1)
        n=self.n
        V,E,W=self.get_processed_parameters()

        true_marginals=self.get_true_marginals(evidence)

        marginals=[0 for i in range(n)]
        norm=0

        klds=[]
        wts=[]
        G = nx.Graph()
        for i in range(self.n):
            for j in range(i+1,self.n):
                G.add_edge(i,j,weight=W[i][j].item())

        data=utils.generate_marginal_terms(n,evidence)
        m_e=torch.sum(torch.exp(self.forward(data))).item()

        for it in range(num_samples):
            st, parents = self.sample_spanning_tree(G,W)
            wt,marginal_vector=utils.sample_from_tree_parallel(n,parents,E,V,evidence)
            for i in range(n):
                marginals[i]+=wt*max(0,min(1,marginal_vector[i]))
            norm+=wt
            wts.append(wt/m_e)
            approximate_marginals=[marginals[i]/norm for i in range(n)]
            kld=utils.kld(true_marginals,approximate_marginals)/num_missing
            klds.append(kld)


        return klds,wts

    def get_gibbs_samples(self,evidence,burn_in=10,num_samples=1):
        num_missing=evidence.count(-1)
        n=self.n
        V,E,W=self.get_processed_parameters()

        true_marginals=self.get_true_marginals(evidence)

        marginals=[0 for i in range(n)]
        norm=0

        cur=evidence.copy()
        for i in range(n):
            if cur[i]==-1:
                cur[i]=random.randint(0, 1)

        idx=-1
        klds=[]
        for it in range(burn_in+num_samples):
            evi=cur.copy()
            for idx in range(n):
                if evidence[idx]!=-1:
                    continue
                evi[idx]=-1
                # note that this generates the 0 term first followed by the 1 term
                data=utils.generate_marginal_terms(n,evi)
                p=torch.exp(self.forward(data))
                cur[idx]=0 if random.uniform(0, 1)<=p[0].item()/(p[0].item()+p[1].item()) else 1
                evi[idx]=cur[idx]

            if it<burn_in:
                continue

            for i in range(n):
                if cur[i]:
                    marginals[i]+=1
            norm+=1

            if it >= burn_in:
                approximate_marginals=[marginals[i]/norm for i in range(n)]
                kld=utils.kld(true_marginals,approximate_marginals)/num_missing
                klds.append(kld)

        return klds

