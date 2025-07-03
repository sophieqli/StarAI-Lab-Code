import torch
import math

n, l = 2,2

#SPECIFY TARGET MARGINALS, R AND C 
R= torch.randn(l) + 1e-7
C= torch.randn(l) + 1e-7
R[0], R[1] = 0.2, 0.8
C[0], C[1] = 0.75, 0.25

#SPECIFY ORIGINAL DISTRIBUTION E 
E = torch.randn(l, l) + 1e-7
E = E/E.sum (dim = (-2, -1), keepdim= True)
E[0][0], E[0][1] = 0.2, 0.4
E[1][0], E[1][1] = 0.1, 0.3

E_rows = E.sum(dim = 1, keepdim = False)
E_cols = E.sum(dim = 0, keepdim = False)
pref_r = torch.zeros(l)
pref_c = torch.zeros(l)
pref_f1 = torch.zeros(l)
pref_f2 = torch.zeros(l)

for i in range(l):
    if i == 0:
        pref_r[0] = R[0]
        pref_c[0] = C[0]
        pref_f1[0] = E_rows[0]
        pref_f2[0] = E_cols[0]
    else:
        pref_r[i] = pref_r[i-1] + R[i]
        pref_c[i] = pref_c[i-1] + C[i]
        pref_f1[i] = pref_f1[i-1] + E_rows[i]
        pref_f2[i] = pref_f2[i-1] + E_cols[i]

def G1(x1):
    if x1 == l: return 1
    fl = math.floor(x1)
    st = 0
    if fl != 0: st = pref_r[fl - 1]
    return st + R[fl]*(x1-fl)

def G2(x2):
    if x2 == l: return 1
    fl = math.floor(x2)
    st = 0
    if fl != 0: st = pref_c[fl - 1]
    return st + C[fl]*(x2-fl)

def F1_inv(u1):
    if u1 <= E_rows[0]: return u1/E_rows[0]
    cnt = 0
    st = u1
    while st > 0:
        st -= E_rows[cnt]
        cnt += 1
        if st < 0: 
          cnt -= 1 
          st += E_rows[cnt]
          break
    return cnt + st/E_rows[cnt]

def F2_inv(u2):
    if u2 <= E_cols[0]: return u2/E_cols[0]
    cnt = 0
    st = u2
    while st > 0 and cnt < l:
        st -= E_cols[cnt]
        cnt += 1
        if st < 0:
          cnt -= 1
          st += E_cols[cnt]
          break

    return cnt + st/E_cols[cnt]

def F(x1, x2):
    m,n = math.floor(x1), math.floor(x2)
    if m == l: m -= 1
    if n == l: n -= 1
    A, B, C_val = E[:m, :n].sum(), E[:m, :n+1].sum(), E[:m+1, :n].sum()
    return A+ (C_val-A)*(x1-m) + (B-A)*(x2-n)+E[m,n]*(x1-m)*(x2-n)

def Cop(u1, u2):
    return F(F1_inv(u1), F2_inv(u2))

def G(x1, x2):
    return Cop(G1(x1), G2(x2))

E_cop = [G(i+1,j+1) - (G(i, j+1) + G(i+1, j) - G(i,j)) for i in range(l) for j in range(l)]

#RETRIEVE THE NEW DISTRIBUTION VIA COPULA
print(E_cop)
