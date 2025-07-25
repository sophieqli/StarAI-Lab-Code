import numpy as np
import torch 

'''
x = np.loadtxt('imgn.val.datarand', delimiter=',')
n, c = x.shape
x = x.reshape(n, 3, 32, 32) #channel first # channel-first: (n, C, H, W)
x = np.transpose(x, (0, 2, 3, 1)) # (n, H, W, C) = (n, 32, 32, 3)
x = x.reshape(n, 16, 2, 16, 2, 3) # breaks into 16x16 grid of 4x4 patches
x = x.transpose(0, 1, 3, 2, 4, 5)
x = x.reshape(n, 16*16, 2, 2, 3) # flatten grid to (n, 64 patches, 4, 4, 3)

R = torch.from_numpy(x[:,:,:,:,0]).long()
G = torch.from_numpy(x[:,:,:,:,1]).long()
B = torch.from_numpy(x[:,:,:,:,2]).long()


Co  = R - B
tmp = B + torch.div(Co, 2, rounding_mode='trunc')
Cg  = G - tmp
Y   = tmp + torch.div(Cg, 2, rounding_mode='trunc')

Co += 256
Cg += 256

x = torch.stack((Y, Co, Cg), dim = -1)

x = x.reshape(n*16*16, 2*2*3) # flatten each patch to vector of size 48
np.savetxt('imgn.val.data', x, fmt='%d', delimiter=',')
'''


x = np.loadtxt('imgn.train.datrand', delimiter=',')
n, c = x.shape
x = x.reshape(n, 3, 32, 32) #channel first # channel-first: (n, C, H, W)
x = np.transpose(x, (0, 2, 3, 1)) # (n, H, W, C) = (n, 32, 32, 3)
x = x.reshape(n, 16, 2, 16, 2, 3) # breaks into 16x16 grid of 4x4 patches
x = x.transpose(0, 1, 3, 2, 4, 5)
x = x.reshape(n, 16*16, 2, 2, 3) # flatten grid to (n, 64 patches, 4, 4, 3)
x = x.reshape(n*16*16, 2*2*3) # flatten each patch to vector of size 48
np.savetxt('imgn.train.data', x, fmt='%d', delimiter=',')
