import torch
import time

for i in range(100):

    x = torch.randn(100, 50, 2048).cuda()
    y = torch.randn(200, 70, 2048).cuda()

    st = time.time()
    z = torch.matmul(x.reshape(-1, 2048), y.reshape(-1, 2048).permute(1, 0))
    z = z.reshape(x.shape[0], x.shape[1], y.shape[0], y.shape[1])
    et = time.time()

    scac = time.time()
    z_hat = torch.einsum('abc,xyc->abxy', x, y)
    ecac = time.time()

    print(torch.sum(z - z_hat), et-st, ecac - scac)