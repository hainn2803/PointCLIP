import torch
import time
from trainers.sinkhorn import SinkhornAlgorithm
from torch.nn import functional as F
import ot

# for i in range(100):

#     x = torch.randn(100, 50, 2048).cuda()
#     y = torch.randn(200, 70, 2048).cuda()

#     st = time.time()
#     z = torch.matmul(x.reshape(-1, 2048), y.reshape(-1, 2048).permute(1, 0))
#     z = z.reshape(x.shape[0], x.shape[1], y.shape[0], y.shape[1])
#     et = time.time()

#     scac = time.time()
#     z_hat = torch.einsum('abc,xyc->abxy', x, y)
#     ecac = time.time()

#     print(torch.sum(z - z_hat), et-st, ecac - scac)


batch_size = 10
num_views = 4
num_classes = 8
num_prompts = 4
dims = 512

eps = 0.1
max_iter = 100
thresh = 1e-2


for i in range(10):
    image_feat = torch.randn(batch_size, num_views, dims)
    text_feat = torch.randn(num_classes, num_prompts, dims)

    text_feat_1 = text_feat / text_feat.norm(dim=-1, keepdim=True)
    image_feat_1 = image_feat / image_feat.norm(dim=-1, keepdim=True)

    batch_size = image_feat.shape[0]
    num_views = image_feat.shape[1]
    dims = image_feat.shape[2]
    num_classes = text_feat.shape[0]
    num_prompts = text_feat.shape[1]
    sim = torch.einsum('bvd,cnd->bcvn', image_feat, text_feat).contiguous() # shape == (batch_size, num_classes, num_views, num_prompts)
    sim = sim.view(batch_size * num_classes, num_views, num_prompts)
    wdist = 1.0 - sim

    p = torch.zeros(batch_size * num_classes, num_views, dtype=wdist.dtype, device=wdist.device).fill_(1. / num_views)
    q = torch.zeros(batch_size * num_classes, num_prompts, dtype=wdist.dtype, device=wdist.device).fill_(1. / num_prompts)
    sinkhorn_solver = SinkhornAlgorithm(epsilon=eps, iterations=max_iter, threshold=thresh)
    T = sinkhorn_solver(p, q, wdist) # shape == (batch_size * num_classes, num_views, num_prompts)



    # def Sinkhorn(K, u, v, max_iter):
    #     r = torch.ones_like(u)
    #     c = torch.ones_like(v)
    #     for i in range(max_iter):
    #         r0 = r
    #         r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
    #         c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
    #         err = (r - r0).abs().mean()
    #         if err.item() < thresh:
    #             break
    #     T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    #     return T
    # T_hat = Sinkhorn(wdist, p, q, max_iter)
    # print(torch.mean(T - T_hat))

    total_error = 0
    for i in range(len(p)):
        T_pot = ot.sinkhorn(a=p[i], b=q[i], M=wdist[i], reg=eps, numItermax=max_iter, stopThr=thresh, method="sinkhorn_log")
        # print(T_pot)
        total_error += torch.mean(T[i] - T_pot)
    
    print(total_error)