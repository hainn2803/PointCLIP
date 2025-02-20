import torch
import torch.nn as nn
from torch.nn import functional as F


# model_adapter_path = "output/PROMPT_LR_0.08/ADAPTER_LR_0.01/NUM_SHOTS_16/NUM_PROMPTS_4/PLOT_learnable_prompt_logit_scale_on/PointCLIP_FS/rn101/modelnet40/adapter/model-best.pth.tar"

# saved_dict_adapter = torch.load(model_adapter_path)

# print(saved_dict_adapter["epoch"])


# model_prompt_path = "output/PROMPT_LR_0.08/ADAPTER_LR_0.01/NUM_SHOTS_16/NUM_PROMPTS_4/PLOT_learnable_prompt_logit_scale_on/PointCLIP_FS/rn101/modelnet40/prompt_learner/model-best.pth.tar"

# saved_dict_prompt = torch.load(model_prompt_path)

# print(saved_dict_prompt["epoch"])




def smooth_loss(pred, gold):
    eps = 0.2
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    print(gold)
    print(one_hot)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

    print(one_hot)

    one_hot_2 = one_hot * (1 - eps) + eps / (n_class)

    print(one_hot_2)

    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()

    return loss

# Example usage
batch_size = 4
num_classes = 5

pred = torch.randn(batch_size, num_classes)  # Simulated predictions
gold = torch.randint(0, num_classes, (batch_size,))  # Randomly generated class labels

l1 = smooth_loss(pred, gold)
print("Loss l1:", l1)

l2 = F.cross_entropy(input=pred, target=gold, label_smoothing=0.2)

print("Loss l2:", l2)
