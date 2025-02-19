import torch 

model_adapter_path = "output/PROMPT_LR_0.08/ADAPTER_LR_0.01/NUM_SHOTS_16/NUM_PROMPTS_4/PLOT_learnable_prompt_logit_scale_on/PointCLIP_FS/rn101/modelnet40/adapter/model-best.pth.tar"

saved_dict_adapter = torch.load(model_adapter_path)

print(saved_dict_adapter["epoch"])


model_prompt_path = "output/PROMPT_LR_0.08/ADAPTER_LR_0.01/NUM_SHOTS_16/NUM_PROMPTS_4/PLOT_learnable_prompt_logit_scale_on/PointCLIP_FS/rn101/modelnet40/prompt_learner/model-best.pth.tar"

saved_dict_prompt = torch.load(model_prompt_path)

print(saved_dict_prompt["epoch"])