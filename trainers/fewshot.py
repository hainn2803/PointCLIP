import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from trainers.mv_utils_fs import PCViews

from trainers.sinkhorn import SinkhornAlgorithm

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

import ot


CUSTOM_TEMPLATES = {
    'ModelNet40': 'point cloud of a big {}.'
}

# source: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/util.py
def smooth_loss(pred, gold):
    eps = 0.2
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()
    return loss

class BatchNormPoint(nn.Module):
    def __init__(self, feat_size, sync_bn=False):
        super().__init__()
        self.feat_size = feat_size
        self.sync_bn=sync_bn
        if self.sync_bn:
            self.bn = BatchNorm2dSync(feat_size)
        else:
            self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        if self.sync_bn:
            # 4d input for BatchNorm2dSync
            x = x.view(s1 * s2, self.feat_size, 1, 1)
            x = self.bn(x)
        else:
            x = x.view(s1 * s2, self.feat_size)
            x = self.bn(x)
        return x.view(s1, s2, s3)

def load_clip_to_cpu(cfg, model_path="clip/pretrained_weights/RN101.pt"):
    if model_path is None:
        backbone_name = cfg.MODEL.BACKBONE.NAME
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model



class Textual_Encoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames] # len(prompts) == NUM_CLASSES
        prompts = torch.cat([clip.tokenize(p) for p in prompts]) # shape == torch.Size([NUM_CLASSES, 77])
        prompts = prompts.cuda()
        text_feat = self.clip_model.encode_text(prompts) # shape == torch.Size([NUM_CLASSES, 512])
        text_feat = text_feat.repeat(1, self.cfg.MODEL.PROJECT.NUM_VIEWS) # shape == torch.Size([NUM_CLASSES, 512 * NUM_VIEWS])
        return text_feat


###### ADDED ######
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PLOT.N_CTX
        ctx_init = cfg.TRAINER.PLOT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.num_prompts = cfg.TRAINER.PLOT.NUM_PROMPTS
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.PLOT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.num_prompts, n_ctx, ctx_dim, dtype=dtype) 
            nn.init.normal_(ctx_vectors, std=0.02)   # define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)    

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        

        classnames = [name.replace("_", " ") for name in classnames]   
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # shape == (n_cls, ?)
        tokenized_prompts = tokenized_prompts.repeat(self.num_prompts, 1)  # shape == (n_cls * self.num_prompts, ?)
        # tokenized_prompts3.view(3,100,77)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 
        

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PLOT.CLASS_TOKEN_POSITION


    def forward(self):
       
        ctx = self.ctx # shape == (self.num_prompts, n_ctx, ctx_dim)
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1) 
        # ctx.shape == (self.n_cls, self.num_prompts, n_ctx, ctx_dim)
        ctx = ctx.permute(1, 0, 2, 3) # ctx.shape == (self.num_prompts, self.n_cls, n_ctx, ctx_dim)
        ctx = ctx.contiguous().view(self.num_prompts*self.n_cls,self.n_ctx,ctx.shape[3])

        prefix = self.token_prefix # tokenized_prompts
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (self.num_prompts*self.n_cls, 1, dim)
                    ctx,     # (self.num_prompts*self.n_cls, n_ctx, dim)
                    suffix,  # (self.num_prompts*self.n_cls, *, dim)
                ],
                dim=1,
            )
            # print(prefix.shape, ctx.shape, suffix.shape, prompts.shape)
            # torch.Size([160, 1, 512]) torch.Size([160, 16, 512]) torch.Size([160, 60, 512]) torch.Size([160, 77, 512])

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
###### ADDED ######



class PointCLIP_Model(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        
        # Encoders from CLIP
        self.visual_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Multi-view projection
        self.num_views = cfg.MODEL.PROJECT.NUM_VIEWS
        pc_views = PCViews()
        self.get_img = pc_views.get_img

        # inter-view Adapter
        self.adapter = Adapter(cfg).to(clip_model.dtype)

        # Store features for post-process view-weight search
        self.store = False
        self.feat_store = []
        self.label_store = []

        self.num_prompts = cfg.TRAINER.PLOT.NUM_PROMPTS
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.in_features = cfg.MODEL.BACKBONE.CHANNEL

        self.have_logit_scale = cfg.TRAINER.logit_scale

    
    def forward(self, pc, label=None): 

        # Project to multi-view depth maps
        images = self.mv_proj(pc).type(self.dtype) # shape == torch.Size([BATCH_SIZE * NUM_VIEWS, 3, 224, 224])
        # Image features
        image_feat = self.visual_encoder(images) # shape == torch.Size([BATCH_SIZE * NUM_VIEWS, 512])
        image_feat = self.adapter(image_feat) # shape == torch.Size([BATCH_SIZE, 512 * NUM_VIEWS])
        image_feat = image_feat.reshape(-1, self.num_views, self.in_features)  # shape == torch.Size([BATCH_SIZE, NUM_VIEWS, 512])
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True) 

        # Store for the best ckpt
        if self.store:
            self.feat_store.append(image_feat)
            self.label_store.append(label)

        # Text features
        prompts = self.prompt_learner() # torch.Size([160, 77, 512])
        tokenized_prompts = self.tokenized_prompts # torch.Size([160, 77])
        text_feat = self.text_encoder(prompts, tokenized_prompts) # torch.Size([160, 512])
        text_feat = text_feat.contiguous().view(self.num_prompts, self.num_classes, self.in_features)
        text_feat = text_feat.permute(1, 0, 2) # shape == (num_classes, num_prompts, 512)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # print(image_feat.shape, text_feat.shape) # torch.Size([32, 10, 512]) torch.Size([40, 4, 512])

        if self.have_logit_scale is True:
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = 1
        logits = logit_scale * compute_logits(image_feat=image_feat, text_feat=text_feat, eps=0.01, max_iter=1000)
        return logits

    def mv_proj(self, pc):
        img = self.get_img(pc).cuda()
        img = img.unsqueeze(1).repeat(1, 3, 1, 1)
        return img


def compute_logits(image_feat, text_feat, eps=0.01, max_iter=1000):
    """
        image_feat.shape == (batch_size, num_views, dims)
        text_feat.shape == (num_classes, num_prompts, dims)
    """
    batch_size = image_feat.shape[0]
    num_views = image_feat.shape[1]
    in_features = image_feat.shape[2]
    num_classes = text_feat.shape[0]
    num_prompts = text_feat.shape[1]

    sim = torch.matmul(image_feat.reshape(-1, in_features), text_feat.reshape(-1, in_features).permute(1, 0))
    sim = sim.reshape(image_feat.shape[0], num_classes, num_views, num_prompts)

    sim = sim.view(batch_size * num_classes, num_views, num_prompts)
    wdist = 1.0 - sim

    p = torch.zeros(batch_size * num_classes, num_views, dtype=wdist.dtype, device=wdist.device).fill_(1. / num_views)
    q = torch.zeros(batch_size * num_classes, num_prompts, dtype=wdist.dtype, device=wdist.device).fill_(1. / num_prompts)
    sinkhorn_solver = SinkhornAlgorithm(epsilon=eps, iterations=max_iter)
    with torch.no_grad():
        wdist_exp = torch.exp(-wdist / eps)
        T = sinkhorn_solver(p, q, wdist_exp) # shape == (batch_size * num_classes, num_views, num_prompts)
        print(torch.sum(T))
        # assert torch.sum(T) == batch_size * num_classes

    d_OT = torch.sum(T * wdist, dim=(1, 2))
    d_OT = d_OT.contiguous().view(batch_size, num_classes)

    return d_OT


class Adapter(nn.Module):
    """
    Inter-view Adapter
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_views = cfg.MODEL.PROJECT.NUM_VIEWS
        self.in_features = cfg.MODEL.BACKBONE.CHANNEL
        self.adapter_ratio = cfg.MODEL.ADAPTER.RATIO
        self.fusion_init = cfg.MODEL.ADAPTER.INIT
        self.dropout = cfg.MODEL.ADAPTER.DROPOUT

        
        self.fusion_ratio = nn.Parameter(torch.tensor([self.fusion_init] * self.num_views), requires_grad=True)
        
        self.global_f = nn.Sequential(
                BatchNormPoint(self.in_features),
                nn.Dropout(self.dropout),
                nn.Flatten(),
                nn.Linear(in_features=self.in_features * self.num_views,
                          out_features=self.in_features),
                nn.BatchNorm1d(self.in_features),
                nn.ReLU(),
                nn.Dropout(self.dropout))

        self.view_f = nn.Sequential(
                nn.Linear(in_features=self.in_features,
                          out_features=self.in_features),
                nn.ReLU(),
                nn.Linear(in_features=self.in_features,
                          out_features=self.in_features * self.num_views),
                nn.ReLU())


    def forward(self, feat):

        img_feat = feat.reshape(-1, self.num_views, self.in_features)
        res_feat = feat.reshape(-1, self.num_views * self.in_features)
        
        # Global feature
        global_feat = self.global_f(img_feat * self.fusion_ratio.reshape(1, -1, 1)) # shape == (BATCH_SIZE, 512)
        # View-wise adapted features
        view_feat = self.view_f(global_feat) # shape == (BATCH_SIZE, 5120)
        
        img_feat = view_feat * self.adapter_ratio + res_feat * (1 - self.adapter_ratio) # shape == (BATCH_SIZE, 5120)

        # print(global_feat.shape, view_feat.shape, img_feat.shape)
        return img_feat



@TRAINER_REGISTRY.register()
class PointCLIP_FS(TrainerX):
    """
        PointCLIP: Point Cloud Understanding by CLIP
        https://arxiv.org/pdf/2112.02413.pdf
    """ 

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)

        print('Building PointCLIP')
        self.model = PointCLIP_Model(cfg, classnames, clip_model)

        print('Turning off gradients in both visual and textual encoders')
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name:
                continue
            elif "adapter" in name:
                continue
            else:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # give adapter to the optimizer
        self.adapter_optim = build_optimizer(self.model.adapter, cfg.OPTIM.ADAPTER)
        self.adapter_sched = build_lr_scheduler(self.adapter_optim, cfg.OPTIM.ADAPTER)
        self.register_model("adapter", self.model.adapter, self.adapter_optim, self.adapter_sched)

        # give prompt_learner to the optimizer
        self.prompt_optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM.PROMPT)
        self.prompt_sched = build_lr_scheduler(self.prompt_optim, cfg.OPTIM.PROMPT)
        self.register_model("prompt_learner", self.model.prompt_learner, self.prompt_optim, self.prompt_sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)
        

        self.sinkhorn_solver = SinkhornAlgorithm(epsilon=0.01, iterations=1000)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        d_OT = self.model(image) # shape == (batch_size, num_classes) # half

        batch_size = d_OT.shape[0]
        num_classes = d_OT.shape[1]

        T_empirical = torch.zeros(batch_size, num_classes).to(self.device).scatter(1, label.view(-1, 1), 1) # float
        T_empirical = T_empirical / torch.sum(T_empirical)

        p = torch.zeros(batch_size, dtype=d_OT.dtype, device=d_OT.device).fill_(1. / batch_size)
        q = torch.zeros(num_classes, dtype=d_OT.dtype, device=d_OT.device).fill_(1. / num_classes)
        # T_opt = self.sinkhorn_solver(p, q, d_OT)[0] # half

        reg_kl = (float("inf"), 0.001)
        reg = 0.01
        d_OT = d_OT.float() / d_OT.max()
        T_opt = ot.unbalanced.sinkhorn_unbalanced(a=p.float(), b=q.float(), reg=reg, reg_m=reg_kl, M=d_OT.float(), numItermax=1000, method="sinkhorn_stabilized")
        print(torch.sum(T_opt))

        # print(torch.sum(T_opt))
        loss = torch.sum(-(T_empirical) * torch.log(T_opt + 1e-4))
        # print(loss)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(-d_OT, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, image, label):
        d_OT = self.model(image)
        return -d_OT

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )

            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)