import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle
from PIL import Image
import open_clip
from open_clip.model import CLIP
from open_clip.tokenizer import SimpleTokenizer
from open_clip.transformer import ResidualAttentionBlock
from torchvision.transforms import Compose, Normalize
import matplotlib.pyplot as plt


def _project_value(module: nn.Module, value: Tensor) -> Tensor:
    if hasattr(module, "v_proj"):
        return module.v_proj(value)  # type: ignore[attr-defined]

    in_proj_weight = getattr(module, "in_proj_weight", None)
    if in_proj_weight is not None:
        embed_dim = getattr(module, "embed_dim")
        v_proj_weight = in_proj_weight[2 * embed_dim:]
        in_proj_bias = getattr(module, "in_proj_bias", None)
        v_proj_bias = in_proj_bias[2 * embed_dim:] if in_proj_bias is not None else None
        return F.linear(value, v_proj_weight, v_proj_bias)

    v_proj_weight = getattr(module, "v_proj_weight", None)
    if v_proj_weight is not None:
        v_proj_bias = getattr(module, "v_proj_bias", None)
        return F.linear(value, v_proj_weight, v_proj_bias)

    raise TypeError(f"Unsupported attention module type: {type(module).__name__}")

class TransformerVHook:
    def __init__(self):
        self.output: Tensor = torch.tensor(0)
        self.handle: RemovableHandle = None # type: ignore

    def __call__(self, module: nn.Module,
                 input: tuple[Tensor, Tensor, Tensor],
                 output: tuple[Tensor, Tensor|None]):
        v = input[-1]
        v_proj = _project_value(module, v)

        self.output = v + v_proj # (B, L+1, D)

def encode_image(model: CLIP, image, normalize: bool = False) -> tuple[Tensor, Tensor|None]:
    global_z = model.encode_image(image)
    v_hook = getattr(model, "v_hook", None)
    if v_hook is None:
        if normalize:
            global_z = F.normalize(global_z, dim=-1)
        return global_z, None
    else:
        v = v_hook.output[:, 1:, :]  # (B, L, D)
        last_block: ResidualAttentionBlock = model.visual.transformer.resblocks[-1] # type: ignore
        local_z = model.visual.ln_post(v + last_block.mlp(last_block.ln_2(v))).contiguous() # type: ignore
        if model.visual.proj is not None:
            local_z = local_z @ model.visual.proj
        if normalize:
            global_z = F.normalize(global_z, dim=-1)
            local_z = F.normalize(local_z, dim=-1)
        return global_z, local_z

def inject_output_image_tokens_method(model: CLIP) -> CLIP:
    trans_blocks: nn.ModuleList = model.visual.transformer.resblocks # type: ignore
    last_block: ResidualAttentionBlock = trans_blocks[-1] # type: ignore

    v_hook = TransformerVHook()
    v_hook.handle = last_block.attn.register_forward_hook(v_hook)

    setattr(model, "v_hook", v_hook)

    return model

def get_clip(model_name: str="ViT-B-16", pretrained:str|None="openai", inject: bool = False) -> tuple[CLIP, Compose, Compose, SimpleTokenizer]:
    model, train_trans, eval_trans = open_clip.create_model_and_transforms(model_name, pretrained, jit=False)
    tokenizer = open_clip.get_tokenizer(model_name)
    if inject:
        model = inject_output_image_tokens_method(model) # type: ignore
    return model, train_trans, eval_trans, tokenizer # type: ignore

def clip_encode(model: CLIP, images: Tensor, texts: Tensor, normalize: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor|None]:
    global_z, local_z = encode_image(model, images, normalize)
    text_features = model.encode_text(texts, normalize)
    logit_scale = model.logit_scale.exp()
    return logit_scale, global_z, text_features, local_z

def read_image(image_path: str, transform: Compose) -> Tensor:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    return image_tensor # type: ignore

def de_normalize_image(images_tensor: Tensor, transform: Compose) -> Tensor:
    for t in transform.transforms:
        if isinstance(t, Normalize):
            mean = torch.tensor(t.mean).view(-1, 1, 1)
            std = torch.tensor(t.std).view(-1, 1, 1)
    return torch.clamp(images_tensor * std + mean, 0, 1)

def plot_cosine_similarity_map(sim_tensor: Tensor, title: str, save_path: str, image: Tensor|None=None,
                               alpha: float = 0.5, cmap: str = "jet", interpolation: str = "nearest"):
    plt.figure(figsize=(6, 6))
    if image is not None:
        C, H, W = image.shape
        HH, WW = sim_tensor.shape
        sim_tensor = sim_tensor.repeat_interleave(H//HH, dim=0).repeat_interleave(W//WW, dim=1)
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    sim_np = sim_tensor.detach().cpu().numpy()
    plt.imshow(
        sim_np,
        alpha=alpha,
        cmap=cmap,
        interpolation=interpolation
    )
    plt.title(title, fontsize=40, pad=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()