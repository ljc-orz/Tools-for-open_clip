from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QKVSeparatedMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.bias = bias
        self.add_zero_attn = add_zero_attn
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim, **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim, **factory_kwargs))
            self._reset_bias_kv()
        else:
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

    def _reset_bias_kv(self) -> None:
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    @classmethod
    def from_multihead_attention(cls, module: nn.MultiheadAttention) -> "QKVSeparatedMultiheadAttention":
        new_module = cls(
            embed_dim=module.embed_dim,
            num_heads=module.num_heads,
            dropout=module.dropout,
            bias=module.in_proj_bias is not None,
            add_bias_kv=module.bias_k is not None,
            add_zero_attn=module.add_zero_attn,
            kdim=module.kdim,
            vdim=module.vdim,
            batch_first=module.batch_first,
            device=module.out_proj.weight.device,
            dtype=module.out_proj.weight.dtype,
        )
        new_module.load_from_multihead_attention(module)
        return new_module

    def load_from_multihead_attention(self, module: nn.MultiheadAttention) -> "QKVSeparatedMultiheadAttention":
        if module.embed_dim != self.embed_dim or module.num_heads != self.num_heads:
            raise ValueError("Shape mismatch between source MultiheadAttention and target module")
        if module.add_zero_attn != self.add_zero_attn:
            raise ValueError("add_zero_attn mismatch between source MultiheadAttention and target module")
        if bool(module.bias_k is not None) != bool(self.bias_k is not None):
            raise ValueError("add_bias_kv mismatch between source MultiheadAttention and target module")

        with torch.no_grad():
            if module._qkv_same_embed_dim:
                in_proj_weight = module.in_proj_weight
                assert in_proj_weight is not None
                self.q_proj.weight.copy_(in_proj_weight[: self.embed_dim])
                self.k_proj.weight.copy_(in_proj_weight[self.embed_dim : 2 * self.embed_dim])
                self.v_proj.weight.copy_(in_proj_weight[2 * self.embed_dim :])
            else:
                assert module.q_proj_weight is not None
                assert module.k_proj_weight is not None
                assert module.v_proj_weight is not None
                self.q_proj.weight.copy_(module.q_proj_weight)
                self.k_proj.weight.copy_(module.k_proj_weight)
                self.v_proj.weight.copy_(module.v_proj_weight)

            if module.in_proj_bias is not None:
                self.q_proj.bias.copy_(module.in_proj_bias[: self.embed_dim])
                self.k_proj.bias.copy_(module.in_proj_bias[self.embed_dim : 2 * self.embed_dim])
                self.v_proj.bias.copy_(module.in_proj_bias[2 * self.embed_dim :])
            elif self.q_proj.bias is not None:
                self.q_proj.bias.zero_()
                self.k_proj.bias.zero_()
                self.v_proj.bias.zero_()

            self.out_proj.weight.copy_(module.out_proj.weight)
            if module.out_proj.bias is not None and self.out_proj.bias is not None:
                self.out_proj.bias.copy_(module.out_proj.bias)

            if self.bias_k is not None and module.bias_k is not None:
                self.bias_k.copy_(module.bias_k)
            if self.bias_v is not None and module.bias_v is not None:
                self.bias_v.copy_(module.bias_v)

        self.train(module.training)
        if module._qkv_same_embed_dim:
            source_weights = [module.in_proj_weight, module.in_proj_weight, module.in_proj_weight]
        else:
            source_weights = [module.q_proj_weight, module.k_proj_weight, module.v_proj_weight]
        for target_param, source_param in zip(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], source_weights
        ):
            if source_param is not None:
                target_param.requires_grad = source_param.requires_grad
        if self.q_proj.bias is not None and module.in_proj_bias is not None:
            self.q_proj.bias.requires_grad = module.in_proj_bias.requires_grad
            self.k_proj.bias.requires_grad = module.in_proj_bias.requires_grad
            self.v_proj.bias.requires_grad = module.in_proj_bias.requires_grad
        if self.out_proj.bias is not None and module.out_proj.bias is not None:
            self.out_proj.bias.requires_grad = module.out_proj.bias.requires_grad
        if self.bias_k is not None and module.bias_k is not None:
            self.bias_k.requires_grad = module.bias_k.requires_grad
        if self.bias_v is not None and module.bias_v is not None:
            self.bias_v.requires_grad = module.bias_v.requires_grad
        return self

    def _reshape_to_heads(self, tensor: Tensor, batch_size: int, sequence_length: int) -> Tensor:
        tensor = tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: Tensor, batch_size: int, sequence_length: int) -> Tensor:
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, sequence_length, self.embed_dim)

    def _broadcast_attn_mask(self, attn_mask: Tensor, batch_size: int, tgt_len: int, src_len: int) -> Tensor:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            if attn_mask.size(0) == batch_size * self.num_heads:
                attn_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len)
            elif attn_mask.size(0) == batch_size:
                attn_mask = attn_mask.unsqueeze(1)
            else:
                raise ValueError("attn_mask has incompatible shape")
        else:
            raise ValueError("attn_mask must have shape [L, S] or [N * num_heads, L, S]")
        return attn_mask

    def _broadcast_key_padding_mask(self, key_padding_mask: Tensor, batch_size: int, src_len: int) -> Tensor:
        if key_padding_mask.dim() == 1:
            key_padding_mask = key_padding_mask.unsqueeze(0)
        if key_padding_mask.size(0) != batch_size or key_padding_mask.size(1) != src_len:
            raise ValueError("key_padding_mask has incompatible shape")
        return key_padding_mask.view(batch_size, 1, 1, src_len)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3
        if not is_batched:
            if self.batch_first:
                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)
                if key_padding_mask is not None and key_padding_mask.dim() == 1:
                    key_padding_mask = key_padding_mask.unsqueeze(0)
            else:
                query = query.unsqueeze(1)
                key = key.unsqueeze(1)
                value = value.unsqueeze(1)
                if key_padding_mask is not None and key_padding_mask.dim() == 1:
                    key_padding_mask = key_padding_mask.unsqueeze(0)

        if self.batch_first:
            batch_size, tgt_len, _ = query.shape
            src_len = key.shape[1]
        else:
            tgt_len, batch_size, _ = query.shape
            src_len = key.shape[0]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.bias_k is not None and self.bias_v is not None:
            bias_k = self.bias_k.expand(batch_size, -1, -1)
            bias_v = self.bias_v.expand(batch_size, -1, -1)
            k = torch.cat([k, bias_k], dim=1)
            v = torch.cat([v, bias_v], dim=1)
            src_len = src_len + 1
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1), value=False)

        if self.add_zero_attn:
            zero_k = torch.zeros(batch_size, 1, self.embed_dim, device=k.device, dtype=k.dtype)
            zero_v = torch.zeros(batch_size, 1, self.embed_dim, device=v.device, dtype=v.dtype)
            k = torch.cat([k, zero_k], dim=1)
            v = torch.cat([v, zero_v], dim=1)
            src_len = src_len + 1
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1), value=False)

        q = self._reshape_to_heads(q, batch_size, tgt_len)
        k = self._reshape_to_heads(k, batch_size, src_len)
        v = self._reshape_to_heads(v, batch_size, src_len)

        attn_logits = torch.matmul(q, k.transpose(-2, -1))

        if is_causal:
            causal_mask = torch.ones(tgt_len, src_len, device=attn_logits.device, dtype=torch.bool).triu(diagonal=1)
            attn_logits = attn_logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn_logits.dtype).min)

        if attn_mask is not None:
            attn_mask = self._broadcast_attn_mask(attn_mask.to(device=attn_logits.device), batch_size, tgt_len, src_len)
            if attn_mask.dtype == torch.bool:
                attn_logits = attn_logits.masked_fill(attn_mask, torch.finfo(attn_logits.dtype).min)
            else:
                attn_logits = attn_logits + attn_mask.to(dtype=attn_logits.dtype)

        if key_padding_mask is not None:
            key_padding_mask = self._broadcast_key_padding_mask(
                key_padding_mask.to(device=attn_logits.device), batch_size, src_len
            )
            if key_padding_mask.dtype == torch.bool:
                attn_logits = attn_logits.masked_fill(key_padding_mask, torch.finfo(attn_logits.dtype).min)
            else:
                attn_logits = attn_logits + key_padding_mask.to(dtype=attn_logits.dtype)

        attn_weights = torch.softmax(attn_logits, dim=-1)
        if self.training and self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = self._merge_heads(attn_output, batch_size, tgt_len)
        attn_output = self.out_proj(attn_output)

        if not self.batch_first and is_batched:
            attn_output = attn_output.transpose(0, 1)

        if not is_batched:
            attn_output = attn_output.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        if not need_weights:
            return attn_output, None

        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1 if is_batched else 0)
        return attn_output, attn_weights


def replace_multihead_attention(module: nn.Module) -> tuple[nn.Module, int]:
    replaced_count = 0

    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, nn.MultiheadAttention):
            replacement = QKVSeparatedMultiheadAttention.from_multihead_attention(child_module)
            setattr(module, child_name, replacement)
            replaced_count += 1
        else:
            _, child_count = replace_multihead_attention(child_module)
            replaced_count += child_count

    return module, replaced_count


def convert_multihead_attention(module: nn.Module) -> tuple[nn.Module, int]:
    return replace_multihead_attention(module)


if __name__ == "__main__":
    import open_clip
    from rich import print
    x = torch.randn(1, 3, 224, 224)
    model = open_clip.create_model("ViT-B-16-quickgelu", pretrained="openai")
    
    y_before = model.encode_image(x).clone() # type: ignore[union-attr]
    
    model.visual, replaced_count = convert_multihead_attention(model.visual)  # type: ignore[arg-type]
    print(f"Replaced MultiheadAttention layers: {replaced_count}")
    print(model.visual)
    
    y_after = model.encode_image(x).clone() # type: ignore[union-attr]
    
    print("Output difference after replacement:", torch.norm(y_before - y_after).item())
    print("Outputs are close:", torch.allclose(y_before, y_after, atol=1e-6))
