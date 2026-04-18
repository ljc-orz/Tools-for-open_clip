# Torch Multi-head attention to Naive MHA

```python
def convert_multihead_attention(module: nn.Module) -> tuple[nn.Module, int]:
    ...
```

将module中的`nn.MultiheadAttention`替换为简单的多头注意力，其中Q/K/V的投影层相互分离。返回替换后的module以及替换的MHA数目。

权重等效迁移：

```python
model = open_clip.create_model("ViT-B-16-quickgelu", pretrained="openai")

y_before = model.encode_image(x).clone() # type: ignore[union-attr]

model.visual, replaced_count = convert_multihead_attention(model.visual)  # type: ignore[arg-type]
y_after = model.encode_image(x).clone() # type: ignore[union-attr]

print("Output difference after replacement:", torch.norm(y_before - y_after).item())
print("Outputs are close:", torch.allclose(y_before, y_after, atol=1e-6))
```


输出：

```bash
Output difference after replacement: 8.951687050284818e-06
Outputs are close: True
```

---

以`ViT-B-16-quickgelu/openai`为例，转换前：

```bash
VisionTransformer(
  (conv1): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16), bias=False)
  (patch_dropout): Identity()
  (ln_pre): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (transformer): Transformer(
    (resblocks): ModuleList(
      (0-11): 12 x ResidualAttentionBlock(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (ls_1): Identity()
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ls_2): Identity()
      )
    )
  )
  (ln_post): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
```

转换后：

```bash
Replaced MultiheadAttention layers: 12
VisionTransformer(
  (conv1): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16), bias=False)
  (patch_dropout): Identity()
  (ln_pre): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (transformer): Transformer(
    (resblocks): ModuleList(
      (0-11): 12 x ResidualAttentionBlock(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): QKVSeparatedMultiheadAttention(
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (ls_1): Identity()
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (gelu): QuickGELU()
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (ls_2): Identity()
      )
    )
  )
  (ln_post): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
```