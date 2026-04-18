from peft import LoraConfig, get_peft_model, PeftModel


def get_naive_mha_qkvo_lora(module, r, alpha, dropout, target_qkvo: list[str], target_layer: list[int]|None = None):
    assert len(target_qkvo) > 0 and set(target_qkvo).issubset({"q", "k", "v", "out"}), "target_qkvo must be a subset of {'q', 'k', 'v', 'out'}"
    target_qkvo = sorted(target_qkvo, key=lambda x: {"q": 0, "k": 1, "v": 2, "out": 3}[x])
    if target_layer is None or len(target_layer) == 0:
        target_modules = r"transformer\.resblocks\.\d+\.attn\.(%s)_proj" % "|".join(target_qkvo)
    else:
        target_modules = r"transformer\.resblocks\.(%s)\.attn\.(%s)_proj" % ("|".join(map(str, target_layer)), "|".join(target_qkvo))
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )
    return get_peft_model(module, config)

def save_naive_mha_qkvo_lora(model, path):
    model.save_pretrained(path)

def load_naive_mha_qkvo_lora(model, path):
    return PeftModel.from_pretrained(model, path)


if __name__ == "__main__":
    from rich import print
    import open_clip
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../torch_mha_2_naive_mha')))
    import naive_mha # type: ignore[import]
    
    model = open_clip.create_model("ViT-B-16-quickgelu", pretrained="openai")
    model.visual, _ = naive_mha.convert_multihead_attention(model.visual)  # type: ignore[arg-type]
    model.visual = get_naive_mha_qkvo_lora(model.visual, 
                                           r=4, alpha=16, dropout=0.1, 
                                           target_qkvo=["q", "v", "out"], 
                                           target_layer=[8, 9, 10, 11])
    print(model.visual)
    model.visual.print_trainable_parameters()