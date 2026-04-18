import open_clip
import torch
from visual_utils import inject_output_image_tokens_method, clip_encode, read_image, plot_cosine_similarity_map, de_normalize_image

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../torch_mha_2_naive_mha')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../peft_lora')))
import naive_mha # type: ignore[import]
import naive_mha_lora # type: ignore[import]

@torch.inference_mode()
def test_official_open_clip(img_path, caption, model_name, pretrained):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = inject_output_image_tokens_method(model) # type: ignore[union-attr]
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    image = read_image(img_path, preprocess).unsqueeze(0) # type: ignore[union-attr]
    text = tokenizer(caption) # type: ignore[union-attr]

    _, _, text_feat, local_feats = clip_encode(model, image, text, True)
    assert local_feats is not None
    h = int(local_feats.shape[1]**0.5)
    local_feats = local_feats.reshape(local_feats.shape[0], h, h, local_feats.shape[-1]) # (B, H, W, D)

    sim = -1 * torch.einsum("bhwd,bd->bhw", local_feats, text_feat) # (B, H, W)
    image = de_normalize_image(image.squeeze(0), preprocess) # type: ignore[union-attr]
    plot_cosine_similarity_map(sim[0], title="official", save_path="official.png", image=image)
    return sim

@torch.inference_mode()
def test_naive_mha(img_path, caption, model_name, pretrained):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.visual, _ = naive_mha.convert_multihead_attention(model.visual)  # type: ignore[arg-type]
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    image = read_image(img_path, preprocess).unsqueeze(0) # type: ignore[union-attr]
    text = tokenizer(caption) # type: ignore[union-attr]

    model = inject_output_image_tokens_method(model) # type: ignore[union-attr]
    _, _, text_feat, local_feats = clip_encode(model, image, text, True)
    assert local_feats is not None
    h = int(local_feats.shape[1]**0.5)
    local_feats = local_feats.reshape(local_feats.shape[0], h, h, local_feats.shape[-1]) # (B, H, W, D)

    sim = -1 * torch.einsum("bhwd,bd->bhw", local_feats, text_feat) # (B, H, W)
    image = de_normalize_image(image.squeeze(0), preprocess) # type: ignore[union-attr]
    plot_cosine_similarity_map(sim[0], title="naive_mha", save_path="naive_mha.png", image=image)
    return sim

@torch.inference_mode()
def test_mha_lora(img_path, caption, model_name, pretrained):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.visual, _ = naive_mha.convert_multihead_attention(model.visual)  # type: ignore[arg-type]
    model.visual = naive_mha_lora.get_naive_mha_qkvo_lora(model.visual,  # type: ignore[call-arg]
                                                            r=4, alpha=16, dropout=0.1, 
                                                            target_qkvo=["q", "k", "v", "out"], 
                                                            target_layer=[8, 9, 10, 11])
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    image = read_image(img_path, preprocess).unsqueeze(0) # type: ignore[union-attr]
    text = tokenizer(caption) # type: ignore[union-attr]

    model = inject_output_image_tokens_method(model) # type: ignore[union-attr]
    _, _, text_feat, local_feats = clip_encode(model, image, text, True)
    assert local_feats is not None
    h = int(local_feats.shape[1]**0.5)
    local_feats = local_feats.reshape(local_feats.shape[0], h, h, local_feats.shape[-1]) # (B, H, W, D)

    sim = -1 * torch.einsum("bhwd,bd->bhw", local_feats, text_feat) # (B, H, W)
    image = de_normalize_image(image.squeeze(0), preprocess) # type: ignore[union-attr]
    plot_cosine_similarity_map(sim[0], title="mha_lora", save_path="mha_lora.png", image=image)
    return sim

if __name__ == "__main__":
    img_path = "./assets/catdog.jpg"
    caption = "a photo of a dog and a cat"
    model_name = "ViT-B-16-quickgelu"
    pretrained = "openai"
    sim_official = test_official_open_clip(img_path, caption, model_name, pretrained)
    sim_naive_mha = test_naive_mha(img_path, caption, model_name, pretrained)
    sim_mha_lora = test_mha_lora(img_path, caption, model_name, pretrained)
    print(torch.allclose(sim_official, sim_naive_mha), torch.allclose(sim_official, sim_mha_lora))
