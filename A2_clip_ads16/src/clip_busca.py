"""CLIP: carregamento com revisão fixada e embeddings normalizados de imagens e textos (só inferência)."""
import os

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from src.dados import carregar_imagem


def carregar_clip(checkpoint: str, revisao: str, device: str):
    """Modelo (fp32, eval, no device) e processor do MESMO checkpoint e revisão.

    * processor: resize, crop, normalização com a média/desvio do pré-treino e tokenização, ou seja,
      exatamente a entrada que o modelo espera;
    * eval(): desliga o dropout, então a mesma imagem gera sempre o mesmo embedding;
    * fp32: o modelo cabe com folga na T4; fp16 pode mudar o cosseno na 3ª casa, perto do threshold;
    * pesos: o commit fixado do openai/clip-vit-large-patch14-336 só tem pytorch_model.bin (o
      model.safetensors existe apenas em PRs do robô SFconvertbot). use_safetensors=True dá erro com a
      revisão fixada, e o transformers, ao carregar o .bin, baixaria em segundo plano o .safetensors de
      um PR (+1,7 GB, sem uso). DISABLE_SAFETENSORS_CONVERSION desliga isso: os pesos são os do SHA.
    """
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    modelo = CLIPModel.from_pretrained(checkpoint, revision=revisao, dtype=torch.float32).to(device).eval()
    processor = CLIPProcessor.from_pretrained(checkpoint, revision=revisao)
    return modelo, processor


def _projecao(saida, dim: int) -> torch.Tensor:
    """Embedding no espaço compartilhado imagem-texto, a partir da saída de get_*_features.

    No transformers 5, get_image_features/get_text_features devolvem um BaseModelOutputWithPooling:
    .pooler_output já passou pela projeção (visual_projection/text_projection) e tem `dim` dimensões.
    O primeiro campo ([0]) é o last_hidden_state, sem projeção (na imagem, (lote, 577 tokens, 1024)):
    usá-lo daria um embedding errado.
    """
    emb = saida if isinstance(saida, torch.Tensor) else saida.pooler_output
    assert emb.ndim == 2 and emb.shape[-1] == dim, tuple(emb.shape)
    return emb


def embeddings_imagens(modelo, processor, caminhos, tamanho_lote: int = 32) -> torch.Tensor:
    """Embeddings L2-normalizados (N, projection_dim) na ordem de `caminhos`, em lotes (CPU, fp32).

    Em lotes porque o ViT-L/14 a 336 px tem 577 tokens por imagem: as 301 de uma vez pesariam na VRAM.
    As imagens passam por carregar_imagem (RGB, alfa composto sobre fundo branco).
    """
    device = next(modelo.parameters()).device
    dim = modelo.config.projection_dim
    partes = []
    for i in range(0, len(caminhos), tamanho_lote):
        imagens = [carregar_imagem(c) for c in caminhos[i:i + tamanho_lote]]
        pixels = processor(images=imagens, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            emb = _projecao(modelo.get_image_features(pixel_values=pixels), dim)
        partes.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(partes)


def embeddings_textos(modelo, processor, textos) -> torch.Tensor:
    """Embeddings L2-normalizados (N, projection_dim) de textos (CPU, fp32).

    padding=True completa as frases até a mais longa do lote; a attention_mask devolvida pelo
    processor faz o modelo ignorar esses tokens de padding (ver R14).
    """
    device = next(modelo.parameters()).device
    entrada = processor(text=list(textos), return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        emb = _projecao(modelo.get_text_features(**entrada), modelo.config.projection_dim)
    return F.normalize(emb, dim=-1).cpu()
