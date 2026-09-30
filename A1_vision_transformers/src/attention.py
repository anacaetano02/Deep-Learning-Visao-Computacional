"""Attention maps dos dois ViTs (R24–R26, Fase 6).

Fluxo sugerido no notebook:
    idx = escolher_exemplos(y_true, y_pred_pre, ...)              # as MESMAS imagens nos dois modelos
    at_zero = atencoes_de_indices(vit_zero, ds_zero_test, idx, DEVICE)
    at_pre  = atencoes_de_indices(vit_pre_eager, ds_pre_test, idx, DEVICE)
    mapas = {"ViT do zero": attention_rollout(at_zero), "ViT pré-treinado": attention_rollout(at_pre)}
    plotar_comparacao(imagens, mapas, titulos)

Como os datasets de teste dos dois modelos seguem a mesma ordem (SPLIT_OFICIAL, shuffle=False),
o mesmo índice aponta para a mesma imagem em 128 px (zero) e em 224 px (pré-treinado).

Só a extração usa PyTorch. Depois dela, tudo é numpy com forma (L, B, H, T, T):
L camadas, B imagens, H heads, T tokens (CLS na posição 0 + g×g patches).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CLASSES
from .evaluation import sigla


# ---------------------------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------------------------

def _eh_modelo_hf(modelo) -> bool:
    return hasattr(modelo, "config") and hasattr(modelo.config, "num_hidden_layers")


def _atencoes_da_saida(saida):
    """Lista de tensores (B, H, T, T), um por camada, a partir da saída do modelo."""
    if hasattr(saida, "attentions"):  # Hugging Face
        atencoes = saida.attentions
        if atencoes is None or any(a is None for a in atencoes):
            raise RuntimeError(
                "O ViT do HF não devolveu as atenções. Carregue-o com attn_implementation='eager' "
                "(a SDPA não calcula a matriz de pesos)."
            )
        return list(atencoes)
    if isinstance(saida, (tuple, list)) and len(saida) > 1:  # ViT do zero: (logits, pesos)
        return list(saida[1])
    raise TypeError(f"Não sei extrair as atenções de uma saída do tipo {type(saida).__name__}.")


@torch.no_grad()
def extrair_atencoes(modelo, x, device):
    """Roda um batch e devolve (probs, atencoes).

    probs: (B, n_classes) em numpy; atencoes: (L, B, H, T, T) em numpy float32.
    Sem AMP: os pesos ficam em fp32 e o batch é pequeno. O modo do modelo é restaurado no fim.
    """
    estava_treinando = modelo.training
    modelo.eval()
    try:
        x = x.to(device)
        saida = modelo(x, output_attentions=True) if _eh_modelo_hf(modelo) else modelo(x)
        logits = saida.logits if hasattr(saida, "logits") else saida[0]
        atencoes = _atencoes_da_saida(saida)
    finally:
        modelo.train(estava_treinando)

    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    atencoes = torch.stack([a.float().cpu() for a in atencoes]).numpy()
    if atencoes.ndim != 5 or atencoes.shape[-1] != atencoes.shape[-2]:
        raise ValueError(f"Forma inesperada das atenções: {atencoes.shape} (esperado L,B,H,T,T).")
    return probs, atencoes


def atencoes_de_indices(modelo, dataset, indices: Sequence[int], device):
    """Monta um batch com `dataset[i]` para cada índice e extrai as atenções.

    Devolve (probs, atencoes, imagens_normalizadas). Use poucos índices (≤ 16): no ViT-B as
    atenções de 12 camadas × 12 heads × 197² ocupam ~22 MB por imagem.
    """
    itens = [dataset[int(i)] for i in indices]
    x = torch.stack([it["pixel_values"] if isinstance(it, Mapping) else it[0] for it in itens])
    probs, atencoes = extrair_atencoes(modelo, x, device)
    return probs, atencoes, x.cpu()


def desnormalizar(x, media=(0.5, 0.5, 0.5), desvio=(0.5, 0.5, 0.5)) -> np.ndarray:
    """Tensor (B, 3, H, W) ou (3, H, W) normalizado → numpy (…, H, W, 3) em [0, 1] para exibir."""
    x = torch.as_tensor(x).float().cpu()
    m = torch.tensor(media).view(3, 1, 1)
    s = torch.tensor(desvio).view(3, 1, 1)
    img = (x * s + m).clamp(0, 1)
    return img.movedim(-3, -1).numpy()


# ---------------------------------------------------------------------------------------------
# Mapas (numpy)
# ---------------------------------------------------------------------------------------------

def _lado_grade(n_tokens: int) -> int:
    g = int(round(np.sqrt(n_tokens - 1)))
    if g * g != n_tokens - 1:
        raise ValueError(f"{n_tokens} tokens não formam CLS + grade quadrada.")
    return g


def mapa_cls(atencoes: np.ndarray, camada: int = -1, head: int | str = "media") -> np.ndarray:
    """Linha do CLS sobre os patches numa camada: (B, g, g).

    head="media" tira a média das heads; um inteiro escolhe uma head.
    Cada mapa é renormalizado para somar 1 nos patches (a atenção do CLS nele mesmo sai).
    """
    a = atencoes[camada]  # (B, H, T, T)
    a = a.mean(axis=1) if head == "media" else a[:, head]
    linha = a[:, 0, 1:]  # CLS olhando para os patches
    linha = linha / linha.sum(axis=1, keepdims=True)
    g = _lado_grade(a.shape[-1])
    return linha.reshape(-1, g, g)


def attention_rollout(atencoes: np.ndarray, peso_residual: float = 0.5,
                      ate_camada: int | None = None) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema, 2020): (B, g, g).

    Em cada camada: média das heads, soma da identidade para representar a conexão residual
    (A' = (1 − p)·A + p·I), renormalização das linhas; depois multiplica as camadas em ordem.
    A linha do CLS do produto mostra quanto cada patch de entrada chega ao CLS final, somando
    todos os caminhos — mais estável que olhar uma camada só.
    """
    a = atencoes if ate_camada is None else atencoes[: ate_camada + 1]
    L, B, H, T, _ = a.shape
    identidade = np.eye(T, dtype=np.float64)
    rollout = np.broadcast_to(identidade, (B, T, T)).copy()
    for camada in range(L):
        m = a[camada].mean(axis=1).astype(np.float64)  # (B, T, T)
        m = (1 - peso_residual) * m + peso_residual * identidade
        m = m / m.sum(axis=-1, keepdims=True)
        rollout = m @ rollout
    linha = rollout[:, 0, 1:]
    linha = linha / linha.sum(axis=1, keepdims=True)
    g = _lado_grade(T)
    return linha.reshape(B, g, g)


def fracao_na_borda(mapas: np.ndarray, largura: int = 1) -> np.ndarray:
    """Fração da atenção no anel externo da grade, por imagem: (B,).

    Serve para checar atalhos (vinheta do dermatoscópio, cantos, régua): compare com a fração
    de patches que o anel ocupa (`fracao_uniforme_borda`), que é o valor de uma atenção uniforme.
    """
    g = mapas.shape[-1]
    borda = np.ones((g, g), dtype=bool)
    borda[largura:g - largura, largura:g - largura] = False
    return (mapas * borda).sum(axis=(1, 2)) / mapas.sum(axis=(1, 2))


def fracao_uniforme_borda(g: int, largura: int = 1) -> float:
    """Fração dos patches no anel externo (referência para `fracao_na_borda`)."""
    return 1 - (g - 2 * largura) ** 2 / g**2


def distancia_media_atencao(atencoes: np.ndarray, tamanho_patch: int = 16,
                            tamanho_img: int | None = None) -> np.ndarray:
    """Distância média de atenção por camada e head: (L, H), em pixels da entrada.

    Para cada patch de consulta, média das distâncias até os patches de chave, ponderada pelos
    pesos (CLS excluído). É a análise da Fig. 7 do artigo do ViT: heads com distância pequena são
    locais (parecidas com convolução); com distância grande, globais. Em pixels para comparar a
    grade 8×8 (128 px) com a 14×14 (224 px); com `tamanho_img`, devolve a fração do lado da imagem.
    """
    L, B, H, T, _ = atencoes.shape
    g = _lado_grade(T)
    lin, col = np.divmod(np.arange(g * g), g)
    pos = np.stack([lin, col], axis=1) * tamanho_patch
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)  # (P, P)

    a = atencoes[..., 1:, 1:]
    a = a / a.sum(axis=-1, keepdims=True)  # renormaliza sem o CLS
    d = (a * dist).sum(axis=-1).mean(axis=(1, 3))  # média no batch e nas consultas → (L, H)
    return d / tamanho_img if tamanho_img else d


# ---------------------------------------------------------------------------------------------
# Escolha das imagens
# ---------------------------------------------------------------------------------------------

def escolher_exemplos(y_true, y_pred, classes_alvo: Sequence[int], n_acertos: int = 1,
                      n_erros: int = 1, seed: int = 42) -> list[int]:
    """Índices do teste com acertos e erros de cada classe-alvo, sorteados com seed fixa.

    Use o y_pred de um modelo de referência (ex.: o pré-treinado) e passe os MESMOS índices aos
    dois modelos. Classes sem erros (ou sem acertos) simplesmente contribuem com menos imagens.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    escolhidos = []
    for c in classes_alvo:
        da_classe = y_true == c
        for mascara, n in ((da_classe & (y_pred == c), n_acertos), (da_classe & (y_pred != c), n_erros)):
            candidatos = np.flatnonzero(mascara)
            if len(candidatos):
                escolhidos += rng.choice(candidatos, size=min(n, len(candidatos)), replace=False).tolist()
    return escolhidos


# ---------------------------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------------------------

def _sobrepor(ax, imagem: np.ndarray, mapa: np.ndarray, alpha: float = 0.5):
    """Mostra o mapa g×g esticado sobre a imagem (interpolação bilinear só na exibição)."""
    h, w = imagem.shape[:2]
    ax.imshow(imagem, extent=(0, w, h, 0))
    ax.imshow(mapa / mapa.max(), cmap="jet", alpha=alpha, extent=(0, w, h, 0),
              interpolation="bilinear", vmin=0, vmax=1)
    ax.axis("off")


def plotar_comparacao(imagens: np.ndarray, mapas: Mapping[str, np.ndarray],
                      titulos_linhas: Sequence[str] | None = None, alpha: float = 0.5,
                      titulo: str = "Attention rollout"):
    """Uma linha por imagem: original + um mapa por modelo.

    `imagens` (B, H, W, 3) em [0, 1] (ex.: `desnormalizar` da entrada de 224 px); cada mapa em
    `mapas` tem forma (B, g, g) e é esticado ao tamanho da imagem, qualquer que seja o g.
    """
    B = len(imagens)
    nomes = list(mapas)
    fig, eixos = plt.subplots(B, 1 + len(nomes), figsize=(3.2 * (1 + len(nomes)), 3.2 * B),
                              squeeze=False)
    for i in range(B):
        eixos[i, 0].imshow(imagens[i])
        eixos[i, 0].axis("off")
        if titulos_linhas is not None:
            eixos[i, 0].set_title(titulos_linhas[i], fontsize=9)
        for j, nome in enumerate(nomes, start=1):
            _sobrepor(eixos[i, j], imagens[i], mapas[nome][i], alpha)
            if i == 0:
                eixos[i, j].set_title(f"{nome}\n(grade {mapas[nome].shape[-1]}×{mapas[nome].shape[-1]})",
                                      fontsize=10)
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


def plotar_heads(atencoes: np.ndarray, imagem: np.ndarray, indice: int = 0, camada: int = -1,
                 alpha: float = 0.5, titulo: str | None = None):
    """Mapa do CLS de cada head de uma camada, para uma imagem — ajuda a escolher a head."""
    H = atencoes.shape[2]
    n_col = min(H, 6)
    n_lin = int(np.ceil(H / n_col))
    fig, eixos = plt.subplots(n_lin, n_col, figsize=(2.6 * n_col, 2.6 * n_lin), squeeze=False)
    for h, ax in enumerate(eixos.flat):
        if h < H:
            _sobrepor(ax, imagem, mapa_cls(atencoes, camada, h)[indice], alpha)
            ax.set_title(f"head {h}", fontsize=9)
        else:
            ax.axis("off")
    n_camada = camada if camada >= 0 else atencoes.shape[0] + camada
    fig.suptitle(titulo or f"CLS → patches, camada {n_camada}")
    fig.tight_layout()
    return fig


def plotar_distancia(distancias: Mapping[str, np.ndarray], titulo: str = "Distância média de atenção"):
    """Um ponto por head, por camada, para cada modelo (entrada: saída de distancia_media_atencao)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (nome, d), marcador in zip(distancias.items(), "os^D"):
        L, H = d.shape
        camadas = np.repeat(np.arange(L), H) + 1
        ax.scatter(camadas, d.ravel(), label=nome, marker=marcador, alpha=0.7)
    ax.set_xlabel("Camada")
    ax.set_ylabel("Distância média (fração do lado)" if max(d.max() for d in distancias.values()) <= 1.5
                  else "Distância média (px)")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def titulos_exemplos(indices, y_true, probs_por_modelo: Mapping[str, np.ndarray],
                     classes: Sequence[str] = CLASSES):
    """Títulos 'real: mel | zero: nv (0,81) | pré: mel (0,93)' para as linhas da comparação.

    `probs_por_modelo[nome]` tem forma (len(indices), n_classes) — a saída de atencoes_de_indices.
    """
    abreviar = lambda c: sigla(classes[c])
    titulos = []
    for k, i in enumerate(indices):
        partes = [f"real: {abreviar(int(np.asarray(y_true)[i]))}"]
        for nome, p in probs_por_modelo.items():
            c = int(p[k].argmax())
            partes.append(f"{nome}: {abreviar(c)} ({p[k, c]:.2f})")
        titulos.append("\n".join(partes))
    return titulos