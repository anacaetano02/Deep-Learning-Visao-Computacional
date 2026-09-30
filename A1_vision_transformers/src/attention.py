"""Attention maps dos dois ViTs (R24–R26, Fase 6).

Fluxo sugerido no notebook (obrigatório: mapa de UMA head, R24/R26):
    idx = escolher_exemplos(y_true, y_pred_pre, ..., grupos=lesion_ids)   # MESMAS imagens nos dois
    probs_z, at_zero, x_z, _ = atencoes_de_indices(vit_zero, ds_zero_test, idx, DEVICE)
    probs_p, at_pre, x_p, _  = atencoes_de_indices(vit_pre_eager, ds_pre_test, idx, DEVICE)
    # Critério FIXO, decidido antes de olhar as imagens: última camada, TODAS as heads
    plotar_heads(at_zero, desnormalizar(x_p)[0], indice=0)                   # 3 heads (R26)
    plotar_heads(at_pre,  desnormalizar(x_p)[0], indice=0)                   # 12 heads (R24)
    mapas = {"zero (head 0)": mapa_cls(at_zero, -1, 0), "pré (head 0)": mapa_cls(at_pre, -1, 0)}
    plotar_comparacao(desnormalizar(x_p), mapas, titulos, probs_por_modelo={...})

`attention_rollout` (média das heads × camadas) é o bônus B3: não substitui o mapa de uma head.

Como os datasets de teste dos dois modelos seguem a mesma ordem (SPLIT_OFICIAL, shuffle=False),
o mesmo índice aponta para a mesma imagem em 128 px (zero) e em 224 px (pré-treinado); os dois
vêm do mesmo resize quadrado, então o mapa 8×8 pode ser exibido sobre a imagem de 224 px.

Cuidados de interpretação (R25):
* Atenção alta num patch não prova que ele decidiu a predição ("Attention is not Explanation",
  Jain & Wallace, 2019). ViTs grandes também concentram atenção em patches de fundo pouco
  informativos ("Vision Transformers Need Registers", Darcet et al., 2023). Para afirmar que
  a vinheta é um atalho, é preciso uma intervenção (ex.: mascarar os patches escuros e ver se a
  predição muda); sem ela, é hipótese.
* Os mapas são exibidos com escala por imagem: acompanhe cada um de `concentracao` (1 = uniforme).
* Teste de sanidade (Adebayo et al., 2018): compare com o mapa de um ViT com pesos aleatórios.

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

CMAP = "inferno"  # perceptualmente uniforme (o "jet" cria bordas falsas entre as cores)


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

    Devolve (probs, atencoes, imagens_normalizadas, rotulos). Os rótulos vêm do próprio dataset,
    para os títulos não dependerem de um y_true global. Use poucos índices (≤ 16): no ViT-B as
    atenções de 12 camadas × 12 heads × 197² ocupam ~22 MB por imagem.
    """
    itens = [dataset[int(i)] for i in indices]  # Ham10000Dataset devolve (imagem, rótulo)
    x = torch.stack([img for img, _ in itens])
    rotulos = np.array([int(r) for _, r in itens])
    probs, atencoes = extrair_atencoes(modelo, x, device)
    return probs, atencoes, x.cpu(), rotulos


def desnormalizar(x, media=(0.5, 0.5, 0.5), desvio=(0.5, 0.5, 0.5)) -> np.ndarray:
    """Tensor (B, 3, H, W) ou (3, H, W) normalizado → numpy (…, H, W, 3) em [0, 1] para exibir.

    O padrão 0,5/0,5 vale para os dois modelos (mesma normalização do checkpoint pré-treinado).
    """
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


def _indice_camada(camada: int, n_camadas: int) -> int:
    """Aceita índices negativos (-1 = última) e valida o intervalo."""
    if not -n_camadas <= camada < n_camadas:
        raise IndexError(f"camada {camada} fora do intervalo para {n_camadas} camadas")
    return camada % n_camadas


def mapa_cls(atencoes: np.ndarray, camada: int = -1, head: int | str = "media") -> np.ndarray:
    """Linha do CLS sobre os patches numa camada: (B, g, g).

    head: um inteiro escolhe UMA head (o que R24/R26 pedem); "media" tira a média das heads.
    Cada mapa é renormalizado para somar 1 nos patches: a atenção do CLS nele mesmo sai (veja
    `massa_no_cls` para saber quanto do fluxo o mapa representa).
    """
    a = atencoes[_indice_camada(camada, atencoes.shape[0])]  # (B, H, T, T)
    a = a.mean(axis=1) if head == "media" else a[:, head]
    linha = a[:, 0, 1:]  # linha 0 = consulta do CLS; colunas 1: = patches
    linha = linha / linha.sum(axis=1, keepdims=True)
    g = _lado_grade(a.shape[-1])
    return linha.reshape(-1, g, g)


def massa_no_cls(atencoes: np.ndarray, camada: int = -1, head: int | str = "media") -> np.ndarray:
    """Fração da atenção do CLS que vai para ele mesmo, por imagem: (B,).

    Se for 0,8, o heatmap de `mapa_cls` descreve só os 20% restantes do fluxo.
    """
    a = atencoes[_indice_camada(camada, atencoes.shape[0])]
    a = a.mean(axis=1) if head == "media" else a[:, head]
    return a[:, 0, 0]


def concentracao(mapas: np.ndarray) -> np.ndarray:
    """Entropia normalizada de cada mapa, por imagem: (B,). 1 = uniforme; perto de 0 = focado.

    Os heatmaps são exibidos com escala por imagem, então um mapa quase uniforme e um focado
    ganham o mesmo "pico" visual; este número mostra a diferença que a figura esconde.
    """
    p = mapas.reshape(len(mapas), -1)
    p = p / p.sum(axis=1, keepdims=True)
    entropia = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)
    return entropia / np.log(p.shape[1])


def attention_rollout(atencoes: np.ndarray, peso_residual: float = 0.5,
                      ate_camada: int | None = None) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema, 2020): (B, g, g). Bônus B3.

    Em cada camada: média das heads, soma da identidade para representar a conexão residual
    (A' = (1 − p)·A + p·I); depois multiplica as camadas em ordem (A_L···A_1). A linha do CLS do
    produto mostra quanto cada patch de entrada chega ao CLS final, somando todos os caminhos.
    ate_camada aceita índices negativos (-1 = todas as camadas).
    """
    if ate_camada is not None:
        atencoes = atencoes[: _indice_camada(ate_camada, atencoes.shape[0]) + 1]
    L, B, H, T, _ = atencoes.shape
    identidade = np.eye(T, dtype=np.float64)
    rollout = np.broadcast_to(identidade, (B, T, T)).copy()
    for camada in range(L):
        m = atencoes[camada].mean(axis=1).astype(np.float64)  # (B, T, T)
        m = (1 - peso_residual) * m + peso_residual * identidade
        # Redundante em teoria ((1 − p)·A + p·I já soma 1 por linha); protege de erro numérico
        m = m / m.sum(axis=-1, keepdims=True)
        rollout = m @ rollout
    linha = rollout[:, 0, 1:]
    linha = linha / linha.sum(axis=1, keepdims=True)
    g = _lado_grade(T)
    return linha.reshape(B, g, g)


# ---------------------------------------------------------------------------------------------
# Atalho da vinheta: atenção nos patches escuros da PRÓPRIA imagem
# ---------------------------------------------------------------------------------------------

def patches_escuros(imagens: np.ndarray, g: int, limiar: float = 0.15) -> np.ndarray:
    """Máscara (B, g, g) dos patches com luminância média abaixo do limiar.

    `imagens` (B, H, W, 3) em [0, 1] (saída de `desnormalizar`), com H e W múltiplos de g. A
    vinheta do dermatoscópio é circular e ocupa sobretudo os cantos, e nem toda imagem a tem;
    por isso a máscara vem de cada imagem, e não de um anel fixo da grade.
    """
    B, H, W, _ = imagens.shape
    if H % g or W % g:
        raise ValueError(f"imagem {H}×{W} não se divide numa grade {g}×{g}")
    lum = imagens.mean(axis=-1).reshape(B, g, H // g, g, W // g).mean(axis=(2, 4))
    return lum < limiar


def fracao_em_patches_escuros(mapas: np.ndarray, escuros: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(observada, referência) por imagem: (B,) cada.

    observada = fração da atenção nos patches escuros; referência = fração de patches escuros
    (o que uma atenção uniforme daria). observada ≫ referência sugere atração pela vinheta.
    Imagens sem nenhum patch escuro recebem NaN nas duas (separe-as na análise). Os dois
    modelos são comparáveis porque a máscara é calculada na grade de cada um sobre a mesma imagem.
    """
    if mapas.shape != escuros.shape:
        raise ValueError(f"mapas {mapas.shape} e máscara {escuros.shape} com formas diferentes")
    referencia = escuros.mean(axis=(1, 2)).astype(float)
    observada = (mapas * escuros).sum(axis=(1, 2)) / mapas.sum(axis=(1, 2))
    sem_escuro = referencia == 0
    observada[sem_escuro] = np.nan
    referencia[sem_escuro] = np.nan
    return observada, referencia


def distancia_media_atencao(atencoes: np.ndarray, tamanho_patch: int = 16,
                            tamanho_img: int | None = None) -> np.ndarray:
    """Distância média de atenção por camada e head: (L, H), em pixels da entrada. Bônus B3.

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
                      n_erros: int = 1, seed: int = 42, grupos=None) -> list[int]:
    """Índices do teste com acertos e erros de cada classe-alvo, sorteados com seed fixa.

    Use o y_pred de um modelo de referência (ex.: o pré-treinado) e passe os MESMOS índices aos
    dois modelos. Com `grupos` (lesion_id de cada imagem), nunca escolhe duas fotos da mesma
    lesão. Classes sem erros (ou sem acertos) contribuem com menos imagens. Para casos
    específicos (vinheta forte, régua), passe a lista de índices escolhida por critério explícito.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    grupos = None if grupos is None else np.asarray(grupos)
    rng = np.random.default_rng(seed)
    escolhidos, usados = [], set()
    for c in classes_alvo:
        da_classe = y_true == c
        for mascara, n in ((da_classe & (y_pred == c), n_acertos), (da_classe & (y_pred != c), n_erros)):
            candidatos = rng.permutation(np.flatnonzero(mascara))
            pegos = 0
            for i in candidatos:
                if pegos == n:
                    break
                if grupos is not None and grupos[i] in usados:
                    continue
                escolhidos.append(int(i))
                if grupos is not None:
                    usados.add(grupos[i])
                pegos += 1
    return escolhidos


# ---------------------------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------------------------

def _sobrepor(ax, imagem: np.ndarray, mapa: np.ndarray, alpha: float = 0.5,
              interpolacao: str = "nearest"):
    """Mostra o mapa g×g esticado sobre a imagem, com escala POR IMAGEM (0 a máximo do mapa).

    "nearest" (padrão) mostra os blocos da grade como são; "bilinear" suaviza e sugere um
    detalhe que a grade (8×8 ou 14×14) não tem.
    """
    h, w = imagem.shape[:2]
    ax.imshow(imagem, extent=(0, w, h, 0))
    ax.imshow(mapa / mapa.max(), cmap=CMAP, alpha=alpha, extent=(0, w, h, 0),
              interpolation=interpolacao, vmin=0, vmax=1)
    ax.axis("off")


def _barras_top3(ax, probs_por_modelo: Mapping[str, np.ndarray], k: int, classes: Sequence[str]):
    """Barras horizontais com as 3 classes mais prováveis de cada modelo para a imagem k."""
    y, ticks, rotulos = 0.0, [], []
    for cor, (nome, p) in enumerate(probs_por_modelo.items()):
        for c in np.argsort(p[k])[::-1][:3]:
            ax.barh(y, p[k, c], color=f"C{cor}")
            ticks.append(y)
            rotulos.append(f"{nome}: {sigla(classes[c])}")
            y += 1
        y += 0.5
    ax.set_yticks(ticks, rotulos, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("probabilidade", fontsize=8)


def plotar_comparacao(imagens: np.ndarray, mapas: Mapping[str, np.ndarray],
                      titulos_linhas: Sequence[str] | None = None, alpha: float = 0.5,
                      titulo: str = "CLS → patches (escala por imagem)",
                      probs_por_modelo: Mapping[str, np.ndarray] | None = None,
                      classes: Sequence[str] = CLASSES, interpolacao: str = "nearest"):
    """Uma linha por imagem: original + um mapa por modelo (+ barras top-3, se houver probs).

    `imagens` (B, H, W, 3) em [0, 1] (ex.: `desnormalizar` da entrada de 224 px); cada mapa em
    `mapas` tem forma (B, g, g) e é esticado ao tamanho da imagem, qualquer que seja o g. Cada
    painel mostra a concentração do mapa (1 = uniforme), porque a escala é por imagem.
    """
    B = len(imagens)
    nomes = list(mapas)
    n_col = 1 + len(nomes) + (probs_por_modelo is not None)
    fig, eixos = plt.subplots(B, n_col, figsize=(3.2 * n_col, 3.2 * B), squeeze=False)
    conc = {nome: concentracao(m) for nome, m in mapas.items()}
    for i in range(B):
        eixos[i, 0].imshow(imagens[i])
        eixos[i, 0].axis("off")
        if titulos_linhas is not None:
            eixos[i, 0].set_title(titulos_linhas[i], fontsize=9)
        for j, nome in enumerate(nomes, start=1):
            _sobrepor(eixos[i, j], imagens[i], mapas[nome][i], alpha, interpolacao)
            g = mapas[nome].shape[-1]
            cabecalho = f"{nome} (grade {g}×{g})\n" if i == 0 else ""
            eixos[i, j].set_title(f"{cabecalho}concentração {conc[nome][i]:.2f}", fontsize=9)
        if probs_por_modelo is not None:
            _barras_top3(eixos[i, -1], probs_por_modelo, i, classes)
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


def plotar_heads(atencoes: np.ndarray, imagem: np.ndarray, indice: int = 0, camada: int = -1,
                 alpha: float = 0.5, titulo: str | None = None, interpolacao: str = "nearest"):
    """Mapa do CLS de TODAS as heads de uma camada, para uma imagem.

    Mostrar todas (em vez de escolher a "mais interpretável" depois de vê-las) evita escolher
    o resultado; o título de cada painel traz a concentração e a massa no próprio CLS.
    """
    H = atencoes.shape[2]
    n_col = min(H, 6)
    n_lin = int(np.ceil(H / n_col))
    fig, eixos = plt.subplots(n_lin, n_col, figsize=(2.6 * n_col, 2.8 * n_lin), squeeze=False)
    for h, ax in enumerate(eixos.flat):
        if h < H:
            mapa = mapa_cls(atencoes, camada, h)[indice]
            _sobrepor(ax, imagem, mapa, alpha, interpolacao)
            ax.set_title(f"head {h} | conc. {concentracao(mapa[None])[0]:.2f}\n"
                         f"CLS→CLS {massa_no_cls(atencoes, camada, h)[indice]:.2f}", fontsize=8)
        else:
            ax.axis("off")
    n_camada = _indice_camada(camada, atencoes.shape[0])
    fig.suptitle(titulo or f"CLS → patches, camada {n_camada} (escala por imagem)")
    fig.tight_layout()
    return fig


def plotar_distancia(distancias: Mapping[str, np.ndarray], unidade: str = "px",
                     titulo: str = "Distância média de atenção"):
    """Um ponto por head, por camada, para cada modelo (entrada: saída de distancia_media_atencao).

    unidade: "px" ou "fração do lado" (quando `distancia_media_atencao` recebeu tamanho_img).
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (nome, d), marcador in zip(distancias.items(), "os^D"):
        L, H = d.shape
        camadas = np.repeat(np.arange(L), H) + 1
        ax.scatter(camadas, d.ravel(), label=nome, marker=marcador, alpha=0.7)
    ax.set_xlabel("Camada")
    ax.set_ylabel(f"Distância média ({unidade})")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def titulos_exemplos(rotulos, probs_por_modelo: Mapping[str, np.ndarray],
                     classes: Sequence[str] = CLASSES):
    """Títulos 'real: mel | zero: nv (0,81) | pré: mel (0,93)' para as linhas da comparação.

    `rotulos` e `probs_por_modelo[nome]` seguem a ordem dos índices usados em
    `atencoes_de_indices` (que devolve os rótulos junto).
    """
    titulos = []
    for k, r in enumerate(np.asarray(rotulos)):
        partes = [f"real: {sigla(classes[int(r)])}"]
        for nome, p in probs_por_modelo.items():
            c = int(p[k].argmax())
            partes.append(f"{nome}: {sigla(classes[c])} ({p[k, c]:.2f})")
        titulos.append("\n".join(partes))
    return titulos
