"""
Vision Transformer implementado do zero em PyTorch (A1, R11–R19).

Blocos básicos permitidos: nn.Linear, nn.LayerNorm, nn.Conv2d, nn.Dropout. Nada de
nn.MultiheadAttention, nn.TransformerEncoderLayer, timm ou transformers.

Configuração base do projeto: imagem 128×128, patch 16 -> grade 8×8 = 64 tokens + CLS = 65.
"""
import math

import torch
import torch.nn as nn


def atencao_produto_escalado(q, k, v, mascara=None):
    """
    Scaled dot-product attention: softmax(Q Kᵀ / √d_k) V.

    Funciona com qualquer número de dimensões à esquerda (ex.: (B, N, d_k) para uma head,
    ou (B, h, N, d_k) para várias), porque só usa as duas últimas.

    Args:
        q: (..., N_q, d_k)
        k: (..., N_k, d_k)
        v: (..., N_k, d_v)
        mascara: opcional, broadcastável para (..., N_q, N_k). Convenção: 0/False = posição
            ignorada, 1/True = posição que pode ser atendida (igual à máscara booleana de
            torch.nn.functional.scaled_dot_product_attention). O ViT não usa máscara: todas
            as sequências têm o mesmo tamanho.

    Returns:
        (saida, pesos): saida (..., N_q, d_v) e pesos (..., N_q, N_k), com cada linha dos
        pesos somando 1 (a linha i responde "para onde o token i olha").
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)          # (..., N_q, N_k)
    if mascara is not None:
        scores = scores.masked_fill(mascara == 0, float("-inf"))  # antes do softmax -> peso 0
    pesos = torch.softmax(scores, dim=-1)
    return pesos @ v, pesos


class ScaledDotProductAttention(nn.Module):
    """Módulo PyTorch (sem parâmetros) da scaled dot-product attention — R11."""

    def forward(self, q, k, v, mascara=None):
        return atencao_produto_escalado(q, k, v, mascara)


class CabecaAtencao(nn.Module):
    """
    Uma attention head com as SUAS PRÓPRIAS projeções: W_i^Q, W_i^K, W_i^V ∈ R^{d×d_k}.
    head_i = Attention(X W_i^Q, X W_i^K, X W_i^V).
    """

    def __init__(self, d: int, d_k: int):
        super().__init__()
        self.w_q = nn.Linear(d, d_k)
        self.w_k = nn.Linear(d, d_k)
        self.w_v = nn.Linear(d, d_k)
        self.atencao = ScaledDotProductAttention()

    def forward(self, x, mascara=None):
        """x: (B, N, d) -> saida (B, N, d_k), pesos (B, N, N)."""
        return self.atencao(self.w_q(x), self.w_k(x), self.w_v(x), mascara)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention com projeções independentes por head (R12):
    MHA(X) = Concat(head_1, ..., head_h) W^O, com d_k = d / h.

    Cada head é um módulo separado (nn.ModuleList), sem parâmetros compartilhados.
    W^O mistura as heads depois da concatenação.
    """

    def __init__(self, d: int, h: int, dropout: float = 0.0):
        super().__init__()
        assert d % h == 0, "d precisa ser divisível por h"
        self.h, self.d_k = h, d // h
        self.heads = nn.ModuleList(CabecaAtencao(d, self.d_k) for _ in range(h))
        self.w_o = nn.Linear(h * self.d_k, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mascara=None):
        """x: (B, N, d) -> saida (B, N, d), pesos (B, h, N, N)."""
        saidas, pesos = zip(*(head(x, mascara) for head in self.heads))
        concatenada = torch.cat(saidas, dim=-1)                  # (B, N, h·d_k) = (B, N, d)
        return self.drop(self.w_o(concatenada)), torch.stack(pesos, dim=1)


class TransformerEncoderBlock(nn.Module):
    """
    Bloco do encoder no formato pré-norm do ViT (R14):
        x = x + MHA(LayerNorm(x))
        x = x + FFN(LayerNorm(x)),  FFN = Linear(d, mlp) -> GELU -> Dropout -> Linear(mlp, d)
    O Transformer original (Vaswani et al., 2017) aplicava a LayerNorm depois da soma
    (pós-norm); o ViT usa pré-norm, que estabiliza o treino.
    """

    def __init__(self, d: int, h: int, mlp: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.atencao = MultiHeadAttention(d, h, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, mlp), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp, d), nn.Dropout(dropout)
        )

    def forward(self, x, mascara=None):
        """x: (B, N, d) -> x (B, N, d), pesos (B, h, N, N)."""
        a, pesos = self.atencao(self.ln1(x), mascara)
        x = x + a                                                # residual 1
        x = x + self.ffn(self.ln2(x))                            # residual 2
        return x, pesos


class PatchEmbedding(nn.Module):
    """
    Recorta a imagem em patches P×P e projeta cada patch em d dimensões (R15).

    Conv2d com kernel = stride = P é equivalente a "achatar cada patch (P·P·C valores) e
    aplicar o mesmo Linear(P·P·C, d)": cada filtro vê exatamente um patch, sem sobreposição.
    """

    def __init__(self, img: int, patch: int, canais: int, d: int):
        super().__init__()
        assert img % patch == 0, "O tamanho da imagem precisa ser múltiplo do patch"
        self.grade = img // patch                                # patches por lado
        self.n_patches = self.grade ** 2
        self.proj = nn.Conv2d(canais, d, kernel_size=patch, stride=patch)

    def forward(self, x):
        """x: (B, C, img, img) -> tokens (B, grade², d), em ordem de leitura (linha a linha)."""
        return self.proj(x).flatten(2).transpose(1, 2)


def pe_senoidal(n_posicoes: int, d: int) -> torch.Tensor:
    """Positional encoding senoidal fixo (Vaswani et al., 2017): (1, n_posicoes, d). d par."""
    assert d % 2 == 0, "o PE senoidal exige d par"
    pos = torch.arange(n_posicoes).unsqueeze(1)
    freq = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(n_posicoes, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe.unsqueeze(0)


class VisionTransformer(nn.Module):
    """
    ViT completo (R19): imagem (B, C, img, img) -> logits (B, n_classes), mais a lista com os
    pesos de atenção de cada camada, cada um (B, h, N, N) com N = grade² + 1 (CLS na posição 0).

    Positional encoding (R17), somado aos tokens depois de concatenar o CLS:
      - "aprendivel": nn.Parameter (como no ViT original; é treinado);
      - "senoidal": fixo, registrado como buffer (vai para a GPU com o modelo, não é treinado).
    """

    def __init__(self, n_classes: int = 7, img: int = 128, patch: int = 16, canais: int = 3,
                 d: int = 192, h: int = 3, profundidade: int = 6, mlp: int = 768,
                 dropout: float = 0.1, tipo_pe: str = "aprendivel"):
        super().__init__()
        assert tipo_pe in ("aprendivel", "senoidal"), f"tipo_pe inválido: {tipo_pe}"
        self.img, self.tipo_pe = img, tipo_pe

        self.patches = PatchEmbedding(img, patch, canais, d)
        self.grade = self.patches.grade                          # para reconstruir o mapa de atenção
        n_tokens = self.patches.n_patches + 1                    # +1 pelo CLS

        self.cls = nn.Parameter(torch.zeros(1, 1, d))            # CLS aprendível (R16)
        if tipo_pe == "aprendivel":
            self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        else:
            self.register_buffer("pos", pe_senoidal(n_tokens, d))

        self.drop = nn.Dropout(dropout)
        self.blocos = nn.ModuleList(
            TransformerEncoderBlock(d, h, mlp, dropout) for _ in range(profundidade)
        )
        self.ln = nn.LayerNorm(d)
        self.cabeca = nn.Linear(d, n_classes)
        self._inicializar()

    def _inicializar(self):
        """Inicialização do ViT de referência: trunc_normal(0,02) em Linear/CLS/PE aprendível."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.cls, std=0.02)
        if self.tipo_pe == "aprendivel":
            nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        assert x.shape[-2:] == (self.img, self.img), (
            f"imagem {tuple(x.shape[-2:])}, mas o modelo foi criado com img={self.img}"
        )
        tokens = self.patches(x)                                 # (B, grade², d)
        cls = self.cls.expand(x.size(0), -1, -1)                 # (B, 1, d)
        z = self.drop(torch.cat([cls, tokens], dim=1) + self.pos)  # (B, grade² + 1, d)

        todos_pesos = []
        for bloco in self.blocos:
            z, pesos = bloco(z)
            todos_pesos.append(pesos)                            # (B, h, N, N) por camada
        return self.cabeca(self.ln(z[:, 0])), todos_pesos        # classifica só pelo CLS
