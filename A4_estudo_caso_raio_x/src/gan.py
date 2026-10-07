"""cGAN (DCGAN condicional) para raio-X de tórax 128×128 em tons de cinza: gerador e discriminador condicionados
na classe, treino adversarial com monitoramento, e métricas de diagnóstico (diversidade, memorização, nitidez)."""
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.dados import carregar_imagem

SN = nn.utils.spectral_norm


class Gerador(nn.Module):
    """z (dim_z) + embedding da classe -> 4×4×(8·base) -> 5 convoluções transpostas -> 128×128×1 em [-1, 1]."""

    def __init__(self, n_classes: int, dim_z: int = 128, base: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_classes, dim_z)
        self.inicio = nn.Sequential(nn.Linear(2 * dim_z, 4 * 4 * 8 * base), nn.BatchNorm1d(4 * 4 * 8 * base), nn.ReLU(True))
        self.base = base

        def bloco(c_in, c_out):
            return [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True)]

        self.rede = nn.Sequential(*bloco(8 * base, 8 * base), *bloco(8 * base, 4 * base), *bloco(4 * base, 2 * base),
                                  *bloco(2 * base, base), nn.ConvTranspose2d(base, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z, y):
        h = self.inicio(torch.cat([z, self.emb(y)], dim=1)).view(-1, 8 * self.base, 4, 4)
        return self.rede(h)


class Discriminador(nn.Module):
    """128×128×1 -> convoluções com stride 2 -> vetor; condicionamento por projeção (logit = w·h + <emb(y), h>).

    `spectral_norm=True` aplica normalização espectral em todas as camadas (mitigação de instabilidade): limita
    a constante de Lipschitz do discriminador, que deixa de ficar "forte demais" e de dar gradientes inúteis ao G.
    """

    def __init__(self, n_classes: int, base: int = 64, spectral_norm: bool = False):
        super().__init__()
        sn = SN if spectral_norm else (lambda m: m)

        def bloco(c_in, c_out, bn=True):
            camadas = [sn(nn.Conv2d(c_in, c_out, 4, 2, 1))]
            if bn and not spectral_norm:   # com SN, sem BatchNorm no D (combinação usual)
                camadas.append(nn.BatchNorm2d(c_out))
            return camadas + [nn.LeakyReLU(0.2, True)]

        self.rede = nn.Sequential(*bloco(1, base, bn=False), *bloco(base, 2 * base), *bloco(2 * base, 4 * base),
                                  *bloco(4 * base, 8 * base), *bloco(8 * base, 8 * base))   # 128 -> 4
        self.linear = sn(nn.Linear(8 * base * 4 * 4, 1))
        self.emb = sn(nn.Embedding(n_classes, 8 * base * 4 * 4))

    def forward(self, x, y):
        h = self.rede(x).flatten(1)
        return self.linear(h).squeeze(1) + (self.emb(y) * h).sum(dim=1)


class DatasetGAN(Dataset):
    """Raio-X 128×128 em [-1, 1] com o índice da classe (só imagens do TREINO)."""

    def __init__(self, caminhos, rotulos, lado: int = 128):
        t = transforms.Compose([transforms.Resize((lado, lado)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        # Pré-carregado em memória (1.200 × 128 × 128 ≈ 80 MB): sem decodificar os PNGs a cada época
        self.imagens = torch.stack([t(carregar_imagem(c, "L")) for c in caminhos])
        self.rotulos = list(rotulos)

    def __len__(self):
        return len(self.rotulos)

    def __getitem__(self, i):
        return self.imagens[i], self.rotulos[i]


def treinar_cgan(caminhos, rotulos, n_classes: int, config: dict, mitigacao: bool, seed: int, device, ao_fim_da_epoca=None):
    """Treino adversarial (BCE com logits): a cada passo, 1 atualização do D (real × gerado) e 1 do G.

    Sem mitigação: DCGAN padrão (BatchNorm no D, rótulo real = 1, mesmo lr para D e G).
    Com mitigação: normalização espectral no D + suavização unilateral do rótulo real (0,9) + TTUR
    (lr do D maior que o do G). Amostragem balanceada por classe nas duas versões (o COVID tem só 120 imagens).

    Devolve (G, D, histórico por época). O checkpoint usado é o da ÚLTIMA época (número de épocas fixado antes).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rotulos_t = torch.tensor(rotulos)
    pesos = (1.0 / torch.bincount(rotulos_t, minlength=n_classes).float())[rotulos_t]
    amostrador = WeightedRandomSampler(pesos, num_samples=len(rotulos), replacement=True,
                                       generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(DatasetGAN(caminhos, rotulos, config["lado"]), batch_size=config["tamanho_lote"],
                        sampler=amostrador, num_workers=config["num_workers"], drop_last=True)

    G = Gerador(n_classes, config["dim_z"], config["base"]).to(device)
    D = Discriminador(n_classes, config["base"], spectral_norm=mitigacao).to(device)
    lr_d = config["lr_d_mitigacao"] if mitigacao else config["lr"]
    lr_g = config["lr_g_mitigacao"] if mitigacao else config["lr"]
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    alvo_real = 0.9 if mitigacao else 1.0

    historico = []
    for epoca in range(1, config["epocas"] + 1):
        t0 = time.perf_counter()
        G.train(); D.train()
        soma = {"loss_d": 0.0, "loss_g": 0.0, "d_real": 0.0, "d_falso": 0.0}
        n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            b = x.shape[0]
            z = torch.randn(b, config["dim_z"], device=device)
            falso = G(z, y)
            # D: real -> alvo_real, gerado -> 0
            logit_real, logit_falso = D(x, y), D(falso.detach(), y)
            loss_d = bce(logit_real, torch.full_like(logit_real, alvo_real)) + bce(logit_falso, torch.zeros_like(logit_falso))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            # G: quer que o D chame o gerado de real (perda não saturante)
            logit_g = D(falso, y)
            loss_g = bce(logit_g, torch.ones_like(logit_g))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            soma["loss_d"] += loss_d.item(); soma["loss_g"] += loss_g.item()
            soma["d_real"] += torch.sigmoid(logit_real).mean().item(); soma["d_falso"] += torch.sigmoid(logit_falso).mean().item()
            n += 1
        linha = {"epoca": epoca, **{k: v / n for k, v in soma.items()}, "segundos": round(time.perf_counter() - t0, 1)}
        if ao_fim_da_epoca is not None:
            linha.update(ao_fim_da_epoca(G, epoca) or {})
        historico.append(linha)
    return G, D, historico


@torch.inference_mode()
def gerar(G: nn.Module, classe: int, n: int, dim_z: int, seed: int, device, lote: int = 128) -> torch.Tensor:
    """n imagens geradas da classe, em [0, 1], shape (n, 1, lado, lado), na CPU."""
    G.eval()
    gerador = torch.Generator(device="cpu").manual_seed(seed)
    saidas = []
    for ini in range(0, n, lote):
        m = min(lote, n - ini)
        z = torch.randn(m, dim_z, generator=gerador).to(device)
        y = torch.full((m,), classe, dtype=torch.long, device=device)
        saidas.append(((G(z, y) + 1) / 2).clamp(0, 1).cpu())
    return torch.cat(saidas)


def vetores_de_tensores(imgs: torch.Tensor, lado: int = 64) -> torch.Tensor:
    """Mesmo vetor de vetores_pixels (64×64, centrado, norma 1), a partir de tensores (n, 1, H, W) em [0, 1]."""
    v = nn.functional.interpolate(imgs.float(), size=(lado, lado), mode="bilinear", align_corners=False).flatten(1) * 255
    v = v - v.mean(dim=1, keepdim=True)
    return nn.functional.normalize(v, dim=1)


def diversidade(v: torch.Tensor) -> float:
    """Correlação média entre pares distintos (mais alta = menos diversidade; colapso de modo -> perto de 1)."""
    s = v @ v.T
    n = len(v)
    return float((s.sum() - s.diagonal().sum()) / (n * (n - 1)))


def nitidez(imgs: torch.Tensor) -> np.ndarray:
    """Variância do laplaciano de cada imagem (n, 1, H, W) em [0, 1]: imagens borradas têm valor baixo."""
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    return nn.functional.conv2d(imgs.float(), k).flatten(1).var(dim=1).numpy()
