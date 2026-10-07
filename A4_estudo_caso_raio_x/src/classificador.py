"""Classificador ResNet-18 pré-treinada para Normal / Viral Pneumonia / COVID: transforms, datasets,
treino com seleção pela validação, predições por imagem e extração de features (quase-duplicatas)."""
import copy
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.dados import carregar_imagem

MEDIA_IMAGENET, DESVIO_IMAGENET = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def transforms_raio_x(resolucao: int = 224):
    """(treino, avaliação). Raio-X em tons de cinza repetido em 3 canais (a ResNet foi pré-treinada em RGB)
    e normalizado com a média/desvio do ImageNet.

    Augmentation leve e igual em todas as condições: pequena rotação, translação e escala (posição do
    paciente). Sem flip horizontal: no raio-X de tórax o coração fica à esquerda, e o flip criaria uma
    anatomia que não existe.
    """
    base = [transforms.Resize((resolucao, resolucao)), transforms.Grayscale(num_output_channels=3)]
    final = [transforms.ToTensor(), transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET)]
    treino = transforms.Compose(base + [transforms.RandomAffine(degrees=5, translate=(0.03, 0.03),
                                                                scale=(0.95, 1.05))] + final)
    avaliacao = transforms.Compose(base + final)
    return treino, avaliacao


class DatasetRaioX(Dataset):
    """(imagem transformada, índice da classe) a partir de listas de caminhos e rótulos."""

    def __init__(self, caminhos, rotulos, transform):
        assert len(caminhos) == len(rotulos)
        self.caminhos, self.rotulos, self.transform = list(caminhos), list(rotulos), transform

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, i):
        return self.transform(carregar_imagem(self.caminhos[i], "L")), self.rotulos[i]


def criar_resnet18(n_classes: int) -> nn.Module:
    """ResNet-18 com pesos do ImageNet (torchvision IMAGENET1K_V1) e head trocado por Linear(512, n_classes).
    Toda a rede é ajustada (fine-tuning), com lr baixo."""
    modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    modelo.fc = nn.Linear(modelo.fc.in_features, n_classes)
    return modelo


def _semear(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def prever(modelo: nn.Module, loader: DataLoader, device) -> np.ndarray:
    """Probabilidades (N, n_classes) na ordem do loader."""
    modelo.eval()
    saidas = []
    for x, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(str(device) == "cuda")):
            logits = modelo(x.to(device))
        saidas.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(saidas).numpy()


def recall_por_classe(y_true, y_pred, n_classes: int) -> np.ndarray:
    return np.array([(y_pred[y_true == k] == k).mean() if (y_true == k).any() else np.nan for k in range(n_classes)])


def treinar_classificador(caminhos_treino, rotulos_treino, caminhos_val, rotulos_val, config: dict, seed: int,
                          device, n_classes: int):
    """Um treino completo com seleção do melhor estado pela validação (critério fixo: config["criterio"]).

    Devolve (modelo com o melhor estado, histórico por época, melhor época).
    """
    _semear(seed)
    t_treino, t_aval = transforms_raio_x(config["resolucao"])
    gerador = torch.Generator().manual_seed(seed)
    loader_treino = DataLoader(DatasetRaioX(caminhos_treino, rotulos_treino, t_treino), batch_size=config["tamanho_lote"],
                               shuffle=True, generator=gerador, num_workers=config["num_workers"],
                               worker_init_fn=lambda w: np.random.seed(seed * 100 + w))
    loader_val = DataLoader(DatasetRaioX(caminhos_val, rotulos_val, t_aval), batch_size=config["tamanho_lote"] * 2,
                            shuffle=False, num_workers=config["num_workers"])
    y_val = np.asarray(rotulos_val)

    modelo = criar_resnet18(n_classes).to(device)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterio = nn.CrossEntropyLoss()
    escalador = torch.amp.GradScaler(enabled=(str(device) == "cuda"))

    historico, melhor, melhor_epoca, melhor_estado = [], -1.0, None, None
    for epoca in range(1, config["epocas"] + 1):
        t0 = time.perf_counter()
        modelo.train()
        soma, n = 0.0, 0
        for x, y in loader_treino:
            x, y = x.to(device), y.to(device)
            otimizador.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(str(device) == "cuda")):
                loss = criterio(modelo(x), y)
            escalador.scale(loss).backward()
            escalador.step(otimizador)
            escalador.update()
            soma += loss.item() * len(y)
            n += len(y)
        probs_val = prever(modelo, loader_val, device)
        rec = recall_por_classe(y_val, probs_val.argmax(1), n_classes)
        valor = {"recall_medio": np.nanmean(rec), "recall_covid": rec[-1]}[config["criterio"]]
        historico.append({"epoca": epoca, "loss_treino": soma / n, "val_recall_medio": float(np.nanmean(rec)),
                          **{f"val_recall_{k}": float(r) for k, r in enumerate(rec)},
                          "segundos": round(time.perf_counter() - t0, 1)})
        if valor > melhor:   # deepcopy: state_dict() devolve tensores ligados aos parâmetros
            melhor, melhor_epoca, melhor_estado = valor, epoca, copy.deepcopy(modelo.state_dict())
    modelo.load_state_dict(melhor_estado)
    return modelo, pd.DataFrame(historico), melhor_epoca


def extrator_resnet18(device) -> nn.Module:
    """ResNet-18 do ImageNet sem o head (vetor de 512), em eval: só para medir similaridade entre imagens."""
    modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    modelo.fc = nn.Identity()
    return modelo.to(device).eval()


@torch.inference_mode()
def extrair_features(extrator: nn.Module, caminhos, device, resolucao: int = 224, tamanho_lote: int = 128,
                     num_workers: int = 2) -> torch.Tensor:
    """Features L2-normalizadas (N, 512), na ordem de `caminhos` (CPU)."""
    _, t_aval = transforms_raio_x(resolucao)
    loader = DataLoader(DatasetRaioX(caminhos, [0] * len(caminhos), t_aval), batch_size=tamanho_lote,
                        shuffle=False, num_workers=num_workers)
    feats = []
    for x, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(str(device) == "cuda")):
            f = extrator(x.to(device))
        feats.append(nn.functional.normalize(f.float(), dim=1).cpu())
    return torch.cat(feats)
