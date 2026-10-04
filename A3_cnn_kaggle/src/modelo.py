"""CNN pré-treinada como extrator de features: carregamento, congelamento do backbone, novo head e
extração das features (o backbone nunca é treinado)."""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from src.dados import carregar_imagem


def carregar_backbone(nome: str, pesos: str, carregar_pesos: bool = True):
    """Modelo do torchvision com os pesos indicados e as transforms DOS PRÓPRIOS PESOS.

    As transforms (resize, crop, normalização com a média/desvio do ImageNet) vêm junto com os pesos:
    o backbone só funciona bem com a mesma entrada que viu no pré-treino. `carregar_pesos=False`
    monta só a arquitetura (para testes, sem download).
    """
    enum_pesos = models.get_model_weights(nome)[pesos]
    modelo = models.get_model(nome, weights=enum_pesos if carregar_pesos else None)
    return modelo, enum_pesos, enum_pesos.transforms()


def preparar_feature_extraction(modelo: nn.Module, n_classes: int) -> dict:
    """Congela o backbone e troca o head (ResNet: `fc`) por Linear(features, n_classes).

    * requires_grad=False em todo o backbone: o otimizador só recebe os parâmetros do novo head;
    * eval() no backbone: além de desligar o dropout, faz o BatchNorm usar as estatísticas do
      pré-treino. Em train(), o BatchNorm atualizaria running_mean/running_var a cada lote, e o
      backbone mudaria sem nenhum gradiente.
    Devolve as contagens de parâmetros (total, congelados, treináveis).
    """
    for p in modelo.parameters():
        p.requires_grad = False
    n_features = modelo.fc.in_features
    modelo.fc = nn.Linear(n_features, n_classes)   # parâmetros novos: requires_grad=True
    modelo.eval()
    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return {"n_features": n_features, "total": total, "congelados": total - treinaveis, "treinaveis": treinaveis}


def backbone_sem_head(modelo: nn.Module) -> nn.Module:
    """Tudo antes do `fc` (até o average pooling), achatado: a imagem vira o vetor de features."""
    return nn.Sequential(*list(modelo.children())[:-1], nn.Flatten())


class DatasetImagens(Dataset):
    """(imagem transformada, índice da classe) a partir das colunas `caminho` e `classe`."""

    def __init__(self, df, transform, classes):
        self.caminhos = df["caminho"].tolist()
        self.rotulos = [classes.index(c) for c in df["classe"]]
        self.transform = transform

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, i):
        return self.transform(carregar_imagem(self.caminhos[i])), self.rotulos[i]


@torch.inference_mode()
def extrair_features(extrator: nn.Module, loader: DataLoader, device) -> tuple:
    """Features (N, n_features) e rótulos (N,), na ordem do loader (sem shuffle), na CPU.

    Como o backbone é fixo e não há augmentation, as features de cada imagem são sempre as mesmas:
    basta extraí-las uma vez e treinar o head sobre elas.
    """
    extrator.eval()
    feats, rotulos = [], []
    for x, y in loader:
        feats.append(extrator(x.to(device)).float().cpu())
        rotulos.append(y)
    return torch.cat(feats), torch.cat(rotulos)
