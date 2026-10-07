"""COVID-19 Radiography Database (Kaggle): verificação do download, listagem, fonte das imagens, carregamento
e split com escassez no treino."""
import hashlib
import re
from pathlib import Path

import pandas as pd
from PIL import Image

# Versão 5 do dataset no Kaggle (tawsifurrahman/covid19-radiography-database), verificada pela impressão
# digital (calculada pela listagem do Kaggle antes do download).
VERSAO_DATASET = 5
N_ARQUIVOS_DATASET = 42335
ASSINATURA_DATASET = "81edf2e3981be9c4"

PASTA_RAIZ = "COVID-19_Radiography_Dataset"   # <classe>/images/*.png, <classe>/masks/*.png, <classe>.metadata.xlsx

SPLIT_TREINO, SPLIT_VALIDACAO, SPLIT_TESTE = "treino", "validacao", "teste"


def impressao_digital(raiz: Path):
    """Nº de arquivos, tamanho total (MB) e hash da lista (caminho relativo, tamanho).

    No Colab, o kagglehub pode servir o cache do próprio Colab (/kaggle/input/...), cujo caminho não
    mostra a versão; a impressão digital é o que garante que os dados são os mesmos.
    """
    arquivos = sorted(p for p in raiz.rglob("*") if p.is_file())
    h = hashlib.sha256()
    total = 0
    for p in arquivos:
        tamanho = p.stat().st_size
        total += tamanho
        h.update(f"{p.relative_to(raiz).as_posix()}|{tamanho}\n".encode())
    return len(arquivos), round(total / 1e6, 1), h.hexdigest()[:16]


def arvore(raiz: Path, nivel_max: int = 3, nivel: int = 0):
    """Imprime as pastas até `nivel_max` níveis, com o nº de arquivos de cada uma."""
    for p in sorted(raiz.iterdir()):
        if p.is_dir():
            n = sum(1 for f in p.rglob("*") if f.is_file())
            print("    " * nivel + f"{p.name}/  ({n} arquivos)")
            if nivel + 1 < nivel_max:
                arvore(p, nivel_max, nivel + 1)


def carregar_imagem(caminho, modo: str = "L") -> Image.Image:
    """Abre um raio-X no modo pedido ("L" = tons de cinza, para a GAN; "RGB" = 3 canais iguais, para a
    ResNet pré-treinada). Usada em todo o notebook."""
    with Image.open(caminho) as img:
        return img.convert(modo)


def vetores_pixels(caminhos, lado: int = 64):
    """Cada imagem reduzida a `lado`×`lado` em tons de cinza, centrada e com norma 1 (tensor N × lado²).

    O produto interno entre dois vetores é a correlação de Pearson dos pixels: perto de 1 só para a mesma
    foto (recomprimida, redimensionada, com outro contraste). Features semânticas (ex.: ResNet do ImageNet)
    não servem para isso em raio-X: todos os tórax ficam parecidos (cosseno ~0,98 entre pacientes
    diferentes) e os grupos de "duplicatas" se encadeiam.
    """
    import numpy as np
    import torch

    vetores = np.empty((len(caminhos), lado * lado), dtype=np.float32)
    for i, caminho in enumerate(caminhos):
        a = np.asarray(carregar_imagem(caminho, "L").resize((lado, lado), Image.BILINEAR), dtype=np.float32).ravel()
        a -= a.mean()
        norma = np.linalg.norm(a)
        vetores[i] = a / norma if norma > 0 else a
    return torch.from_numpy(vetores)


def _fonte(url: str) -> str:
    """Resumo da origem a partir da URL dos metadados: domínio e, no Kaggle, o dataset/competição."""
    url = str(url).strip()
    m = re.match(r"https?://(?:www\.)?([^/]+)/?(.*)", url)
    if not m:
        return url or "desconhecida"
    dominio, caminho = m.groups()
    if dominio == "kaggle.com":
        partes = [p for p in caminho.split("/") if p]
        return "kaggle.com/" + "/".join(partes[:2])
    return dominio


def ler_metadados(raiz: Path, classes) -> pd.DataFrame:
    """Uma linha por imagem dos metadados de cada classe (FILE NAME, URL), com a fonte resumida."""
    tabelas = []
    for classe in classes:
        df = pd.read_excel(raiz / PASTA_RAIZ / f"{classe}.metadata.xlsx")
        tabelas.append(pd.DataFrame({"classe": classe, "nome": df["FILE NAME"].astype(str).str.strip().str.lower(),
                                     "url": df["URL"].astype(str), "fonte": df["URL"].map(_fonte)}))
    return pd.concat(tabelas, ignore_index=True)


def listar_imagens(raiz: Path, classes) -> pd.DataFrame:
    """Uma linha por imagem (pasta images/; as máscaras não são usadas): classe, tamanho, modo e md5."""
    registros = []
    for classe in classes:
        for p in sorted((raiz / PASTA_RAIZ / classe / "images").glob("*.png")):
            with Image.open(p) as img:
                largura, altura = img.size
                modo = img.mode
            registros.append({"caminho_rel": p.relative_to(raiz).as_posix(), "classe": classe, "arquivo": p.name,
                              "nome": p.stem.strip().lower(), "largura": largura, "altura": altura, "modo": modo,
                              "md5": hashlib.md5(p.read_bytes()).hexdigest()})
    return pd.DataFrame(registros)


def remover_duplicatas(df: pd.DataFrame):
    """Remove duplicatas exatas (mesmo md5). Mesmo conteúdo em classes diferentes é conflito de rótulo:
    todas as cópias saem. Mesmo conteúdo na mesma classe: fica a primeira.

    Devolve (df sem duplicatas, df das linhas duplicadas, md5s em conflito).
    """
    duplicadas = df[df.duplicated("md5", keep=False)]
    classes_por_md5 = duplicadas.groupby("md5")["classe"].nunique()
    conflitos = set(classes_por_md5[classes_por_md5 > 1].index)
    limpo = df[~df["md5"].isin(conflitos)].drop_duplicates("md5", keep="first").reset_index(drop=True)
    return limpo, duplicadas, conflitos


def dividir_com_escassez(df: pd.DataFrame, n_treino: dict, n_validacao: dict, n_teste: dict, seed: int) -> pd.DataFrame:
    """Coluna `split` com quantidades fixas por classe: primeiro o teste, depois a validação e por fim o
    treino, cada um sorteado (sem reposição) do que sobrou da classe. O resto fica sem split (não usado).

    Assim a escassez do projeto legado fica só no treino, e o teste é real e maior.
    """
    saida = df.copy()
    saida["split"] = None
    for classe, grupo in saida.groupby("classe"):
        restante = grupo
        for split, quantos in ((SPLIT_TESTE, n_teste), (SPLIT_VALIDACAO, n_validacao), (SPLIT_TREINO, n_treino)):
            n = quantos[classe]
            assert n <= len(restante), f"{classe}: pedidas {n} imagens para {split}, só há {len(restante)}"
            escolhidas = restante.sample(n=n, random_state=seed)
            saida.loc[escolhidas.index, "split"] = split
            restante = restante.drop(escolhidas.index)
    return saida
