"""Dataset de imagens do Kaggle (pavansanagapati/images-dataset): verificação do download, listagem de
uma cópia, carregamento das imagens e split estratificado."""
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Versão 1 do dataset no Kaggle (única; licença CC0), verificada pela impressão digital.
VERSAO_DATASET = 1
N_ARQUIVOS_DATASET = 3606
ASSINATURA_DATASET = "1d9391889f4f5ae0"

# O dataset traz duas cópias das mesmas imagens: data/<classe>/ e data/data/<classe>/.
PASTA_COPIA = "data"                 # a cópia usada
PASTA_COPIA_DUPLICADA = "data/data"  # a cópia descartada (idêntica, ver comparar_copias)
CLASSES = ["bike", "cars", "cats", "dogs", "flowers", "horses", "human"]

SPLIT_TREINO, SPLIT_VALIDACAO, SPLIT_TESTE = "treino", "validacao", "teste"

# Modos de cor do PIL que podem ter canal alfa
_MODOS_COM_ALFA = ("RGBA", "LA", "PA")


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


def _md5(caminho: Path) -> str:
    return hashlib.md5(caminho.read_bytes()).hexdigest()


def comparar_copias(raiz: Path) -> dict:
    """Compara as duas cópias (mesmos nomes relativos e mesmo conteúdo em bytes, por md5)."""
    copia = {p.relative_to(raiz / PASTA_COPIA).as_posix(): p
             for p in (raiz / PASTA_COPIA).glob("*/*") if p.is_file()}
    duplicada = {p.relative_to(raiz / PASTA_COPIA_DUPLICADA).as_posix(): p
                 for p in (raiz / PASTA_COPIA_DUPLICADA).glob("*/*") if p.is_file()}
    comuns = sorted(set(copia) & set(duplicada))
    diferentes = [n for n in comuns if _md5(copia[n]) != _md5(duplicada[n])]
    return {"na_copia": len(copia), "na_duplicada": len(duplicada),
            "so_na_copia": sorted(set(copia) - set(duplicada)),
            "so_na_duplicada": sorted(set(duplicada) - set(copia)),
            "conteudo_diferente": diferentes}


def _pode_ter_alfa(img: Image.Image) -> bool:
    return img.mode in _MODOS_COM_ALFA or (img.mode == "P" and "transparency" in img.info)


def tem_transparencia(img: Image.Image) -> bool:
    """True se a imagem tem algum pixel não totalmente opaco (alfa < 255)."""
    if not _pode_ter_alfa(img):
        return False
    return img.convert("RGBA").getchannel("A").getextrema()[0] < 255


def carregar_imagem(caminho) -> Image.Image:
    """Abre uma imagem como RGB. Usada em todo o notebook (EDA, extração de features, erros).

    convert("RGB") direto descarta o alfa: o pixel transparente fica com a cor que estava "por baixo"
    (muitas vezes preto). Por isso RGBA/P/LA são compostos sobre fundo branco antes da conversão; numa
    imagem com alfa 255 em todos os pixels, a composição não muda nada.
    """
    with Image.open(caminho) as img:
        img.load()
        if _pode_ter_alfa(img):
            rgba = img.convert("RGBA")
            fundo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(fundo, rgba).convert("RGB")
        return img.convert("RGB")


def listar_imagens(raiz: Path) -> pd.DataFrame:
    """Uma linha por imagem da cópia usada (data/<classe>/<arquivo>): classe, formato, tamanho, modo de
    cor, transparência e md5."""
    registros = []
    for p in (raiz / PASTA_COPIA).glob("*/*"):
        if not p.is_file():
            continue
        with Image.open(p) as img:
            largura, altura = img.size
            modo = img.mode
            formato = img.format
            transparente = tem_transparencia(img)
        registros.append({"caminho_rel": p.relative_to(raiz).as_posix(), "classe": p.parent.name,
                          "arquivo": p.name, "extensao": p.suffix.lower(), "formato": formato,
                          "largura": largura, "altura": altura, "modo": modo,
                          "transparente": transparente, "md5": _md5(p)})
    return pd.DataFrame(registros).sort_values(["classe", "arquivo"]).reset_index(drop=True)


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


def dividir_estratificado(df: pd.DataFrame, fracoes=(0.70, 0.15, 0.15), seed: int = 42) -> pd.DataFrame:
    """Coluna `split` (treino/validacao/teste), estratificada por classe, em dois cortes: primeiro o
    teste, depois a validação dentro do restante (como na A1)."""
    f_treino, f_val, f_teste = fracoes
    assert abs(f_treino + f_val + f_teste - 1) < 1e-9, fracoes
    resto, teste = train_test_split(df, test_size=f_teste, stratify=df["classe"], random_state=seed)
    treino, val = train_test_split(resto, test_size=f_val / (f_treino + f_val), stratify=resto["classe"],
                                   random_state=seed)
    saida = df.copy()
    saida.loc[treino.index, "split"] = SPLIT_TREINO
    saida.loc[val.index, "split"] = SPLIT_VALIDACAO
    saida.loc[teste.index, "split"] = SPLIT_TESTE
    return saida
