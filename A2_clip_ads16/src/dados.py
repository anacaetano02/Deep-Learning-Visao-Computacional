"""Corpus de anúncios do ADS-16: verificação do download, listagem e carregamento das imagens."""
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

# Estrutura do ADS-16 no Kaggle (groffo/ads16-dataset, versão 1), verificada pela impressão digital.
N_ARQUIVOS_DATASET = 3429
ASSINATURA_DATASET = "e9071258c790f1e9"

# Corpus = só os anúncios: <parte>/<parte>/Ads/Ads/<categoria 1..20>/<n>.png.
# Corpus/ (fotos pessoais dos 120 participantes) e Documents/ ficam de fora: não são publicidade.
PADRAO_ANUNCIOS = "*/*/Ads/Ads/*/*"
N_CATEGORIAS = 20
N_ARQUIVOS_ADS = 301  # 20 × 15 + 1 arquivo extra na categoria 1 (1/16.png)

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
        h.update(f"{p.relative_to(raiz)}|{tamanho}\n".encode())
    return len(arquivos), round(total / 1e6, 1), h.hexdigest()[:16]


def arvore(raiz: Path, nivel_max: int = 3, nivel: int = 0):
    """Imprime as pastas até `nivel_max` níveis, com o nº de arquivos de cada uma."""
    for p in sorted(raiz.iterdir()):
        if p.is_dir():
            n = sum(1 for f in p.rglob("*") if f.is_file())
            print("    " * nivel + f"{p.name}/  ({n} arquivos)")
            if nivel + 1 < nivel_max:
                arvore(p, nivel_max, nivel + 1)


def _pode_ter_alfa(img: Image.Image) -> bool:
    return img.mode in _MODOS_COM_ALFA or (img.mode == "P" and "transparency" in img.info)


def tem_transparencia(img: Image.Image) -> bool:
    """True se a imagem tem algum pixel não totalmente opaco (alfa < 255)."""
    if not _pode_ter_alfa(img):
        return False
    return img.convert("RGBA").getchannel("A").getextrema()[0] < 255


def carregar_imagem(caminho) -> Image.Image:
    """Abre um anúncio como RGB. Usada em todo o notebook (EDA, embeddings, top-5).

    convert("RGB") direto descarta o alfa: o pixel transparente fica com a cor que estava "por baixo"
    (muitas vezes preto), e o CLIP veria um fundo que não existe no anúncio. Por isso RGBA/P/LA são
    compostos sobre fundo branco antes da conversão. Numa imagem com alfa 255 em todos os pixels (caso
    dos 9 anúncios RGBA/P do ADS-16), a composição não muda nada: o tratamento é preventivo.
    """
    with Image.open(caminho) as img:
        img.load()
        if _pode_ter_alfa(img):
            rgba = img.convert("RGBA")
            fundo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(fundo, rgba).convert("RGB")
        return img.convert("RGB")


def listar_anuncios(raiz: Path) -> pd.DataFrame:
    """Um registro por arquivo de anúncio: categoria, tamanho, modo de cor, transparência e md5."""
    registros = []
    for p in raiz.glob(PADRAO_ANUNCIOS):
        if not p.is_file():
            continue
        with Image.open(p) as img:
            largura, altura = img.size
            modo = img.mode
            transparente = tem_transparencia(img)
        registros.append({"caminho_rel": p.relative_to(raiz).as_posix(), "categoria": int(p.parent.name),
                          "n": int(p.stem), "largura": largura, "altura": altura, "modo": modo,
                          "transparente": transparente, "md5": hashlib.md5(p.read_bytes()).hexdigest()})
    df = pd.DataFrame(registros).sort_values(["categoria", "n"]).reset_index(drop=True)
    df["proporcao"] = df["largura"] / df["altura"]
    return df
