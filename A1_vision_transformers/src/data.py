"""Metadados e split por lesão do HAM10000."""
from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split
import torch

# Nome da coluna-alvo definido uma única vez: o split é estratificado por ela
# e não há parâmetro para trocá-la por engano (ex.: "dx_type").
COLUNA_ALVO = "dx"
COLUNA_GRUPO = "lesion_id"
N_IMAGENS_UNICAS = 10_015
ORDEM_SPLITS = ["train", "validation", "test"]

COLUNAS_SPLIT = [
    "image_id", COLUNA_GRUPO, COLUNA_ALVO,
    "split", "split_original", "idx_original",
]


# Ordem fixa das classes — define o mapeamento string -> índice inteiro.
# Fixar isso explicitamente (em vez de deixar polars/sklearn inferir a
# ordem sozinho) garante que o mapeamento não mude entre execuções.
CLASSES = ["basal_cell_carcinoma", "actinic_keratoses", "benign_keratosis-like_lesions", "melanoma",
           "dermatofibroma", "vascular_lesions", "melanocytic_Nevi"]
CLASSE_PARA_INDICE = {c: i for i, c in enumerate(CLASSES)}
INDICE_PARA_CLASSE = {i: c for c, i in CLASSE_PARA_INDICE.items()}


def extrair_metadados(dataset):
    """
    Devolve um único DataFrame com as colunas de metadado, mais duas colunas novas:
    * split_original (train/validation/test): serve para reproduzir as tabelas de vazamento do "antes" no relatório;
    * idx_original, a posição da linha dentro do split do Hugging Face.
    """

    def split_para_polars(dataset, split: str, colunas_remover=("image",)) -> pl.DataFrame:
        """Converte um split do dataset (Hugging Face) em DataFrame Polars, sem as colunas indicadas."""
        dados = dataset[split].remove_columns(list(colunas_remover))
        return pl.from_pandas(dados.to_pandas())

    def carregar_splits(dataset, splits=("train", "validation", "test")) -> dict[str, pl.DataFrame]:
        """Retorna um dicionário {nome_do_split: DataFrame Polars}."""
        return {split: split_para_polars(dataset, split) for split in splits}

    def concatenar_splits(dfs: dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Junta os splits num único DataFrame, preservando o índice original e a origem de cada linha."""
        partes = [
            df.with_row_index("idx_original").with_columns(pl.lit(nome).alias("split_original"))
            for nome, df in dfs.items()
        ]
        return pl.concat(partes, how="vertical_relaxed")

    dfs = carregar_splits(dataset)
    return concatenar_splits(dfs)


def deduplicar_por_imagem(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove duplicatas por image_id, mantendo a primeira ocorrência (do split "train").
    Antes, garante que as cópias de uma mesma imagem concordam em dx e lesion_id,
    para que o keep="first" não escolha entre valores divergentes em silêncio.
    """
    divergentes = (
        df.group_by("image_id")
          .agg(
              pl.col(COLUNA_ALVO).n_unique().alias("n_alvo"),
              pl.col(COLUNA_GRUPO).n_unique().alias("n_grupo"),
          )
          .filter((pl.col("n_alvo") > 1) | (pl.col("n_grupo") > 1))
    )
    assert divergentes.height == 0, (
        f"{divergentes.height} image_id com {COLUNA_ALVO} ou {COLUNA_GRUPO} divergentes, "
        f"ex.: {divergentes['image_id'].head(5).to_list()}"
    )

    df = df.unique(subset="image_id", keep="first", maintain_order=True)
    assert df.height == N_IMAGENS_UNICAS, (
        f"Esperava {N_IMAGENS_UNICAS} imagens únicas, sobraram {df.height}"
    )
    return df


def montar_split_por_lesao(df: pl.DataFrame, proporcoes: dict[str, float], seed: int) -> pl.DataFrame:
    """
    Divide o dataset por lesão: todas as imagens de uma mesma lesão ficam no mesmo split,
    e a proporção de cada classe (COLUNA_ALVO) é mantida em todos os splits.
    proporcoes: dict ordenado, ex. {"test": 0.15, "validation": 0.15, "train": 0.70}
    Retorna o DataFrame de entrada com a coluna "split".
    """
    assert abs(sum(proporcoes.values()) - 1) < 1e-9, "As proporções devem somar 1"

    # Cada lesão precisa ter uma única classe, senão a estratificação por lesão não faz sentido
    inconsistentes = (
        df.group_by(COLUNA_GRUPO)
          .agg(pl.col(COLUNA_ALVO).n_unique().alias("n_classes"))
          .filter(pl.col("n_classes") > 1)
    )
    assert inconsistentes.height == 0, f"{inconsistentes.height} lesões com mais de uma classe"

    # Uma linha por lesão
    lesoes = (
        df.unique(subset=COLUNA_GRUPO, keep="first", maintain_order=True)
          .select(COLUNA_GRUPO, COLUNA_ALVO)
    )
    ids = lesoes[COLUNA_GRUPO].to_list()
    estratos = lesoes[COLUNA_ALVO].to_list()

    # Vai "separando" um split de cada vez do que sobrou; o último fica com o resto
    nomes = list(proporcoes)
    restante = 1.0
    atribuicao = {}
    for nome in nomes[:-1]:
        fracao = proporcoes[nome] / restante  # ex.: 0.15 / 0.85 = 0.176... do que sobrou
        ids_split, ids, _, estratos = train_test_split(
            ids, estratos, train_size=fracao, stratify=estratos, random_state=seed
        )
        atribuicao[nome] = ids_split
        restante -= proporcoes[nome]
    atribuicao[nomes[-1]] = ids

    mapa = pl.DataFrame({
        COLUNA_GRUPO: pl.Series(
            [i for nome in nomes for i in atribuicao[nome]], dtype=df.schema[COLUNA_GRUPO]
        ),
        "split": [nome for nome in nomes for _ in atribuicao[nome]],
    })

    n_antes = df.height
    df_com_split = df.join(mapa, on=COLUNA_GRUPO, how="left")

    # O n_unique conta null como valor, então nulos precisam de checagem própria
    n_nulos = df_com_split["split"].null_count()
    assert n_nulos == 0, f"{n_nulos} imagens ficaram sem split após o join"
    assert df_com_split.height == n_antes, (
        f"O join mudou o número de linhas: {n_antes} -> {df_com_split.height}"
    )

    # Nenhuma lesão aparece em mais de um split
    lesoes_multiplas = (
        df_com_split.group_by(COLUNA_GRUPO)
                    .agg(pl.col("split").n_unique().alias("n_splits"))
                    .filter(pl.col("n_splits") > 1)
    )
    assert lesoes_multiplas.height == 0, f"{lesoes_multiplas.height} lesões em mais de um split"

    return df_com_split


def checar_vazamento(df: pl.DataFrame, coluna: str, coluna_split: str = "split") -> dict[str, set]:
    """
    Para cada par de splits, retorna os valores de `coluna` que aparecem nos dois.
    coluna_split="split_original" gera o "antes"; o padrão "split" gera o "depois".
    """
    presentes = set(df[coluna_split].unique().to_list())
    faltando_na_ordem = presentes - set(ORDEM_SPLITS)
    assert not faltando_na_ordem, f"Splits fora de ORDEM_SPLITS: {faltando_na_ordem}"

    # Ordem fixa (train -> validation -> test): em cada par "a_vs_b", b é o split avaliado
    valores_por_split = {
        split: set(df.filter(pl.col(coluna_split) == split)[coluna])
        for split in ORDEM_SPLITS if split in presentes
    }

    splits = list(valores_por_split)
    vazamentos = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            vazamentos[f"{a}_vs_{b}"] = valores_por_split[a] & valores_por_split[b]

    return vazamentos


def validar_split(df: pl.DataFrame, colunas=("image_id", COLUNA_GRUPO)) -> None:
    """Interrompe a execução se houver imagem sem split ou id em mais de um split."""
    # O filter por split descarta nulos; sem esta checagem eles sumiriam em silêncio
    n_nulos = df["split"].null_count()
    assert n_nulos == 0, f"{n_nulos} linhas sem split"

    n_splits = df["split"].n_unique()
    assert n_splits == 3, f"Esperava 3 splits, encontrei {n_splits}"

    for coluna in colunas:
        vazamentos = checar_vazamento(df, coluna)
        assert len(vazamentos) == 3, f"Esperava 3 pares para {coluna}, encontrei {len(vazamentos)}"

        for par, intersecao in vazamentos.items():
            assert len(intersecao) == 0, (
                f"Vazamento de {coluna} em {par}: {len(intersecao)} ids, "
                f"ex.: {sorted(intersecao)[:5]}"
            )

    print(f"Split OK: zero vazamento de {', '.join(colunas)} entre os pares de splits.")


def salvar_split(df: pl.DataFrame, caminho: str | Path) -> None:
    """Salva só as colunas necessárias para reconstruir o split, em CSV versionável."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.select(COLUNAS_SPLIT).sort("image_id").write_csv(caminho)
    print(f"Split salvo em {caminho} ({df.height} linhas)")


def carregar_split(caminho: str | Path) -> pl.DataFrame:
    """Lê o split salvo por `salvar_split`, com os tipos corretos, e valida as colunas."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Split não encontrado em {caminho}")

    df = pl.read_csv(
        caminho,
        schema_overrides={
            "image_id": pl.String,
            COLUNA_GRUPO: pl.String,
            COLUNA_ALVO: pl.String,
            "split": pl.String,
            "split_original": pl.String,
            "idx_original": pl.UInt32,
        },
    )

    faltando = set(COLUNAS_SPLIT) - set(df.columns)
    assert not faltando, f"Colunas ausentes no CSV: {faltando}"

    print(f"Split carregado de {caminho} ({df.height} linhas)")
    return df.select(COLUNAS_SPLIT)


def computar_pesos(df: pl.DataFrame, split: str = "train") -> torch.Tensor:
    """
    Pesos de classe "balanced" para a CrossEntropyLoss: n_total / (n_classes * n_classe).
    Recebe o DataFrame completo (com a coluna "split") e usa só as linhas de `split`,
    para a distribuição de validação/teste não entrar na loss. Conta imagens, não
    lesões, porque a loss é calculada por imagem. O tensor segue a ordem de CLASSES:
    peso[i] é o peso da classe de índice i.
    """
    df_treino = df.filter(pl.col("split") == split)
    assert df_treino.height > 0, f"Nenhuma linha com split == {split!r}"

    presentes = set(df_treino[COLUNA_ALVO].unique().to_list())
    assert presentes == set(CLASSES), (
        f"Classes do {split} diferentes de CLASSES: "
        f"faltando {set(CLASSES) - presentes}, sobrando {presentes - set(CLASSES)}"
    )

    contagens = df_treino.group_by(COLUNA_ALVO).agg(pl.len().alias("n"))
    contagem_por_classe = dict(zip(contagens[COLUNA_ALVO], contagens["n"]))

    total = df_treino.height
    num_classes = len(CLASSES)

    # Acesso direto (sem .get): uma classe ausente daria KeyError em vez de peso ~1000
    pesos = torch.tensor(
        [total / (num_classes * contagem_por_classe[classe]) for classe in CLASSES],
        dtype=torch.float32,
    )

    # Invariante da fórmula "balanced": cada classe contribui igualmente (peso * n = total / k)
    for classe, peso in zip(CLASSES, pesos.tolist()):
        assert abs(peso * contagem_por_classe[classe] - total / num_classes) < 1e-3, classe

    return pesos