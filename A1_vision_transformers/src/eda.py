"""Tabelas do relatório: vazamento antes/depois, contagens e distribuição dx × split."""
import polars as pl

from src.data import COLUNA_ALVO, COLUNA_GRUPO, ORDEM_SPLITS, checar_vazamento


def _sem_nulos(df: pl.DataFrame, *colunas: str) -> None:
    """Nulos seriam descartados em silêncio pelo is_in/filter/pivot; falha antes."""
    for c in colunas:
        n = df[c].null_count()
        assert n == 0, f"{n} nulos em '{c}': a tabela descartaria essas linhas em silêncio"


def tabela_vazamento(df: pl.DataFrame, coluna: str, coluna_split: str = "split") -> pl.DataFrame:
    """
    Para cada par "a_vs_b" (na ordem train -> validation -> test):
    * grupos_em_comum: valores de `coluna` presentes em a e em b;
    * linhas_de_B_afetadas: linhas de b cujo valor de `coluna` também está em a;
    * pct_linhas_de_B: essas linhas como percentual de todas as linhas de b.
    Usa os mesmos pares e interseções da checar_vazamento, então a tabela e os asserts
    nunca discordam.
    """
    _sem_nulos(df, coluna, coluna_split)

    linhas = []
    for par, comum in checar_vazamento(df, coluna, coluna_split).items():
        _, b = par.split("_vs_")
        df_b = df.filter(pl.col(coluna_split) == b)
        afetadas = df_b.filter(pl.col(coluna).is_in(list(comum))).height
        linhas.append({
            "par": par,
            "grupos_em_comum": len(comum),
            "linhas_de_B_afetadas": afetadas,
            "pct_linhas_de_B": round(100 * afetadas / df_b.height, 2),
        })

    return pl.DataFrame(
        linhas,
        schema={
            "par": pl.String,
            "grupos_em_comum": pl.Int64,
            "linhas_de_B_afetadas": pl.Int64,
            "pct_linhas_de_B": pl.Float64,
        },
    )


def tabela_contagens(df_bruto: pl.DataFrame, df_dedup: pl.DataFrame) -> pl.DataFrame:
    """Contagens explícitas do pipeline, para o relatório."""
    return pl.DataFrame({
        "metrica": [
            "linhas_bruto",
            "image_id_unicos",
            f"{COLUNA_GRUPO}_unicos",
            "linhas_deduplicado",
            "duplicatas_removidas",
            f"{COLUNA_GRUPO}_unicos_deduplicado",
        ],
        "valor": [
            df_bruto.height,
            df_bruto["image_id"].n_unique(),
            df_bruto[COLUNA_GRUPO].n_unique(),
            df_dedup.height,
            df_bruto.height - df_dedup.height,
            df_dedup[COLUNA_GRUPO].n_unique(),
        ],
    })


def tabela_classes_por_split(df: pl.DataFrame, coluna_split: str = "split") -> pl.DataFrame:
    """
    Tabela dx × split com o número de imagens e de lesões, e o percentual de cada
    classe dentro do split (soma 100 por coluna).
    """
    _sem_nulos(df, COLUNA_ALVO, COLUNA_GRUPO, coluna_split)

    presentes = set(df[coluna_split].unique().to_list())
    splits = [s for s in ORDEM_SPLITS if s in presentes]
    colunas = [f"{valor}_{s}" for valor in ("n_imagens", "n_lesoes") for s in splits]

    return (
        df.group_by([COLUNA_ALVO, coluna_split])
          .agg(
              pl.len().alias("n_imagens"),
              pl.col(COLUNA_GRUPO).n_unique().alias("n_lesoes"),
          )
          .pivot(on=coluna_split, index=COLUNA_ALVO, values=["n_imagens", "n_lesoes"])
          # Ordem fixa das colunas; classe ausente de um split vira 0 em vez de null
          .select(COLUNA_ALVO, *[pl.col(c).fill_null(0) for c in colunas])
          .with_columns(
              (pl.col(c) / pl.col(c).sum() * 100).round(2).alias(f"pct_{c}")
              for c in colunas
          )
          .sort(COLUNA_ALVO)
    )
