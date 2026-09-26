import polars as pl
import duckdb
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

def analyze_skin_cancer_data(dataset):
    """
    Realiza uma análise abrangente do conjunto de dados de câncer de pele, incluindo
    extração de metadados, análise de distribuição entre divisões (treino, validação, teste),
    índices de desequilíbrio, verificação de vazamento e estatísticas demográficas básicas.

    Args:
        dataset: Um objeto `DatasetDict` carregado do Hugging Face contendo as divisões 'train',
                 'validation' e 'test', cada uma com imagem e metadados.

    Returns:
        tuple: Uma tupla contendo os DataFrames Polars processados:
               (df_train, df_validation, df_test, df_comparativo)
    """

    print("--- Extraindo Metadados e Criando DataFrames Polars ---")
    train_data = dataset['train']
    validation_data = dataset['validation']
    test_data = dataset['test']

    # extrai só as colunas de metadado, sem carregar a imagem em si
    train_meta = train_data.remove_columns("image").to_pandas()
    df_train = pl.from_pandas(train_meta)

    validation_meta = validation_data.remove_columns("image").to_pandas()
    df_validation = pl.from_pandas(validation_meta)

    test_meta = test_data.remove_columns("image").to_pandas()
    df_test = pl.from_pandas(test_meta)

    print(f"df_train shape: {df_train.shape}")
    display(df_train.head())
    print(f"df_validation shape: {df_validation.shape}")
    display(df_validation.head())
    print(f"df_test shape: {df_test.shape}")
    display(df_test.head())

    print("\n--- Análise da Distribuição de Classes para o Conjunto de Treino ---")
    distribuicao = duckdb.sql("""
      select
        dx,
        count(*) as contagem,
        round(count(*) * 100.0 / sum(count(*)) over (), 2) as percentual
      from df_train
      group by dx
      order by contagem desc
    """).pl()
    display(distribuicao)

    razao_desbalanceamento = duckdb.sql("""
      select max(contagem) / min(contagem) as razao from distribuicao
    """).pl()
    print("Taxa de Desequilíbrio do Treino:")
    display(razao_desbalanceamento)

    distribuicao_com_razao = duckdb.sql("""
      select
        t1.*,
        round(max(t1.contagem) over () / t1.contagem, 1) as razao_vs_majoritaria
      from distribuicao t1
      group by dx, contagem, percentual
      order by contagem desc
    """).pl()
    print("Distribuição do Treino com Taxa de Desequilíbrio (vs. classe majoritária):")
    display(distribuicao_com_razao)

    print("\n--- Análise da Distribuição de Classes para o Conjunto de Validação ---")
    distribuicao_validacao = duckdb.sql("""
      select
        dx,
        count(*) as contagem,
        round(count(*) * 100.0 / sum(count(*)) over (), 2) as percentual
      from df_validation
      group by dx
      order by contagem desc
    """).pl()
    display(distribuicao_validacao)

    razao_desbalanceamento_val = duckdb.sql("""
      select max(contagem) / min(contagem) as razao from distribuicao_validacao
    """).pl()
    print("Taxa de Desequilíbrio da Validação:")
    display(razao_desbalanceamento_val)

    distribuicao_com_razao_val = duckdb.sql("""
      select
        t1.*,
        round(max(t1.contagem) over () / t1.contagem, 1) as razao_vs_majoritaria
      from distribuicao_validacao t1
      group by dx, contagem, percentual
      order by contagem desc
    """).pl()
    print("Distribuição da Validação com Taxa de Desequilíbrio (vs. classe majoritária):")
    display(distribuicao_com_razao_val)

    print("\n--- Análise da Distribuição de Classes para o Conjunto de Teste ---")
    distribuicao_test = duckdb.sql("""
      select
        dx,
        count(*) as contagem,
        round(count(*) * 100.0 / sum(count(*)) over (), 2) as percentual
      from df_test
      group by dx
      order by contagem desc
    """).pl()
    display(distribuicao_test)

    razao_desbalanceamento_test = duckdb.sql("""
      select max(contagem) / min(contagem) as razao from distribuicao_test
    """).pl()
    print("Taxa de Desequilíbrio do Teste:")
    display(razao_desbalanceamento_test)

    distribuicao_com_razao_test = duckdb.sql("""
      select
        t1.*,
        round(max(t1.contagem) over () / t1.contagem, 1) as razao_vs_majoritaria
      from distribuicao_test t1
      group by dx, contagem, percentual
      order by contagem desc
    """).pl()
    print("Distribuição do Teste com Taxa de Desequilíbrio (vs. classe majoritária):")
    display(distribuicao_com_razao_test)

    print("\n--- Distribuição Comparativa Entre as Divisões ---")
    df_comparativo = duckdb.sql("""
      select
        t1.dx,
        t1.contagem as contagem_treino,
        t2.contagem as contagem_validacao,
        t3.contagem as contagem_teste,
        t1.percentual as percentual_treino,
        t2.percentual as percentual_validacao,
        t3.percentual as percentual_teste,
        t1.razao_vs_majoritaria as razao_vs_majoritaria_treino,
        t2.razao_vs_majoritaria as razao_vs_majoritaria_validacao,
        t3.razao_vs_majoritaria as razao_vs_majoritaria_test
      from distribuicao_com_razao t1
      join distribuicao_com_razao_val t2 on t1.dx = t2.dx
      join distribuicao_com_razao_test t3 on t1.dx = t3.dx
    """).pl()
    print(f"df_comparativo shape: {df_comparativo.shape}")
    display(df_comparativo.head())

    # Plot comparative distribution
    print("\n--- Plotando a Distribuição Comparativa de Classes ---")
    x = np.arange(len(df_comparativo))
    largura = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - largura, df_comparativo["percentual_treino"], largura, label="Treino")
    ax.bar(x, df_comparativo["percentual_validacao"], largura, label="Validação")
    ax.bar(x + largura, df_comparativo["percentual_teste"], largura, label="Teste")

    ax.set_xticks(x)
    ax.set_xticklabels(df_comparativo["dx"], rotation=45, ha="right")
    ax.set_ylabel("Percentual")
    ax.set_title("Comparação da Distribuição de Classes entre Conjuntos de Dados")
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Análise de Vazamento (ID da Lesão) ---")
    # Assumindo que df_train, df_validation, df_test estão disponíveis das etapas anteriores
    df_lesion = duckdb.sql("""
      select distinct
        t.lesion_id,
        case
          when t2.lesion_id is not null then 1
          else 0
        end as is_validation,
        case
          when t3.lesion_id is not null then 1
          else 0
        end as is_test
      from df_train t
      left join df_validation t2 on t.lesion_id = t2.lesion_id
      left join df_test t3 on t.lesion_id = t3.lesion_id
      where t2.lesion_id is not null or t3.lesion_id is not null
    """).pl()
    print(f"df_lesion shape: {df_lesion.shape}")
    display(df_lesion.head())

    contagem_lesion = duckdb.sql("""
      select
        sum(is_validation) as qtd_validation_overlap,
        sum(is_test) as qtd_test_overlap
      from df_lesion
    """).pl()
    print("Sobreposição de lesion_id entre treino e validação/teste:")
    display(contagem_lesion)

    unique_lesions_in_train = df_train["lesion_id"].n_unique()
    total_overlapping_lesions = df_lesion.height
    percentual_vazamento = (total_overlapping_lesions / unique_lesions_in_train) * 100
    print(f"\nTotal de lesion_ids únicos no treino: {unique_lesions_in_train}")
    print(f"Número de lesion_ids no treino que aparecem na validação ou teste: {total_overlapping_lesions}")
    print(f"Porcentagem de vazamento: {percentual_vazamento:.2f}%")


    print("\n--- Estatísticas de Idade por Classe ---")
    age_stats = df_train.group_by("dx").agg(
        pl.col("age").mean().round(1).alias("idade_media"),
        pl.col("age").median().alias("idade_mediana"),
        pl.col("age").null_count().alias("faltantes"),
        pl.col("age").min().alias("idade_min"),
        pl.col("age").max().alias("idade_max"),
    ).sort("idade_media", descending=True)
    display(age_stats)

    print("\n--- Distribuição de Gênero por Classe ---")
    gender_dist = df_train.group_by(["dx", "sex"]).agg(
        pl.len().alias("contagem")
    ).with_columns(
        (pl.col("contagem") / pl.col("contagem").sum().over("dx") * 100).round(1).alias("percentual_na_classe")
    ).sort(["dx", "sex"])
    display(gender_dist)

    print("\n--- Distribuição de Localização por Classe (Top 3) ---")
    localization_dist = df_train.group_by(["dx", "localization"]).agg(
        pl.len().alias("contagem")
    ).with_columns(
        pl.col("contagem").rank(method="ordinal", descending=True).over("dx").alias("posicao")
    ).filter(pl.col("posicao") <= 3).sort(["dx", "posicao"])
    display(localization_dist)

    print("\n--- Amostras com Idade 0 ---")
    age_zero_count = df_train.filter(pl.col("age") == 0).height
    print(f"Número de amostras nos dados de treino com idade 0: {age_zero_count}")

    return df_train, df_validation, df_test, df_comparativo