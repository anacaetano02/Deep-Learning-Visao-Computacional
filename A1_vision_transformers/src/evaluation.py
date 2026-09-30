"""Avaliação dos modelos da A1 (R2, R3, R22).

Bloco 1 (mínimo para R2/R3): prever, calcular_metricas, plotar_matriz_confusao, plotar_curvas.
Bloco 2 (comparação e robustez): ids_do_split, baseline_dummy, bootstrap_f1, bootstrap_pareado,
    tabela_comparativa, descrever_diferenca, tabela_por_classe.
Bloco 3 (exemplos): exemplos_de_erro.

Reaproveita o que já existe no projeto:
- `extrair_logits` (training.py) trata a tupla do ViT do zero e o `.logits` do HF;
- `CLASSES` / `CLASSE_PARA_INDICE` (data.py) dão os nomes e o mapa classe → índice do Dataset;
- os loaders de validação/teste têm shuffle=False, então a ordem das predições é a ordem das
  linhas de `SPLIT_OFICIAL.filter(split == "test")` (conferida em `ids_do_split`);
- os batches e os itens do Dataset são tuplas (imagem, rótulo).

As chaves `relatorio_por_classe` e `matriz_confusao` devolvidas por `calcular_metricas` são as que
o `registrar_experimento` já grava num JSON à parte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import SequentialSampler

from .data import CLASSE_PARA_INDICE, CLASSES
from .training import extrair_logits

N_CLASSES = len(CLASSES)
LABELS = list(range(N_CLASSES))  # sempre explícito: evita matriz 6×6 se uma classe sumir

# Nomes completos do HF → siglas usuais (cabem nos eixos dos gráficos)
SIGLAS = {
    "actinic_keratoses": "akiec",
    "basal_cell_carcinoma": "bcc",
    "benign_keratosis-like_lesions": "bkl",
    "dermatofibroma": "df",
    "melanoma": "mel",
    "melanocytic_Nevi": "nv",
    "vascular_lesions": "vasc",
}


def sigla(classe: str) -> str:
    """Sigla curta da classe (devolve o próprio nome se não houver sigla conhecida)."""
    return SIGLAS.get(classe, classe)


def indice_da_classe(classe: int | str, classes: Sequence[str] = CLASSES) -> int:
    """Aceita índice, nome completo ou sigla ('mel') e devolve o índice da classe."""
    if isinstance(classe, (int, np.integer)):
        return int(classe)
    for i, c in enumerate(classes):
        if classe in (c, sigla(c)):
            return i
    raise ValueError(f"Classe desconhecida: {classe!r}.")


def _imagens_e_rotulos(item):
    """Separa imagem e rótulo de um batch ou item do Dataset (tupla/lista (imagem, rótulo))."""
    return item[0], item[1]


def ler_marca(dir_checkpoints, nome: str) -> dict:
    """Lê o concluido.json do experimento: a mesma fonte do melhor.pt (melhor_f1, epoca_melhor)."""
    caminho = Path(dir_checkpoints) / nome / "concluido.json"
    return json.loads(caminho.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# Bloco 1: o mínimo para o R2 e o R3
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def prever(modelo, loader, device, usar_amp: bool = True):
    """Roda o modelo no loader e devolve (y_true, y_pred, probs) em numpy.

    - `probs` tem forma (N, n_classes), já com softmax (em fp32), útil depois para ROC/PR (B5).
    - A ordem é a do loader; por isso o loader precisa ter shuffle=False (verificado).
    - O modo do modelo (train/eval) é restaurado no fim.
    """
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("O loader de avaliação precisa de shuffle=False: a ordem das predições "
                         "é usada para recuperar lesion_id/image_id.")

    tipo_device = torch.device(device).type
    estava_treinando = modelo.training
    modelo.eval()

    y_true, probs = [], []
    try:
        for batch in loader:
            x, y = _imagens_e_rotulos(batch)
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=tipo_device, dtype=torch.float16,
                                enabled=usar_amp and tipo_device == "cuda"):
                logits = extrair_logits(modelo(x))
            # softmax em fp32 e .cpu() a cada batch: nada se acumula na GPU
            probs.append(torch.softmax(logits.float(), dim=-1).cpu())
            y_true.append(torch.as_tensor(y).cpu())
    finally:
        modelo.train(estava_treinando)

    probs = torch.cat(probs).numpy()
    y_true = torch.cat(y_true).numpy().astype(np.int64)
    y_pred = probs.argmax(axis=1)
    if probs.shape[1] != N_CLASSES:
        raise ValueError(f"O modelo devolveu {probs.shape[1]} classes; esperado {N_CLASSES}.")
    return y_true, y_pred, probs


def calcular_metricas(y_true, y_pred, classes: Sequence[str] = CLASSES) -> dict:
    """F1 macro (principal), F1 weighted, acurácia (referência), relatório e matriz de confusão.

    `relatorio_por_classe` e `matriz_confusao` são os nomes que o registrar_experimento grava
    num JSON à parte. A matriz vai como lista (serializável), sem normalizar.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted",
                                      zero_division=0)),
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "relatorio_por_classe": classification_report(
            y_true, y_pred, labels=LABELS, target_names=list(classes),
            output_dict=True, zero_division=0,
        ),
        "matriz_confusao": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def plotar_matriz_confusao(matriz, classes: Sequence[str] = CLASSES, normalizar: str | None = "true",
                           titulo: str = "Matriz de confusão", ax=None):
    """Plota a matriz de confusão com siglas nos eixos e o suporte (n) de cada classe real.

    normalizar="true" divide cada linha pelo total da classe real: a diagonal vira o recall
    (a leitura que interessa para `mel`). None mostra as contagens.
    Para duas matrizes lado a lado: `fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))` e
    `ax=axes[i]` em cada chamada.
    """
    m = np.asarray(matriz, dtype=float)
    suporte = m.sum(axis=1).astype(int)  # antes de normalizar
    if normalizar == "true":
        m = np.divide(m, suporte[:, None], out=np.zeros_like(m), where=suporte[:, None] > 0)
        formato = ".2f"
    elif normalizar is None:
        formato = ".0f"
    else:
        raise ValueError('normalizar deve ser "true" ou None.')

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure
    siglas = [sigla(c) for c in classes]
    disp = ConfusionMatrixDisplay(m, display_labels=siglas)
    disp.plot(ax=ax, cmap="Blues", values_format=formato, colorbar=False)
    if normalizar == "true":
        disp.im_.set_clim(0, 1)
    ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(siglas, suporte)])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    fig.tight_layout()
    return fig


def plotar_curvas(historico, titulo: str, marca: Mapping | None = None,
                  chave_loss_treino: str = "train_loss", chave_loss_val: str = "val_loss",
                  chave_f1: str = "val_f1_macro"):
    """Dois painéis: loss de treino e validação | F1 macro de validação, com a época escolhida.

    `historico` é o dict devolvido pelo treinar_modelo (ou um pl.DataFrame com as mesmas colunas).
    `marca` é o concluido.json do experimento (`ler_marca`): a época escolhida vem dele, a mesma
    fonte do melhor.pt, em vez de ser digitada. Se as chaves do seu histórico tiverem outros
    nomes, passe-os nos parâmetros `chave_*`.
    """
    h = historico.to_dict(as_series=False) if isinstance(historico, pl.DataFrame) else historico
    faltando = [c for c in (chave_loss_treino, chave_loss_val, chave_f1) if c not in h]
    if faltando:
        raise KeyError(f"Chaves ausentes no histórico: {faltando}; disponíveis: {list(h)}")

    epocas = h.get("epoca", list(range(1, len(h[chave_f1]) + 1)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax1.plot(epocas, h[chave_loss_treino], label="treino (augmentation e dropout ligados)")
    ax1.plot(epocas, h[chave_loss_val], label="validação")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss (CE ponderada)")
    ax1.set_title("Loss")

    ax2.plot(epocas, h[chave_f1], color="tab:green")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("F1 macro")
    ax2.set_title("F1 macro de validação")

    if marca is not None:
        epoca_melhor = int(marca["epoca_melhor"])
        rotulo = f"escolhida: época {epoca_melhor} (F1 {marca['melhor_f1']:.4f})"
        for ax in (ax1, ax2):
            ax.axvline(epoca_melhor, color="gray", linestyle="--", linewidth=1, label=rotulo)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:  # só onde há artistas com label
            ax.legend(fontsize=9)
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# Bloco 2: comparação (R22) e robustez
# ---------------------------------------------------------------------------------------------

def ids_do_split(split_oficial: pl.DataFrame, y_true, split: str = "test",
                 coluna_classe: str = "dx"):
    """lesion_id e image_id na ordem das predições (a do loader com shuffle=False).

    Confere a ordem: a classe de cada linha do split, convertida com o mesmo mapa do Dataset
    (`CLASSE_PARA_INDICE`), tem de ser igual ao `y_true` devolvido por `prever`.
    """
    y_true = np.asarray(y_true)
    df = split_oficial.filter(pl.col("split") == split)
    assert df.height == len(y_true), (
        f"O split '{split}' tem {df.height} linhas, mas há {len(y_true)} predições.")
    esperado = np.array([CLASSE_PARA_INDICE[c] for c in df[coluna_classe]])
    assert np.array_equal(esperado, y_true), (
        f"Metadados do split '{split}' fora da ordem das predições: lesion_id/image_id não "
        f"correspondem às imagens ({int((esperado != y_true).sum())} rótulos diferentes).")
    return df["lesion_id"].to_numpy(), df["image_id"].to_numpy()


def baseline_dummy(y_train, y_test, seed: int = 42,
                   estrategias: Sequence[str] = ("most_frequent", "stratified")) -> dict:
    """Baselines que só olham os rótulos do treino: {estratégia: {"y_pred", "metricas"}}.

    Mostram a partir de onde um F1 macro começa a significar alguma coisa.
    """
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    saida = {}
    for estrategia in estrategias:
        dummy = DummyClassifier(strategy=estrategia, random_state=seed)
        dummy.fit(np.zeros((len(y_train), 1)), y_train)
        y_pred = dummy.predict(np.zeros((len(y_test), 1)))
        saida[estrategia] = {"y_pred": y_pred, "metricas": calcular_metricas(y_test, y_pred)}
    return saida


class ICBootstrap(NamedTuple):
    """Resultado do bootstrap. Os índices 0–2 (média, IC 2,5%, IC 97,5%) são os de antes."""
    media: float
    ic_inf: float
    ic_sup: float
    replicas_sem_classe: int  # réplicas sem nenhuma lesão de alguma classe
    amostras: np.ndarray


def _replicas_por_lesao(y_true, grupos, n: int, seed: int):
    """Gera, para cada réplica, (índices das imagens, faltou_classe).

    Sorteia com reposição tantas lesões quantas existem e inclui todas as imagens de cada lesão
    sorteada. Os sorteios dependem só de `grupos`, `n` e `seed`: com os mesmos valores, qualquer
    modelo é avaliado nas mesmas réplicas.
    """
    _, codigos = np.unique(grupos, return_inverse=True)
    ordem = np.argsort(codigos, kind="stable")
    inicios = np.searchsorted(codigos[ordem], np.arange(codigos.max() + 1))
    imagens_por_lesao = np.split(ordem, inicios[1:])
    classe_da_lesao = np.array([y_true[ims[0]] for ims in imagens_por_lesao])
    n_lesoes = len(imagens_por_lesao)

    rng = np.random.default_rng(seed)
    for _ in range(n):
        sorteadas = rng.integers(0, n_lesoes, size=n_lesoes)
        faltou = np.unique(classe_da_lesao[sorteadas]).size < N_CLASSES
        yield np.concatenate([imagens_por_lesao[j] for j in sorteadas]), faltou


def _resumir(amostras: np.ndarray, sem_classe: int) -> ICBootstrap:
    inf, sup = (float(v) for v in np.percentile(amostras, [2.5, 97.5]))
    return ICBootstrap(float(amostras.mean()), inf, sup, int(sem_classe), amostras)


def _checar_entradas(y_true, grupos, *y_preds):
    y_true, grupos = np.asarray(y_true), np.asarray(grupos)
    y_preds = [np.asarray(p) for p in y_preds]
    if any(len(v) != len(y_true) for v in (grupos, *y_preds)):
        raise ValueError("y_true, y_pred e grupos precisam ter o mesmo tamanho (mesma ordem).")
    return y_true, grupos, y_preds


def bootstrap_f1(y_true, y_pred, grupos, n: int = 1000, seed: int = 42) -> ICBootstrap:
    """Bootstrap do F1 macro reamostrando LESÕES (não imagens).

    Imagens da mesma lesão são correlacionadas; reamostrar imagens subestimaria a incerteza.
    Devolve ICBootstrap(media, ic_inf, ic_sup, replicas_sem_classe, amostras); `ic[1]`/`ic[2]`
    continuam sendo os limites do IC. `replicas_sem_classe` conta as réplicas em que alguma
    classe ficou sem nenhuma lesão (nelas o F1 dessa classe entra como 0 na média macro).
    """
    y_true, grupos, (y_pred,) = _checar_entradas(y_true, grupos, y_pred)
    amostras, sem_classe = np.empty(n), 0
    for b, (idx, faltou) in enumerate(_replicas_por_lesao(y_true, grupos, n, seed)):
        amostras[b] = f1_score(y_true[idx], y_pred[idx], labels=LABELS, average="macro",
                               zero_division=0)
        sem_classe += faltou
    return _resumir(amostras, sem_classe)


def bootstrap_pareado(y_true_a, y_pred_a, y_true_b, y_pred_b, grupos, n: int = 1000,
                      seed: int = 42, nomes: tuple[str, str] = ("zero", "pre")) -> dict:
    """Bootstrap pareado: os dois modelos nas MESMAS réplicas, com a diferença (b − a) por réplica.

    Confere antes que os dois y_true são idênticos (mesmas imagens, mesma ordem). Devolve
    {nome_a: ICBootstrap, nome_b: ICBootstrap, "diferenca": ICBootstrap}. Se o IC da diferença
    não contém 0, a vantagem de b sobre a não é explicada pela amostra de teste.
    """
    y_true_a, y_true_b = np.asarray(y_true_a), np.asarray(y_true_b)
    assert np.array_equal(y_true_a, y_true_b), (
        "Os y_true dos dois modelos diferem: as predições não estão na mesma ordem de imagens.")
    y_true, grupos, (pa, pb) = _checar_entradas(y_true_a, grupos, y_pred_a, y_pred_b)

    f1_a, f1_b, sem_classe = np.empty(n), np.empty(n), 0
    for b, (idx, faltou) in enumerate(_replicas_por_lesao(y_true, grupos, n, seed)):
        f1_a[b] = f1_score(y_true[idx], pa[idx], labels=LABELS, average="macro", zero_division=0)
        f1_b[b] = f1_score(y_true[idx], pb[idx], labels=LABELS, average="macro", zero_division=0)
        sem_classe += faltou
    return {
        nomes[0]: _resumir(f1_a, sem_classe),
        nomes[1]: _resumir(f1_b, sem_classe),
        "diferenca": _resumir(f1_b - f1_a, sem_classe),
    }


def descrever_diferenca(pareado: Mapping, nomes: tuple[str, str] = ("zero", "pre")) -> str:
    """Frase para pôr abaixo da tabela comparativa, a partir de `bootstrap_pareado`."""
    d = pareado["diferenca"]
    conclusao = ("o IC não contém 0" if d.ic_inf > 0 or d.ic_sup < 0 else "o IC contém 0")
    return (f"Diferença pareada de F1 macro ({nomes[1]} − {nomes[0]}): {d.media:+.4f} "
            f"[IC 95%: {d.ic_inf:+.4f}; {d.ic_sup:+.4f}], {conclusao}; "
            f"{len(d.amostras)} réplicas por lesão, {d.replicas_sem_classe} sem alguma classe.")


def _float(v):
    return None if v is None else float(v)


def _int(v):
    return None if v is None else int(v)


_SCHEMA_COMPARATIVA = {
    "modelo": pl.Utf8,
    "f1_macro": pl.Float64,
    "f1_ic_2_5": pl.Float64,
    "f1_ic_97_5": pl.Float64,
    "f1_macro_val": pl.Float64,
    "f1_weighted": pl.Float64,
    "recall_mel": pl.Float64,
    "acuracia": pl.Float64,
    "resolucao": pl.Int64,
    "parametros_treinaveis": pl.Int64,
    "epoca_melhor": pl.Int64,
    "epocas_treinadas": pl.Int64,
    "minutos": pl.Float64,
    "vram_gb": pl.Float64,
}


def tabela_comparativa(resultados: Mapping[str, Mapping], classe_mel: str = "melanoma") -> pl.DataFrame:
    """Uma linha por modelo (inclua os baselines), para a Tabela 5 do relatório.

    Cada valor de `resultados` é um dict com as métricas de teste de `calcular_metricas` e, se
    houver: `ic_f1` (ICBootstrap), `f1_macro_val`, `resolucao` (px), `parametros_treinaveis`,
    `epoca_melhor`, `epocas_treinadas`, `minutos` e `vram_gb`. O que faltar vira nulo (ex.:
    baselines não têm resolução nem tempo). A diferença pareada vai em `descrever_diferenca`.
    """
    linhas = []
    for nome, r in resultados.items():
        ic = r.get("ic_f1")
        relatorio = r.get("relatorio_por_classe", {})
        linhas.append({
            "modelo": str(nome),
            "f1_macro": _float(r.get("f1_macro")),
            "f1_ic_2_5": _float(ic[1]) if ic is not None else None,
            "f1_ic_97_5": _float(ic[2]) if ic is not None else None,
            "f1_macro_val": _float(r.get("f1_macro_val")),
            "f1_weighted": _float(r.get("f1_weighted")),
            "recall_mel": _float(relatorio.get(classe_mel, {}).get("recall")),
            "acuracia": _float(r.get("acuracia")),
            "resolucao": _int(r.get("resolucao")),
            "parametros_treinaveis": _int(r.get("parametros_treinaveis")),
            "epoca_melhor": _int(r.get("epoca_melhor")),
            "epocas_treinadas": _int(r.get("epocas_treinadas")),
            "minutos": _float(r.get("minutos")),
            "vram_gb": _float(r.get("vram_gb")),
        })
    return (pl.DataFrame(linhas, schema=_SCHEMA_COMPARATIVA, strict=True)
            .with_columns(pl.col(pl.Float64).round(4)))


def tabela_por_classe(resultados: Mapping[str, Mapping], classes: Sequence[str] = CLASSES) -> pl.DataFrame:
    """Precision, recall e F1 por classe, uma coluna por métrica × modelo, e o support uma vez.

    `resultados[nome]` precisa ter `relatorio_por_classe` (de `calcular_metricas`). O support é
    conferido entre os modelos: todos têm de ter sido avaliados nas mesmas imagens.
    """
    metricas = (("precision", "precision"), ("recall", "recall"), ("f1-score", "f1"))
    linhas = []
    for i in LABELS:
        classe = classes[i]
        suportes = {int(r["relatorio_por_classe"][classe]["support"]) for r in resultados.values()}
        assert len(suportes) == 1, f"Support de '{classe}' difere entre os modelos: {suportes}."
        linha = {"classe": sigla(classe), "support": suportes.pop()}
        for chave, curto in metricas:
            for nome, r in resultados.items():
                linha[f"{curto}_{nome}"] = float(r["relatorio_por_classe"][classe][chave])
        linhas.append(linha)
    return pl.DataFrame(linhas).with_columns(pl.col(pl.Float64).round(3))


# ---------------------------------------------------------------------------------------------
# Bloco 3: exemplos de erro (R3)
# ---------------------------------------------------------------------------------------------

def exemplos_de_erro(probs, y_true, dataset_exibicao, lesion_ids, k: int = 8,
                     classe_real: int | str | None = None, indices=None, image_ids=None,
                     classes: Sequence[str] = CLASSES, media=(0.5, 0.5, 0.5), desvio=(0.5, 0.5, 0.5),
                     n_colunas: int = 4, titulo: str = "Erros com maior confiança"):
    """Grade com os k erros mais confiantes (maior probabilidade na classe errada), um por lesão.

    - `probs`/`y_true` vêm de `prever`; `dataset_exibicao` só fornece as imagens e pode ser outro
      dataset na mesma ordem — por exemplo, o de 224 px para mostrar os erros do ViT do zero
      (128 px) com mais detalhe. `media`/`desvio` são os do dataset de exibição.
    - `lesion_ids` (de `ids_do_split`): depois de ordenar por confiança, fica só o erro mais
      confiante de cada lesão, para a grade não repetir fotos da mesma lesão.
    - `classe_real` (índice, nome ou sigla, ex.: "mel") restringe aos erros dessa classe.
    - `indices[i]` é o índice no dataset da predição i (padrão: i).
    Devolve (figura, índices das predições mostradas).
    """
    probs, y_true, lesion_ids = np.asarray(probs), np.asarray(y_true), np.asarray(lesion_ids)
    y_pred = probs.argmax(axis=1)
    indices = np.arange(len(y_true)) if indices is None else np.asarray(indices)

    mascara = y_pred != y_true
    if classe_real is not None:
        mascara &= y_true == indice_da_classe(classe_real, classes)
    erros = np.flatnonzero(mascara)
    if len(erros) == 0:
        raise ValueError("Nenhum erro para mostrar com esses filtros.")
    erros = erros[np.argsort(-probs[erros, y_pred[erros]], kind="stable")]

    escolhidos, vistas = [], set()
    for i in erros:  # um por lesão, o mais confiante
        if lesion_ids[i] not in vistas:
            vistas.add(lesion_ids[i])
            escolhidos.append(i)
            if len(escolhidos) == k:
                break
    escolhidos = np.array(escolhidos)

    n_linhas = int(np.ceil(len(escolhidos) / n_colunas))
    fig, eixos = plt.subplots(n_linhas, n_colunas, figsize=(3.2 * n_colunas, 3.6 * n_linhas),
                              squeeze=False)
    m = torch.tensor(media).view(3, 1, 1)
    s = torch.tensor(desvio).view(3, 1, 1)

    for ax, i in zip(eixos.flat, escolhidos):
        x, _ = _imagens_e_rotulos(dataset_exibicao[int(indices[i])])
        img = (torch.as_tensor(x).float().cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        rotulo = (f"real: {sigla(classes[y_true[i]])}\n"
                  f"previsto: {sigla(classes[y_pred[i]])} ({probs[i, y_pred[i]]:.2f})")
        if image_ids is not None:
            rotulo += f"\n{image_ids[i]}"
        ax.set_title(rotulo, fontsize=9)
    for ax in eixos.flat:
        ax.axis("off")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig, escolhidos