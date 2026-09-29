"""
Módulo src/models.py (A1 — Vision Transformers)
ViT pré-treinado do Hugging Face com head de 7 classes e congelamento por
número de blocos finais treináveis.

Importa só de src.data (nunca de src.training), para não depender da ordem
do importlib.reload no notebook.
"""
import re

import torch
from transformers import ViTForImageClassification

from src.data import CLASSE_PARA_INDICE, INDICE_PARA_CLASSE

# ViT-B/16 pré-treinado no ImageNet-21k e depois ajustado no ImageNet-1k, a 224 px (head de
# 1000 classes, trocado aqui). A alternativa "-in21k" (usada na aula 5) para no pré-treino do
# 21k, com head de 21.843 classes. Justificativa da escolha: R4, no notebook.
CHECKPOINT_VIT = "google/vit-base-patch16-224"

_SHA_COMMIT = re.compile(r"[0-9a-f]{40}")


def carregar_vit_pretreinado(
    revisao: str,
    seed: int,
    checkpoint: str = CHECKPOINT_VIT,
    attn_implementation: str = "sdpa",
) -> tuple[ViTForImageClassification, dict]:
    """
    Carrega o ViT pré-treinado no ImageNet e troca o head Linear(768 -> 1000)
    por um Linear(768 -> 7) novo, com pesos aleatórios. O corpo (patch
    embedding + 12 blocos) é mantido.

    revisao: SHA completo (40 caracteres) do commit do checkpoint no Hub.
        Obrigatório, sem padrão, pelo mesmo motivo do dataset: o mesmo nome de
        checkpoint (ou "main") pode apontar para pesos diferentes depois de uma
        atualização (reprodutibilidade, R7).
    seed: semente fixada imediatamente antes do carregamento. O head novo é
        inicializado com o RNG global do torch; sem a semente logo antes, ele
        muda entre execuções, e no linear probe (só o head treina) essa variação
        pode ser do tamanho da diferença entre dois experimentos.
    attn_implementation: "sdpa" (padrão) no treino, por ser mais rápido;
        "eager" na Fase 6, ao recarregar o melhor.pt, porque só ele devolve
        os pesos de atenção com output_attentions=True. Não muda a matemática,
        por isso fica fora do dict de metadados (e da config conferida).

    O aviso de que o classifier foi reinicializado é esperado: é a evidência
    (R21) de que o head de 1000 classes foi trocado.

    Retorna (modelo, metadados): metadados é um dict compatível com JSON
    (checkpoint, revisão, tamanho de imagem esperado, semente) para compor o
    config_modelo do experimento junto com o dict devolvido por `congelar`.
    """
    # Valida ANTES do download: None cairia em "main" sem aviso, e "main" não fixa a versão
    assert isinstance(revisao, str) and _SHA_COMMIT.fullmatch(revisao), (
        f"revisao deve ser o SHA completo do commit (40 caracteres hexadecimais), recebi {revisao!r}"
    )

    torch.manual_seed(seed)
    modelo = ViTForImageClassification.from_pretrained(
        checkpoint,
        revision=revisao,
        num_labels=len(INDICE_PARA_CLASSE),
        id2label=INDICE_PARA_CLASSE,
        label2id=CLASSE_PARA_INDICE,
        ignore_mismatched_sizes=True,  # o checkpoint tem head 1000×768; pedimos 7×768
        # fp32 explícito: pesos em fp16 quebram o GradScaler e as atualizações
        # pequenas do fine-tuning se perderiam por falta de precisão
        dtype=torch.float32,
        attn_implementation=attn_implementation,
    )

    # Confirma o que o aviso promete, em vez de confiar só no log
    n_classes = len(INDICE_PARA_CLASSE)
    assert modelo.classifier.out_features == n_classes, (
        f"Head com {modelo.classifier.out_features} saídas, esperava {n_classes}"
    )
    dtypes = {p.dtype for p in modelo.parameters()}
    assert dtypes == {torch.float32}, f"Parâmetros fora de fp32: {dtypes}"

    metadados = {
        "checkpoint": checkpoint,
        "revisao": revisao,
        "img": modelo.config.image_size,  # o transform tem de usar este tamanho (224)
        "seed": seed,
    }
    print(f"ViT carregado: {checkpoint}@{revisao[:8]}, head {modelo.classifier.in_features} -> "
          f"{n_classes}, imagem {metadados['img']} px, atenção '{attn_implementation}'.")
    return modelo, metadados


def congelar(modelo: ViTForImageClassification, blocos_treinaveis: int,
             treinar_layernorm_final: bool | None = None) -> dict:
    """
    Congela o modelo e libera só o head e os últimos `blocos_treinaveis` blocos.

    * 0: só o head (linear probe, ~5 mil parâmetros);
    * N entre 1 e 11: últimos N blocos + LayerNorm final + head;
    * 12 (todos os blocos): full fine-tuning, inclusive os embeddings
      (patch embedding, cls_token e position_embeddings).

    treinar_layernorm_final: a LayerNorm final fica entre o último bloco e o
    head. Se algum bloco é treinado, ela precisa acompanhar (a distribuição
    que chega nela muda), então fica sempre treinável. No linear probe o
    padrão é congelá-la, para ser um probe puro das features do ImageNet;
    passe True para liberá-la mesmo assim.

    Decida o congelamento ANTES de chamar treinar_modelo: o
    grupos_weight_decay só coloca no otimizador o que está treinável naquele
    momento.

    Congelar (requires_grad=False) não é o mesmo que eval(): dropout e
    BatchNorm dos blocos congelados seguiriam em modo treino. No ViT isso não
    tem efeito, porque ele não tem BatchNorm e o checkpoint tem dropout 0; o
    aviso abaixo avisa se um checkpoint com dropout for usado.

    Retorna um dict (compatível com JSON) para o config do experimento.
    """
    blocos = modelo.vit.encoder.layer
    n_blocos = len(blocos)
    # type(...) is int: isinstance(True, int) é True, e congelar(m, True) viraria 1 bloco
    assert type(blocos_treinaveis) is int and 0 <= blocos_treinaveis <= n_blocos, (
        f"blocos_treinaveis deve ser um inteiro entre 0 e {n_blocos}, recebi {blocos_treinaveis!r}"
    )

    for p in modelo.parameters():
        p.requires_grad = False

    def liberar(modulo) -> None:
        for p in modulo.parameters():
            p.requires_grad = True

    liberar(modelo.classifier)

    for bloco in blocos[n_blocos - blocos_treinaveis:]:  # blocos[12:] é vazio
        liberar(bloco)

    if blocos_treinaveis == n_blocos:
        liberar(modelo.vit.embeddings)

    if treinar_layernorm_final is None:
        treinar_layernorm_final = blocos_treinaveis > 0
    elif blocos_treinaveis > 0 and not treinar_layernorm_final:
        raise ValueError("Com blocos treináveis, a LayerNorm final precisa ser treinável também.")
    if treinar_layernorm_final:
        liberar(modelo.vit.layernorm)

    dropouts = (modelo.config.hidden_dropout_prob, modelo.config.attention_probs_dropout_prob)
    if blocos_treinaveis < n_blocos and any(dropouts):
        print(f"AVISO: dropout {dropouts} nos blocos congelados continua ativo em modelo.train().")

    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    info = {
        "blocos_treinaveis": blocos_treinaveis,
        "layernorm_final_treinavel": treinar_layernorm_final,
        "parametros_totais": total,
        "parametros_treinaveis": treinaveis,
        "pct_treinaveis": round(100 * treinaveis / total, 2),
    }
    print(f"Congelamento: {blocos_treinaveis}/{n_blocos} blocos treináveis, "
          f"{treinaveis:,} de {total:,} parâmetros ({info['pct_treinaveis']}%).")
    return info