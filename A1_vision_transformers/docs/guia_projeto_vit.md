# Guia Detalhado — Projeto ViT com HAM10000

Este guia detalha **o que fazer e o que analisar** em cada etapa do cronograma, com perguntas que você deve conseguir responder no relatório final, e trechos de código (não soluções prontas) usando **Polars** e **PyTorch**.

---

## 1. Preparação

### 1.1 Análise Exploratória (EDA)

O objetivo aqui não é só "olhar os dados", é gerar evidência que sustente decisões técnicas futuras (split, balanceamento, augmentation, escolha de métrica).

**O que analisar:**

- **Distribuição das classes** — quantas imagens por classe (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`)
- **Metadados clínicos** (o HAM10000 vem com um CSV de metadados) — idade, sexo, localização da lesão, tipo de diagnóstico (histopatologia, follow-up, consenso). Vale checar se há correlação entre metadados e classe
- **Duplicatas de paciente** — o dataset tem múltiplas imagens do mesmo `lesion_id`. Isso é crítico: se você não agrupar por paciente/lesão no split, pode vazar dados entre treino e teste (data leakage)
- **Dimensões e qualidade das imagens** — resolução original, se há variação de tamanho, presença de artefatos (pelos, marcações, réguas)
- **Exemplos visuais por classe** — plotar algumas imagens de cada classe lado a lado

**Perguntas que essa etapa deve responder no relatório:**

- Qual o grau de desbalanceamento? (ex: razão entre classe majoritária e minoritária)
- Existe risco de data leakage por lesões repetidas? Quantas lesões têm múltiplas imagens?
- As classes são visualmente distinguíveis "a olho nu" ou o problema é difícil mesmo para um humano leigo?
- Há necessidade de pré-processamento de imagem (remoção de pelos, correção de contraste)?

**Exemplo de código (esqueleto, não solução):**

```python
import polars as pl

# Carregar metadados
df = pl.read_csv("HAM10000_metadata.csv")

# Distribuição de classes
df.group_by("dx").agg(
    pl.len().alias("contagem")
).sort("contagem", descending=True)

# Checar duplicatas por lesão (data leakage)
df.group_by("lesion_id").agg(
    pl.len().alias("n_imagens")
).filter(pl.col("n_imagens") > 1)

# Cruzar classe com metadado (ex: idade)
df.group_by("dx").agg(
    pl.col("age").mean().alias("idade_media"),
    pl.col("age").null_count().alias("faltantes")
)
```

> Dica: gere um gráfico de barras da contagem por classe (matplotlib/seaborn) e guarde — ele vai direto para o relatório na seção de "desbalanceamento".

### 1.2 Setup do ambiente

Não é uma etapa de "análise", mas documente no relatório:
- Versões de bibliotecas usadas (torch, transformers, polars)
- Se usou GPU (qual, Colab/local) — isso justifica escolhas de batch size depois

---

## 2. Pré-processamento

### 2.1 Split estratificado (por lesão, não por imagem)

**O que fazer:**
- Split em treino/validação/teste (ex: 70/15/15) estratificado por classe **e** agrupado por `lesion_id`, para que a mesma lesão não apareça em splits diferentes

**Pergunta a responder no relatório:** Como você garantiu que não houve vazamento de dados entre treino e teste? Qual proporção final ficou em cada split, por classe?

**Esqueleto de código:**

```python
from sklearn.model_selection import train_test_split
import polars as pl

# Pegue um registro por lesão (evita vazamento)
lesoes_unicas = df.unique(subset=["lesion_id"])

# Estratificar pelo split usando a coluna "dx"
train_ids, temp_ids = train_test_split(
    lesoes_unicas["lesion_id"].to_list(),
    test_size=0.30,
    stratify=lesoes_unicas["dx"].to_list(),
    random_state=42
)
# repita a lógica para separar temp em val/test
```

### 2.2 Augmentation e normalização

**O que decidir e justificar:**
- Quais transformações fazem sentido clinicamente? (ex: flips e rotações são razoáveis em dermatoscopia; já um "color jitter" agressivo pode distorcer características diagnósticas de cor)
- Normalização deve usar as estatísticas do ImageNet (se for usar ViT pré-treinado nele) ou calcular as estatísticas do próprio dataset?

**Pergunta para o relatório:** Quais augmentations foram usadas e por quê? Isso ajudou a reduzir overfitting (compare curvas de loss com e sem)?

```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    # adicione o que fizer sentido, com justificativa
    transforms.ToTensor(),
    transforms.Normalize(mean=[...], std=[...]),  # decida a origem das stats
])
```

### 2.3 Estratégia de balanceamento

**O que testar (e comparar):**
1. **Class weights** na loss (`nn.CrossEntropyLoss(weight=...)`) — mais simples, não altera a distribuição real dos dados
2. **WeightedRandomSampler** — reamostra o dataset para balancear os batches
3. **Focal Loss** — foca o gradiente nos exemplos difíceis/minoritários

**Como calcular os pesos com Polars:**

```python
contagem = df.group_by("dx").agg(pl.len().alias("n"))
total = contagem["n"].sum()
# peso inversamente proporcional à frequência
pesos = contagem.with_columns(
    (total / (pl.col("n") * contagem.height)).alias("peso")
)
```

**Pergunta para o relatório:** Você testou mais de uma estratégia? Qual métrica usou para decidir qual funcionou melhor (F1 macro, recall por classe)? Mostre uma tabela comparativa.

---

## 3. Implementação do ViT

### 3.1 Carregar o modelo pré-treinado

**Decisões a documentar:**
- Qual checkpoint usar (`vit-base-patch16-224`, `vit-base-patch32-224`, etc.) e por quê (trade-off tamanho vs. desempenho)
- Vai congelar o backbone inteiro, só a head, ou fazer fine-tuning parcial das últimas camadas?

```python
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=7,  # número de classes do HAM10000
    ignore_mismatched_sizes=True  # porque a head original tem 1000 classes
)

# Exemplo de congelamento parcial — pense em qual critério usar
for name, param in model.named_parameters():
    if "encoder.layer.11" not in name and "classifier" not in name:
        param.requires_grad = False
```

### 3.2 Fine-tuning e ajuste de hiperparâmetros

**O que testar sistematicamente (e registrar em tabela):**
- Learning rate (ex: 1e-5, 3e-5, 1e-4)
- Batch size (limitado pela GPU disponível)
- Número de camadas descongeladas
- Otimizador (AdamW é o padrão para ViT) e uso de scheduler (warmup + decay)

**Pergunta para o relatório:** Para cada hiperparâmetro testado, qual foi o efeito na loss de validação? Justifique a configuração final com base nos experimentos, não só na "literatura".

> Sugestão prática: mantenha uma tabela (pode ser um `pl.DataFrame`) registrando cada experimento — LR, batch size, camadas descongeladas, acurácia/F1 de validação. Isso vira uma tabela direto no relatório.

```python
resultados = pl.DataFrame({
    "experimento": [],
    "learning_rate": [],
    "batch_size": [],
    "camadas_descongeladas": [],
    "f1_macro_val": [],
})
# vá adicionando uma linha por experimento com pl.concat
```

---

## 4. Avaliação

### 4.1 Métricas

**Por que acurácia sozinha não basta aqui:** com `nv` representando ~67% dos dados, um modelo "preguiçoso" que sempre prevê `nv` teria alta acurácia e seria inútil.

**O que calcular:**
- F1 macro (trata todas as classes com peso igual — é a métrica mais honesta aqui)
- F1 weighted (para comparação)
- Precision/recall por classe
- Matriz de confusão 7x7 (normalizada por linha, para ver taxa de acerto por classe)

**Pergunta para o relatório:** Quais classes o modelo mais confunde entre si? Faz sentido clinicamente (ex: confundir `bkl` com `nv`, que são visualmente parecidos)?

```python
from sklearn.metrics import classification_report, confusion_matrix
import polars as pl

# y_true, y_pred vêm da sua etapa de inferência
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pl.DataFrame(report)  # facilita exportar como tabela pro relatório

cm = confusion_matrix(y_true, y_pred, normalize="true")
```

### 4.2 Justificativa de arquitetura/hiperparâmetros

Não é uma análise de dados, é uma seção argumentativa: reúna as tabelas dos experimentos da etapa 3 e explique **por que** a configuração final foi escolhida, citando os números obtidos (não apenas "porque funcionou melhor").

---

## 5. Interpretabilidade (Attention)

### 5.1 Extrair attention weights

**Conceito-chave:** o ViT gera, para cada camada e cada head, uma matriz de atenção entre o token `[CLS]` e os patches da imagem. Você precisa capturar essas matrizes durante o forward pass.

```python
outputs = model(pixel_values=imagem, output_attentions=True)
attentions = outputs.attentions  # tupla: uma matriz por camada
# attentions[camada].shape -> [batch, n_heads, n_tokens, n_tokens]
```

**O que decidir:**
- Vai analisar uma head específica de uma camada, ou fazer "attention rollout" (agregando múltiplas camadas)?
- Attention rollout é mais robusto para interpretar o modelo como um todo, mas uma head isolada é mais simples de explicar (e o enunciado pede "ao menos uma head")

### 5.2 Visualizar sobre a imagem original

**O que fazer:**
- Pegar a linha do `[CLS]` na matriz de atenção (relação do CLS com cada patch)
- Redimensionar esse vetor de volta para o grid de patches (ex: 14x14 para patch16/224)
- Sobrepor como um heatmap na imagem original

**Pergunta para o relatório (a parte mais importante da atividade):**
- Para uma imagem de `mel` (melanoma) corretamente classificada, a atenção está concentrada na lesão ou em áreas irrelevantes (fundo, pelos, marcações)?
- Compare um caso de acerto e um caso de erro: a atenção "errada" ajuda a explicar o erro do modelo?
- Diferentes heads da mesma camada olham para regiões diferentes? Isso sugere que heads distintas capturam padrões distintos?

```python
import torch.nn.functional as F

# attn de uma head específica, camada específica
attn_cls_to_patches = attentions[camada][0, head, 0, 1:]  # remove o próprio CLS
grid_size = int(attn_cls_to_patches.shape[0] ** 0.5)
attn_map = attn_cls_to_patches.reshape(grid_size, grid_size)

# upsample para o tamanho da imagem original
attn_map_upsampled = F.interpolate(
    attn_map.unsqueeze(0).unsqueeze(0),
    size=(224, 224),
    mode="bilinear"
)
```

---

## 6. Fechamento

Não há "análise de dados" nova aqui, mas vale organizar o relatório em torno das perguntas que cada etapa te obrigou a responder. Uma estrutura sugerida para o relatório final:

1. Introdução e dataset (respostas da EDA)
2. Metodologia (split, balanceamento, arquitetura — com tabelas de experimentos)
3. Resultados (métricas, matriz de confusão)
4. Interpretabilidade (attention maps + discussão)
5. Conclusão e limitações (ex: desbalanceamento residual, tamanho do dataset)

---

## Resumo: perguntas-chave por etapa (checklist para o relatório)

| Etapa | Pergunta central |
|---|---|
| EDA | Qual o grau de desbalanceamento e há risco de leakage por lesão? |
| Pré-processamento | Como o split evitou leakage e qual estratégia de balanceamento venceu? |
| Implementação ViT | Qual configuração de fine-tuning foi escolhida e com base em quais experimentos? |
| Avaliação | Quais classes o modelo confunde e por quê (F1 macro, matriz de confusão)? |
| Interpretabilidade | A atenção do modelo se concentra na lesão? Isso muda entre heads/camadas? |
