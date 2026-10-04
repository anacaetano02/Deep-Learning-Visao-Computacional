# Requisitos — Atividade 3: Classificador com CNN Pré-treinada

**Disciplina:** Deep Learning and Vision — Computer Vision
**Professor(a):** [não informado]
**Prazo:** 10/10/2026 (adiado; prazo original 05/10/2026)
**Formato:** individual

> Atividade 3 de 4 do Projeto da Disciplina. Regras comuns (T4, ZIP, PDF, tempo e
> memória no topo do notebook) estão em [../REQUISITOS_GERAIS.md](../REQUISITOS_GERAIS.md).
> Fontes: "Requisitos do Projeto.pdf", enunciado completo da Atividade 3 e rubrica
> do professor (seção 1).

## Objetivo do trabalho
Construir um classificador supervisionado de objetos por transfer learning, usando
uma CNN pré-treinada como feature extractor (backbone congelado, só o novo head é
treinado), reportar os resultados e discutir criticamente quais melhorias de
augmentation e normalização seriam testadas e por quê.

## Dados e experimento
- **Tipo de tarefa:** classificação de imagens multiclasse, **7 classes** (bike, cars,
  cats, dogs, flowers, horses, human)
- **Dataset:** Kaggle `pavansanagapati/images-dataset`, versão 1, CC0 (fornecido pelo
  professor). Duas cópias idênticas de 1.803 imagens; usada uma. Após duplicatas exatas
  (md5) e quase-duplicatas (cosseno das features + inspeção visual): **1.744 imagens**.
- **Modelo:** ResNet-50, pesos `IMAGENET1K_V2` do torchvision (escolha justificada no
  notebook)
- **Framework:** PyTorch/torchvision
- **Ambiente:** Google Colab com GPU T4 (ver requisitos gerais)
- **Métrica(s) exigida(s):** accuracy global e accuracy por classe; curvas de loss e
  accuracy por epoch
- **Meta de desempenho (se houver):** nenhuma
- **Protocolo de avaliação exigido (se houver):** nenhum. Adotado: split estratificado
  70/15/15 (seed 42), 1.220/262/262, versionado em `split_imagens.csv`

## Requisitos obrigatórios

### 3.1 — Transfer learning por feature extraction
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R1 | Usar o dataset Kaggle `pavansanagapati/images-dataset` | ✅ atendido | A3_cnn_kaggle.ipynb |
| R2 | Carregar uma CNN pré-treinada (ResNet-50, EfficientNet-B0 ou equivalente compatível com T4) | ✅ atendido | A3_cnn_kaggle.ipynb |
| R3 | Justificar a escolha do modelo pré-treinado para o domínio, considerando a capacidade do Colab T4 e o número de classes | ✅ atendido | A3_cnn_kaggle.ipynb (markdown); relatório |
| R4 | Congelar o backbone | ✅ atendido | A3_cnn_kaggle.ipynb |
| R5 | Substituir o classification head pelo número de classes do dataset | ✅ atendido | A3_cnn_kaggle.ipynb |
| R6 | Treinar apenas a nova camada, em um único treino | ✅ atendido | A3_cnn_kaggle.ipynb |
| R7 | Reportar accuracy global | ✅ atendido | A3_cnn_kaggle.ipynb |
| R8 | Reportar accuracy por classe | ✅ atendido | A3_cnn_kaggle.ipynb |
| R9 | Documentar curvas de loss e de accuracy por epoch | ✅ atendido | A3_cnn_kaggle.ipynb |

### 3.2 — Análise e propostas de melhoria (texto)
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R10 | Discutir por escrito, com base nos resultados, quais melhorias você testaria e por quê | ✅ atendido | A3_cnn_kaggle.ipynb (markdown); relatório |
| R11 | Propor ao menos três estratégias de augmentation com justificativa específica para o domínio do dataset (considerar geometric, color, scale variation) | ✅ atendido | A3_cnn_kaggle.ipynb (markdown); relatório |
| R12 | Discutir a normalização (média/desvio do pré-treinamento do modelo, ex.: ImageNet) | ✅ atendido | A3_cnn_kaggle.ipynb (markdown); relatório |
| R13 | Para cada opção considerada, justificar se seria benéfica para as categorias do dataset e para quais classes poderia introduzir distorções ou prejudicar o aprendizado | ✅ atendido | A3_cnn_kaggle.ipynb (markdown); relatório |
| R14 | Discutir quando usar feature extraction versus fine-tuning, com base no tamanho e no domínio do dataset | ✅ atendido | relatório (e/ou markdown no notebook) |

### Entrega
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R15 | Notebook `A3_cnn_kaggle.ipynb`, rodando no Colab T4, com tempo estimado e uso de memória no início | 🟡 falta o run final com a tag | A3_cnn_kaggle.ipynb |
| R16 | Seção da A3 no relatório: definição do problema, justificativas técnicas, métricas e análise crítica | ⬜ pendente | relatório |

> Status de 04/10/2026: código executado (2ª execução) e textos escritos; os textos foram
> ajustados a pedido da autora e ainda passam pela revisão dela. R16 (relatório PDF) pendente.

## Requisitos opcionais / bônus
Nenhum definido no enunciado nem na rubrica.

## Restrições (o que NÃO pode)
- Backbone **congelado**: nada de fine-tuning das camadas convolucionais na 3.1.
- **Um único treino** na 3.1 (só o novo head).
- As melhorias da 3.2 são **discussão escrita**; o enunciado não pede implementá-las.
- Rodar no Colab T4 (ver requisitos gerais).

## Critérios de avaliação
Rubrica binária por item. **Seção 1 — Transfer learning com CNNs pré-treinadas:**
- CNN pré-treinada carregada, head substituído pelo número de classes, feature
  extraction com backbone congelado → R2, R4, R5, R6
- Curvas de treinamento, accuracy por classe e global → R7, R8, R9
- Ao menos três estratégias de augmentation com justificativa específica para o
  domínio → R11, R13
- Quando usar feature extraction × fine-tuning, com base no tamanho e no domínio → R14
- Escolha do modelo pré-treinado justificada considerando T4 e número de classes → R3

## Entregáveis
- `A3_cnn_kaggle.ipynb` e a seção da A3 no relatório único (ver requisitos gerais).

## Dúvidas para o professor
- Há split de treino/validação/teste recomendado para as 1.800 imagens?
- "Um treino" significa uma única execução, sem busca de hiperparâmetros?
- Implementar e testar alguma melhoria da 3.2 conta a favor, ou basta a discussão?
