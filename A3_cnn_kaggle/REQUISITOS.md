# Requisitos — Atividade 3: Classificador com CNN Pré-treinada

**Disciplina:** Deep Learning and Vision — Computer Vision
**Professor(a):** [não informado]
**Prazo:** 05/10/2026
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
- **Tipo de tarefa:** classificação de imagens multiclasse [número de classes a
  confirmar no dataset]
- **Dataset:** Kaggle `pavansanagapati/images-dataset`, 1.800 imagens em categorias
  de objetos (fornecido pelo professor)
- **Modelo:** CNN pré-treinada — ResNet-50, EfficientNet-B0 ou equivalente
  compatível com Colab T4 [escolha do aluno, a justificar]
- **Framework:** PyTorch/torchvision [a confirmar pelo aluno]
- **Ambiente:** Google Colab com GPU T4 (ver requisitos gerais)
- **Métrica(s) exigida(s):** accuracy global e accuracy por classe; curvas de loss e
  accuracy por epoch
- **Meta de desempenho (se houver):** nenhuma
- **Protocolo de avaliação exigido (se houver):** nenhum. [split a definir pelo aluno]

## Requisitos obrigatórios

### 3.1 — Transfer learning por feature extraction
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R1 | Usar o dataset Kaggle `pavansanagapati/images-dataset` | ⬜ pendente | |
| R2 | Carregar uma CNN pré-treinada (ResNet-50, EfficientNet-B0 ou equivalente compatível com T4) | ⬜ pendente | |
| R3 | Justificar a escolha do modelo pré-treinado para o domínio, considerando a capacidade do Colab T4 e o número de classes | ⬜ pendente | A3_cnn_kaggle.ipynb (markdown); relatório |
| R4 | Congelar o backbone | ⬜ pendente | |
| R5 | Substituir o classification head pelo número de classes do dataset | ⬜ pendente | |
| R6 | Treinar apenas a nova camada, em um único treino | ⬜ pendente | |
| R7 | Reportar accuracy global | ⬜ pendente | |
| R8 | Reportar accuracy por classe | ⬜ pendente | |
| R9 | Documentar curvas de loss e de accuracy por epoch | ⬜ pendente | |

### 3.2 — Análise e propostas de melhoria (texto)
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R10 | Discutir por escrito, com base nos resultados, quais melhorias você testaria e por quê | ⬜ pendente | A3_cnn_kaggle.ipynb (markdown); relatório |
| R11 | Propor ao menos três estratégias de augmentation com justificativa específica para o domínio do dataset (considerar geometric, color, scale variation) | ⬜ pendente | A3_cnn_kaggle.ipynb (markdown); relatório |
| R12 | Discutir a normalização (média/desvio do pré-treinamento do modelo, ex.: ImageNet) | ⬜ pendente | A3_cnn_kaggle.ipynb (markdown); relatório |
| R13 | Para cada opção considerada, justificar se seria benéfica para as categorias do dataset e para quais classes poderia introduzir distorções ou prejudicar o aprendizado | ⬜ pendente | A3_cnn_kaggle.ipynb (markdown); relatório |
| R14 | Discutir quando usar feature extraction versus fine-tuning, com base no tamanho e no domínio do dataset | ⬜ pendente | relatório (e/ou markdown no notebook) |

### Entrega
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R15 | Notebook `A3_cnn_kaggle.ipynb`, rodando no Colab T4, com tempo estimado e uso de memória no início | ⬜ pendente | A3_cnn_kaggle.ipynb |
| R16 | Seção da A3 no relatório: definição do problema, justificativas técnicas, métricas e análise crítica | ⬜ pendente | relatório |

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
