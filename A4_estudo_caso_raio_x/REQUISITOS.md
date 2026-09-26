# Requisitos — Atividade 4: Estudos de Caso (4.1 Raio-X COVID-19 e 4.2 Tráfego Urbano)

**Disciplina:** Deep Learning and Vision — Computer Vision
**Professor(a):** [não informado]
**Prazo:** 05/10/2026
**Formato:** individual

> Atividade 4 de 4 do Projeto da Disciplina. Regras comuns (T4, ZIP, PDF, tempo e
> memória no topo do notebook) estão em [../REQUISITOS_GERAIS.md](../REQUISITOS_GERAIS.md).
> Fontes: "Requisitos do Projeto.pdf", enunciado completo da Atividade 4 e rubrica
> do professor (seção 5).
> **4.1 = notebook + relatório. 4.2 = só relatório** (nada a implementar).

## Objetivo do trabalho
**4.1:** diagnosticar as falhas de um projeto legado de triagem de COVID-19 em
raio-X, implementar uma GAN (condicional ou CycleGAN) para aumentar sinteticamente a
classe COVID-19, medir o impacto na recall dessa classe (com × sem imagens geradas)
e propor um plano de melhoria integrado.
**4.2:** diagnosticar os problemas de um sistema de classificação de fluxo de tráfego
com transfer learning de ImageNet que falhou em produção e discutir como abordar
cada um.

## Contexto dos projetos legados (do enunciado)
- **4.1 Raio-X:** 1.200 imagens em 3 classes — Normal (840), Pneumonia (240),
  COVID-19 (120); split 80/20 sem estratificação; ResNet-18 sem pré-treinamento;
  SGD, learning rate fixo 0.01, 15 epochs; sem augmentation; avaliação só por
  accuracy global. Resultado: 93% de accuracy de treino, 61% de validação. O time
  clínico relata que o modelo "ignora casos positivos". Referência: Nour & Tariq
  (2023), Scientific Reports — https://www.nature.com/articles/s41598-023-37743-4
- **4.2 Tráfego:** classes "livre", "moderado", "congestionado"; ResNet-50
  pré-treinada em ImageNet com fine-tuning em 800 frames de 12 câmeras; 78% de
  accuracy em validação; implantado. Falhas em produção com chuva, à noite e em
  câmeras com ângulos ausentes do treino. Referência:
  https://www.meegle.com/en_us/topics/transfer-learning/transfer-learning-for-traffic-analysis

## Dados e experimento (4.1)
- **Tipo de tarefa:** classificação multiclasse de raio-X de tórax (Normal,
  Pneumonia, COVID-19) + geração sintética da classe minoritária
- **Dataset:** [a definir — o enunciado não indica o dataset do experimento]
- **Modelos:** GAN condicional (cGAN) **ou** CycleGAN, compatível com T4 [escolha do
  aluno]; classificador para o comparativo com × sem dados gerados [a definir]
- **Framework:** PyTorch [a confirmar pelo aluno]
- **Ambiente:** Google Colab com GPU T4 (ver requisitos gerais)
- **Métrica(s) exigida(s):** **recall da classe COVID-19** (comparação com × sem
  imagens geradas)
- **Meta de desempenho (se houver):** nenhuma
- **Protocolo de avaliação exigido (se houver):** comparar treinos com e sem as
  imagens geradas. [split e demais detalhes a definir pelo aluno]

## Requisitos obrigatórios

### 4.1 — Diagnóstico do projeto de raio-X
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R1 | Identificar ao menos 5 problemas técnicos do projeto legado | ⬜ pendente | relatório (e/ou markdown no notebook) |
| R2 | Justificar o impacto clínico esperado de cada problema | ⬜ pendente | relatório (e/ou markdown no notebook) |

### 4.1 — GAN e experimento
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R3 | Escolher e justificar a abordagem generativa (cGAN para augmentation sintética ou CycleGAN para tradução entre domínios) | ⬜ pendente | A4_estudo_caso_raio_x.ipynb (markdown); relatório |
| R4 | Implementar a GAN para gerar imagens do domínio médico (classe COVID-19), compatível com T4 | ⬜ pendente | |
| R5 | Training loop adversarial correto (gerador × discriminador) | ⬜ pendente | |
| R6 | Diagnosticar instabilidades de treinamento (mode collapse ou divergência) | ⬜ pendente | A4_estudo_caso_raio_x.ipynb; relatório |
| R7 | Aplicar ao menos uma estratégia de mitigação da instabilidade, com evidência de melhoria | ⬜ pendente | |
| R8 | Treinar o classificador **sem** as imagens geradas | ⬜ pendente | |
| R9 | Treinar o classificador **com** as imagens geradas | ⬜ pendente | |
| R10 | Comparar a recall da classe COVID-19 entre os dois treinos e discutir o impacto | ⬜ pendente | A4_estudo_caso_raio_x.ipynb; relatório |

### 4.1 — Plano de melhoria
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R11 | Plano de melhoria integrado que endereça todos os problemas identificados em R1 | ⬜ pendente | relatório |
| R12 | O plano cobre modelo, métrica, augmentation sintética e critério de adoção clínica | ⬜ pendente | relatório |

### 4.2 — Tráfego urbano (só relatório)
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R13 | Identificar ao menos 4 problemas técnicos/metodológicos da solução de classificação de fluxo | ⬜ pendente | relatório |
| R14 | Explicar os impactos operacionais esperados de cada problema em produção (chuva, noite, ângulos) | ⬜ pendente | relatório |
| R15 | Explicar os riscos de transfer learning de ImageNet para classificação de fluxo de tráfego | ⬜ pendente | relatório |
| R16 | Discutir por escrito a abordagem proposta para cada problema | ⬜ pendente | relatório |

### Entrega
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R17 | Notebook `A4_estudo_caso_raio_x.ipynb` (4.1), rodando no Colab T4, com tempo estimado e uso de memória no início | ⬜ pendente | A4_estudo_caso_raio_x.ipynb |
| R18 | Seções da 4.1 e da 4.2 no relatório: definição do problema, justificativas técnicas, métricas e análise crítica | ⬜ pendente | relatório |

## Requisitos opcionais / bônus
Nenhum definido no enunciado nem na rubrica.

## Restrições (o que NÃO pode)
- A abordagem generativa precisa ser **cGAN ou CycleGAN** (não outro tipo de modelo
  generativo, como difusão ou VAE).
- A GAN e os treinos precisam caber no Colab T4.
- 4.2: **não é necessário implementar** nada (só discussão escrita).

## Critérios de avaliação
Rubrica binária por item. **Seção 5 — GANs para síntese condicional e tradução entre
domínios; estudos de caso:**
- Ao menos 5 problemas técnicos no projeto de raio-X, com impacto clínico
  justificado → R1, R2
- GAN (condicional ou CycleGAN) para imagens médicas, com training loop adversarial
  correto → R4, R5
- Instabilidades diagnosticadas (mode collapse ou divergência) e ao menos uma
  mitigação com evidência de melhoria → R6, R7
- Impacto da augmentation sintética na recall de COVID-19, com × sem imagens
  geradas → R8, R9, R10
- Ao menos 4 problemas no projeto de tráfego e riscos do transfer learning de
  ImageNet para classificação de fluxo → R13, R14, R15
- Plano de melhoria integrado do raio-X cobrindo modelo, métrica, augmentation
  sintética e critério de adoção clínica → R11, R12

## Entregáveis
- `A4_estudo_caso_raio_x.ipynb` (4.1) e as seções da 4.1 e da 4.2 no relatório único
  (ver requisitos gerais).

## Dúvidas para o professor
- Qual dataset de raio-X usar no experimento da 4.1? O professor fornece os dados do
  "projeto legado" ou o aluno escolhe um dataset público com a classe COVID-19?
- O experimento deve reproduzir a distribuição do projeto legado (840/240/120), ou
  pode usar a distribuição do dataset escolhido?
- O classificador do comparativo com × sem imagens geradas deve ser o do projeto
  legado (ResNet-18 sem pré-treino) ou já uma versão corrigida?
- Com CycleGAN, a tradução seria entre quais domínios (ex.: Pneumonia → COVID-19)?
