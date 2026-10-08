# Requisitos — Atividade 4: Estudos de Caso (4.1 Raio-X COVID-19 e 4.2 Tráfego Urbano)

**Disciplina:** Deep Learning and Vision — Computer Vision
**Professor(a):** [não informado]
**Prazo:** 10/10/2026 (adiado; prazo original 05/10/2026)
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
- **Dataset:** Kaggle `tawsifurrahman/covid19-radiography-database` (COVID-19 Radiography
  Database, Qatar Univ./Univ. Dhaka; v5): COVID 3.616, Normal 10.192, Viral Pneumonia 1.345
  (+ Lung Opacity, fora do escopo); PNG 299×299 (contagens confirmadas no download, 06/10). Após remover
  duplicatas exatas (md5) e quase-duplicatas (correlação de pixels ≥ 0,97): Normal 10.179, Viral Pneumonia
  1.338, COVID 3.337.
  Decisão (04/10): o enunciado não indica o dataset; escolhido por ter as 3 classes do caso e
  ser o mais usado e documentado.
- **Modelos:** **cGAN** (DCGAN condicional, tons de cinza, baixa resolução) para gerar
  COVID-19; classificador do comparativo: **ResNet-18 pré-treinada** no ImageNet (versão
  corrigida do legado), com a mesma configuração nos treinos com × sem imagens geradas.
- **Framework:** PyTorch/torchvision
- **Ambiente:** Google Colab com GPU T4 (ver requisitos gerais)
- **Métrica(s) exigida(s):** **recall da classe COVID-19** (comparação com × sem
  imagens geradas)
- **Meta de desempenho (se houver):** nenhuma
- **Protocolo de avaliação exigido (se houver):** comparar treinos com e sem as
  imagens geradas. Desenho adotado: **escassez só no treino** (≈ 840 / 240 / 120, como no
  legado) e **teste real maior**, tirado do resto do dataset, para o recall de COVID ser
  mensurável; a GAN só vê o treino; o teste só tem imagens reais; vários seeds.

## Requisitos obrigatórios

### 4.1 — Diagnóstico do projeto de raio-X
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R1 | Identificar ao menos 5 problemas técnicos do projeto legado | ✅ atendido no notebook (levar ao relatório, R18) | A4_estudo_caso_raio_x.ipynb (diagnóstico); relatório |
| R2 | Justificar o impacto clínico esperado de cada problema | ✅ atendido no notebook (levar ao relatório, R18) | A4_estudo_caso_raio_x.ipynb (diagnóstico); relatório |

### 4.1 — GAN e experimento
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R3 | Escolher e justificar a abordagem generativa (cGAN para augmentation sintética ou CycleGAN para tradução entre domínios) | ✅ atendido no notebook (levar ao relatório, R18) | A4_estudo_caso_raio_x.ipynb (cGAN); relatório |
| R4 | Implementar a GAN para gerar imagens do domínio médico (classe COVID-19), compatível com T4 | ✅ atendido | src/gan.py; A4_estudo_caso_raio_x.ipynb |
| R5 | Training loop adversarial correto (gerador × discriminador) | ✅ atendido | src/gan.py (treinar_cgan) |
| R6 | Diagnosticar instabilidades de treinamento (mode collapse ou divergência) | ✅ atendido no notebook (levar ao relatório, R18) | A4_estudo_caso_raio_x.ipynb (instabilidade, apêndice pós-hoc); relatório |
| R7 | Aplicar ao menos uma estratégia de mitigação da instabilidade, com evidência de melhoria | ✅ atendido no notebook (levar ao relatório, R18) | A4_estudo_caso_raio_x.ipynb (instabilidade); relatório |
| R8 | Treinar o classificador **sem** as imagens geradas | ✅ atendido | src/classificador.py; A4_estudo_caso_raio_x.ipynb |
| R9 | Treinar o classificador **com** as imagens geradas | ✅ atendido | A4_estudo_caso_raio_x.ipynb |
| R10 | Comparar a recall da classe COVID-19 entre os dois treinos e discutir o impacto | ✅ atendido | A4_estudo_caso_raio_x.ipynb; relatório |

### 4.1 — Plano de melhoria
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R11 | Plano de melhoria integrado que endereça todos os problemas identificados em R1 | ✅ atendido no notebook (levar ao relatório, R18) | A4_estudo_caso_raio_x.ipynb (plano); relatório |
| R12 | O plano cobre modelo, métrica, augmentation sintética e critério de adoção clínica | ✅ atendido no notebook (levar ao relatório, R18) | A4_estudo_caso_raio_x.ipynb (plano); relatório |

### 4.2 — Tráfego urbano (só relatório)
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R13 | Identificar ao menos 4 problemas técnicos/metodológicos da solução de classificação de fluxo | ✅ atendido no notebook (levar ao relatório, R18) | A4_estudo_caso_raio_x.ipynb (4.2); relatório |
| R14 | Explicar os impactos operacionais esperados de cada problema em produção (chuva, noite, ângulos) | ✅ atendido no notebook (levar ao relatório, R18) | A4_estudo_caso_raio_x.ipynb (4.2); relatório |
| R15 | Explicar os riscos de transfer learning de ImageNet para classificação de fluxo de tráfego | ✅ atendido no notebook (levar ao relatório, R18) | A4_estudo_caso_raio_x.ipynb (4.2); relatório |
| R16 | Discutir por escrito a abordagem proposta para cada problema | ✅ atendido no notebook (levar ao relatório, R18) | A4_estudo_caso_raio_x.ipynb (4.2); relatório |

### Entrega
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R17 | Notebook `A4_estudo_caso_raio_x.ipynb` (4.1), rodando no Colab T4, com tempo estimado e uso de memória no início | ✅ atendido | A4_estudo_caso_raio_x.ipynb (topo e apêndice: ~65 min, VRAM 1,10 GB, RAM 3,4 GB; tag a4-entrega) |
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
