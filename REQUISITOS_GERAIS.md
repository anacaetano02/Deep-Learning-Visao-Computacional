# Requisitos gerais — Projeto da Disciplina

**Disciplina:** Deep Learning and Vision — Computer Vision
**Professor(a):** [não informado]
**Prazo:** 10/10/2026 (adiado; prazo original 05/10/2026)
**Formato:** individual

Regras comuns às 4 atividades. O específico de cada atividade fica no `REQUISITOS.md`
da pasta dela. Fontes: "Requisitos do Projeto.pdf", enunciado completo das atividades
e rubrica de avaliação do professor.

## Atividades
| Atividade | Pasta / notebook | Entrega | Rubrica | Requisitos |
|-----------|------------------|---------|---------|------------|
| A1 — Projeto livre com Vision Transformers | `A1_vision_transformers/` | notebook + relatório | seções 2 e 3 | [A1_vision_transformers/REQUISITOS.md](A1_vision_transformers/REQUISITOS.md) |
| A2 — Reconhecimento semântico com CLIP (ADS-16) | `A2_clip_ads16/` | notebook + relatório | seção 4 | [A2_clip_ads16/REQUISITOS.md](A2_clip_ads16/REQUISITOS.md) |
| A3 — Classificador com CNN pré-treinada (Kaggle) | `A3_cnn_kaggle/` | notebook + relatório | seção 1 | [A3_cnn_kaggle/REQUISITOS.md](A3_cnn_kaggle/REQUISITOS.md) |
| A4.1 — Estudo de caso: COVID-19 em raio-X | `A4_estudo_caso_raio_x/` | notebook + relatório | seção 5 | [A4_estudo_caso_raio_x/REQUISITOS.md](A4_estudo_caso_raio_x/REQUISITOS.md) |
| A4.2 — Estudo de caso: tráfego urbano | — | **só relatório** | seção 5 | [A4_estudo_caso_raio_x/REQUISITOS.md](A4_estudo_caso_raio_x/REQUISITOS.md) (R13–R16) |

## Requisitos gerais obrigatórios
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| G1 | Todos os scripts/notebooks rodam no Google Colab com GPU T4 | ⬜ pendente | os 4 notebooks, `requirements.txt` de cada atividade |
| G2 | Um notebook por atividade executável, com os nomes exatos: `A1_vision_transformers.ipynb`, `A2_clip_ads16.ipynb`, `A3_cnn_kaggle.ipynb`, `A4_estudo_caso_raio_x.ipynb` | 🟡 A1 criado | pastas das atividades |
| G3 | Cada notebook informa no início o tempo estimado de execução | ⬜ pendente | célula inicial de cada notebook |
| G4 | Cada notebook informa no início o uso de memória | ⬜ pendente | célula inicial de cada notebook |
| G5 | Relatório técnico único em PDF, nomeado `nome_sobrenome_deep-learning-and-vision_computer-vision.pdf` | ⬜ pendente | `relatorio/` |
| G6 | O relatório cobre, para cada uma das 4 atividades (incluindo a 4.2): definição do problema, justificativas técnicas, métricas e análises críticas | ⬜ pendente | `relatorio/` |
| G7 | Tudo compactado em um `.ZIP` nomeado `nomedoaluno_nomedadisciplina_pd.ZIP` | ⬜ pendente | — |

## Restrições
- O professor não proibiu bibliotecas nem uso de IA (confirmado pelo aluno).
- Tudo precisa caber no Colab T4 (~15 GB de VRAM) e rodar lá do início ao fim.

## Decisões de organização (do aluno)
- Um único repositório Git na raiz `Deep-Learning-Visao-Computacional/`, com uma
  pasta autocontida por atividade (`src/`, `requirements.txt`, `outputs/`,
  `REQUISITOS.md`). Sem pacote compartilhado entre atividades.
- Cada notebook clona o repositório no Colab (disco local do runtime ou Google Drive,
  à escolha do professor) e põe no `sys.path` apenas a pasta da própria atividade.
- Cada notebook salva figuras e tabelas em `outputs/`, que alimentam `relatorio/`.
- Fora do Git: datasets, checkpoints e outputs pesados. Dentro do Git: CSVs de split
  e tabelas de experimentos (reprodutibilidade).
- O ZIP de entrega contém os 4 notebooks executados, as pastas `src/`, os
  `requirements.txt` e o PDF; não contém dados nem checkpoints.

## Critérios de avaliação
Rubrica binária por item ("Não demonstrou" / "Demonstrou o item de rubrica"),
organizada em 5 seções. O mapeamento item → requisito fica no `REQUISITOS.md` de
cada atividade.
1. Transfer learning com CNNs pré-treinadas → A3
2. Arquiteturas Transformer, do mecanismo de atenção ao fine-tuning de BERT → A1
3. Vision Transformers para classificação de imagens → A1
4. Classificação zero-shot e busca semântica com CLIP → A2
5. GANs para síntese condicional e tradução entre domínios; estudos de caso → A4

## Entregáveis
- `A1_vision_transformers.ipynb`, `A2_clip_ads16.ipynb`, `A3_cnn_kaggle.ipynb`,
  `A4_estudo_caso_raio_x.ipynb` (executados).
- `nome_sobrenome_deep-learning-and-vision_computer-vision.pdf`.
- Tudo em `nomedoaluno_nomedadisciplina_pd.ZIP`.

## Dúvidas para o professor
- Qual nome de disciplina usar em `nomedoaluno_nomedadisciplina_pd.ZIP`?
- Pode incluir no ZIP as pastas `src/` com módulos `.py` e o link do repositório, ou
  o professor espera só os notebooks e o PDF?
- Os notebooks devem ser entregues já executados (com outputs salvos)?
