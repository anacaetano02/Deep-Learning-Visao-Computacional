# Requisitos — Atividade 1: Projeto Livre (Vision Transformers)

**Disciplina:** Deep Learning and Vision — Computer Vision
**Professor(a):** [não informado]
**Prazo:** 10/10/2026 (adiado; prazo original 05/10/2026)
**Formato:** individual

> Esta é a Atividade 1 de 4 do Projeto da Disciplina (A2: CLIP/ADS-16, A3: CNN/Kaggle,
> A4: estudos de caso de raio-X e tráfego). O relatório em PDF e o ZIP são únicos para
> as 4 atividades. Fontes: "Requisitos do Projeto.pdf", enunciado completo da
> Atividade 1 e rubrica de avaliação do professor (seções 2 e 3 se aplicam à A1).

## Objetivo do trabalho
Implementar "a melhor solução que conseguir" para um problema real de classificação
de imagens, aplicando attention, um Vision Transformer construído sobre esses módulos
e estratégias de pré-treinamento e fine-tuning. O enunciado não prescreve
arquitetura, mas a rubrica exige os componentes abaixo. Na prática:
implementar do zero os blocos do Transformer (attention, multi-head attention,
encoder block, patch embedding, CLS token, positional encoding) e montar um ViT
completo treinado do zero, fazer fine-tuning de um ViT pré-treinado no mesmo
domínio e comparar os dois quantitativamente. Avaliar com métricas e visualizações,
justificar decisões de arquitetura/hiperparâmetros e interpretar por escrito os
attention weights: o que o modelo aprende a ponderar no domínio.

## Dados e experimento
- **Tipo de tarefa:** classificação de imagens multiclasse (7 classes)
- **Dataset:** HAM10000 (lesões de pele dermatoscópicas, domínio saúde), ~10 mil
  imagens, classes `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`. Escolhido pelo
  aluno (enunciado exige dataset público, rotulado, ≥ 2 classes, domínio real).
  Carregado via Hugging Face `datasets` (espelho com splits train 9.577 /
  validation 2.492 / test 1.285 = 13.354 linhas). O split pronto é **inválido**:
  84% da validação e 80% do teste são cópias exatas (`image_id`) de imagens do
  treino, e 91%/88% compartilham `lesion_id` com o treino. Decisão: juntar os
  splits, deduplicar por `image_id` e refazer o split agrupado por `lesion_id`.
- **Modelos (exigidos pela rubrica):**
  - **ViT do zero:** implementação própria em PyTorch, treinada do zero no HAM10000.
    Entrada 128×128, patch 16 (decisão do aluno) → grid 8×8 = 64 tokens + CLS.
    Configuração base (Fase 3): d=192, h=3 (d_k=64), 6 camadas, MLP 768, dropout 0,1, PE aprendível
    (senoidal como opção) → 2,83M parâmetros. Valores finais a confirmar no piloto da Fase 4.
  - **ViT pré-treinado:** `google/vit-base-patch16-224` (decisão do aluno), com head
    substituído (1000 → 7 classes) e fine-tuning no HAM10000. Entrada 224×224,
    normalização mean = std = [0.5, 0.5, 0.5] (do `preprocessor_config.json`),
    patch 16 → grid 14×14 = 196 tokens + CLS, 12 camadas × 12 heads, hidden 768
    (~86M parâmetros).
- **Framework:** PyTorch (módulos próprios) + Hugging Face `transformers` (ViT
  pré-treinado); Polars para tabelas; scikit-learn para métricas
- **Ambiente:** Google Colab com GPU T4 (exigência do professor). O aluno desenvolve
  no VS Code com a extensão do Google Colab, rodando no runtime Colab T4. Isso limita
  a VRAM a ~15 GB e afeta a escolha de batch size/checkpoint e o tamanho do ViT do zero.
- **Organização do código (decisão do aluno):** os módulos auxiliares ficam em `src/`
  e são versionados no Git. O notebook principal clona o repositório no Colab e
  oferece dois destinos, à escolha do professor: o disco local do runtime ou o Google
  Drive dele. A estrutura segue a do projeto 2 (CNN), com EDA, avaliação e
  attention em módulos próprios:
  ```
  A1_vision_transformers/
  ├── requirements.txt
  ├── REQUISITOS.md
  ├── A1_vision_transformers.ipynb
  └── src/
      ├── data.py         # carregar, deduplicar, split, Dataset, transforms, DataLoaders
      ├── eda.py          # distribuição, imagens por lesão, grade de exemplos, demografia
      ├── transformer.py  # do zero: attention, multi-head, encoder block, patch embedding, ViT
      ├── models.py       # ViT pré-treinado (HF): carregar, trocar head, congelamento
      ├── training.py     # seeds, pesos, loop de treino, registro de experimentos
      ├── evaluation.py   # métricas, matriz de confusão, curvas, exemplos de erro
      ├── attention.py    # extrair attention (dos 2 modelos), heatmap, comparar heads
      └── utils.py        # só o que é transversal: log, caminhos, salvar figura
  ```
- **Registro de experimentos (decisão do aluno):** `registrar_experimento` /
  `carregar_experimentos` (adaptados do projeto 2), com persistência em CSV/JSON
  no Drive. Sem MLflow nesta entrega.
- **Métrica(s) exigida(s):** nem o enunciado nem a rubrica especificam. Escolha do
  aluno: F1 macro como principal (dataset desbalanceado, `nv` ≈ 67%), mais F1
  weighted, precision/recall por classe e matriz de confusão normalizada.
- **Meta de desempenho (se houver):** nenhuma
- **Protocolo de avaliação exigido (se houver):** nenhum. Escolha do aluno:
  treino/validação/teste com split agrupado por `lesion_id` e estratificado por
  classe, com o mesmo split para os dois modelos; teste usado só na avaliação final.

## Requisitos obrigatórios

### Enunciado e formato de entrega
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R1 | Usar dataset público, rotulado, de classificação de imagens, com ≥ 2 classes, em domínio real (saúde/varejo/indústria/satélite) | 🟡 HAM10000 (`marmal88/skin_cancer`, revisão fixa), deduplicado e com split por lesão validado; motivo escrito (comparar com o projeto 2); falta 1 frase sobre a adequação ao enunciado e cumprir/qualificar a comparação prometida | src/data.py (carregar, deduplicar, split); relatório (justificativa) |
| R2 | Gerar métricas de avaliação dos modelos | ✅ teste (uma vez): F1 macro, F1 weighted, acurácia, P/R/F1 por classe com support, matriz de confusão; zero 0,4257 [0,381; 0,465], pré 0,7446 [0,688; 0,788] | src/evaluation.py |
| R3 | Gerar visualizações (dados, curvas de treino, resultados) | 🟡 tabelas `dx × split`, figuras de batch, curvas, matrizes, erros mais confiantes; comentário de curvas e matrizes escrito; falta comentar as figuras de batch e de erros | src/eda.py (dados), src/evaluation.py (curvas, matriz de confusão) |
| R4 | Justificar por escrito as decisões de arquitetura: por que essa arquitetura para esse domínio | 🟡 ViT do zero justificado (seção de treino do zero); falta a justificativa da arquitetura do pré-treinado (Base/16, `-224` × `-in21k`, full FT) no markdown | A1_vision_transformers.ipynb (markdown); relatório |
| R5 | Justificar por escrito as decisões de hiperparâmetros: por que esses valores | 🟡 zero (lr, warmup, wd, clipping, épocas, paciência, batch) e pré-processamento (resize, 0,5, D4, triângulos bege, sem ColorJitter) escritos; batch 32 e wd 0,01 do pré escritos; faltam lr 5e-5, 8 épocas e paciência = épocas do pré | src/training.py (tabela de experimentos); A1_vision_transformers.ipynb (markdown); relatório |
| R6 | Notebook nomeado `A1_vision_transformers.ipynb` | ✅ | A1_vision_transformers.ipynb |
| R7 | Notebook roda de ponta a ponta no Google Colab com GPU T4 | 🟡 execução sequencial completa (1→37) com TREINAR=False e download dos checkpoints pelo link; falta runtime novo com a tag (`GIT_REF`) | A1_vision_transformers.ipynb (setup: clone, sys.path), requirements.txt |
| R8 | Início do notebook informa o tempo estimado de execução | ✅ tabela no topo: ~6 min com checkpoints / ~60 min treinando | A1_vision_transformers.ipynb (célula inicial) |
| R9 | Início do notebook informa o uso de memória | ✅ tabela no topo: VRAM alocada/reservada 1,2/1,4 GB (padrão) e 4,1/4,4 GB (treino); RAM ~3,1 GB | A1_vision_transformers.ipynb (célula inicial), src/utils.py |
| R10 | Seção da A1 no relatório técnico em PDF: definição do problema, justificativas técnicas, métricas e análise crítica | ⬜ pendente | nome_sobrenome_deep-learning-and-vision_computer-vision.pdf |

### Implementação do zero (rubrica, seções 2 e 3)
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R11 | Scaled dot-product attention implementada do zero como módulo PyTorch | ✅ `ScaledDotProductAttention`; testes no notebook | src/transformer.py |
| R12 | Multi-head attention do zero, com projeções independentes por head e concatenação das saídas | ✅ `CabecaAtencao` em `ModuleList` + concat + `W_O`; teste de equivalência e texto (d_k = d/h, W_O) | src/transformer.py |
| R13 | Módulos de attention testáveis (testes demonstrados: shapes, pesos somando 1, máscara etc.) | ✅ seção de testes executada no Colab T4 | A1_vision_transformers.ipynb (seção de testes) |
| R14 | `TransformerEncoderBlock` completo: feedforward de duas camadas, LayerNorm e residual connections | ✅ pré-norm | src/transformer.py |
| R15 | Patch embedding implementado | ✅ `PatchEmbedding` (teste Conv2d == unfold + Linear) | src/transformer.py |
| R16 | CLS token aprendível | ✅ | src/transformer.py |
| R17 | Positional encoding aplicado à sequência de tokens do ViT | ✅ aprendível (padrão) ou senoidal | src/transformer.py |
| R18 | Explicar por escrito por que attention sem positional encoding não preserva informação posicional | ✅ prova Attn(PX) = P·Attn(X), testes, figura `r18_patches_embaralhados.png`, consequência no domínio e escolha do PE | A1_vision_transformers.ipynb (markdown); relatório |
| R19 | ViT completo montado a partir do `TransformerEncoderBlock`: recebe uma imagem e retorna logits | ✅ batch real (64,3,128,128) → (64,7); overfit de 16 imagens OK | src/transformer.py |

### Treino e comparação (rubrica, seção 3)
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R20 | Treinar o ViT do zero no HAM10000 | ✅ `vit_zero_v1`: 36/50 épocas (early stopping), melhor F1 macro de validação 0,4907 na época 26, ~35 min na T4; registrado em `experimentos.csv` | src/training.py; A1_vision_transformers.ipynb |
| R21 | Fine-tuning de ViT pré-treinado no mesmo domínio, substituindo o classification head | ✅ `vit_pre_v1`: full fine-tuning (12/12 blocos, 85,8M parâmetros), head 1000→7 (LOAD REPORT no notebook), 8 épocas, melhor F1 macro de validação 0,7733 na época 5, 14,8 min e 4,12 GB de VRAM na T4; registrado em `experimentos.csv` | src/models.py, src/training.py |
| R22 | Tabela quantitativa comparando ViT do zero × ViT pré-treinado | ✅ `tabela_r22_teste.csv`: 2 baselines, F1 com IC 95% por bootstrap pareado por lesão (diferença +0,317 [0,258; 0,374]), resolução, parâmetros, épocas, minutos, VRAM; texto com os confundidores | src/evaluation.py; A1_vision_transformers.ipynb; relatório |
| R23 | Justificar a escolha de arquitetura para o domínio com base nos dados da comparação | 🟡 escrito (F1 maior nas 7 classes, ressalvas de `bkl`/`nv`); falta ligar ao domínio/custo e separar calibração de limiar | A1_vision_transformers.ipynb (markdown); relatório |

### Attention (enunciado + rubrica, seções 2 e 3)
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R24 | Visualizar attention weights como heatmap de ao menos um head, para ao menos um exemplo do domínio | ✅ `plotar_heads`: todas as heads da última camada, CLS → patches, nos dois modelos (critério fixo de imagens) | src/attention.py |
| R25 | Interpretar por escrito o que o modelo aprende a ponderar no domínio, com base nesse heatmap | 🟡 interpretação escrita (pré acompanha a lesão, borda como hipótese, viés de centro nos dois); faltam a head específica (placeholder), a faixa com as 6 imagens, os índices e "semelhantes" em vez de "iguais" | A1_vision_transformers.ipynb (markdown); relatório |
| R26 | Gerar attention maps de ao menos um head **do ViT do zero** e identificar por escrito as regiões emergentes | 🟡 mapas por head do zero, sanidade com ViT aleatório e padrão posicional (1,9× no centro); falta descrever uma head específica do zero e a ressalva do r igual ao aleatório | src/attention.py; A1_vision_transformers.ipynb (markdown); relatório |

### Análises escritas (rubrica, seções 2 e 3)
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R27 | Analisar DeiT e Swin Transformer: o que cada um resolve que o ViT original não resolve | 🟡 escrito; corrigir "é o regime do ViT do zero" (DeiT usa ImageNet-1k, augmentation pesada e destilação) | relatório (e/ou markdown no notebook) |
| R28 | Justificar quando ViTs superam CNNs e quando CNNs são preferíveis, com base no domínio escolhido | 🟡 escrito; trocar "explica o 0,43" por hipótese (sem CNN treinada; confundidores do R22) e trazer argumentos de dermatoscopia | relatório (e/ou markdown no notebook) |
| R29 | Analisar as diferenças entre o pré-treinamento de BERT e de ViT, identificando o que cada estratégia maximiza | ✅ o que cada pré-treino maximiza (MLM bidirecional × rótulo da imagem via CLS) | relatório (e/ou markdown no notebook) |
| R30 | Discutir o que os resultados revelam | 🟡 escrito (recall de `mel`, calibração, queda val → teste); falta citar a previsão feita antes do teste e a acurácia do zero abaixo do "sempre nv" | A1_vision_transformers.ipynb (markdown); relatório |
| R31 | Discutir o que você mudaria (limitações e próximos passos) | ✅ lista de mudanças coerente com as limitações | A1_vision_transformers.ipynb (markdown); relatório |

## Requisitos opcionais / bônus
> ⏸️ **Só começar os bônus depois que as 4 atividades (A1–A4) estiverem finalizadas**:
> todos os requisitos obrigatórios ✅, relatório em PDF e ZIP prontos. Se ainda sobrar
> tempo antes do prazo, escolher os bônus por ordem de valor para a análise.

Nenhum definido no enunciado nem na rubrica. Itens do guia pessoal
([docs/guia_projeto_vit.md](docs/guia_projeto_vit.md)), que não são exigência do professor:
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| B1 | EDA do dataset (desbalanceamento, metadados, duplicatas por lesão) | 🟡 em andamento | src/eda.py |
| B2 | Comparar estratégias de balanceamento (class weights, sampler, focal loss) | ⬜ pendente | src/training.py (pesos, sampler, FocalLoss) |
| B3 | Comparar attention entre heads/camadas ou usar attention rollout | ⬜ pendente | src/attention.py |
| B4 | Comparar attention em casos de acerto × erro | ⬜ pendente | src/attention.py, src/evaluation.py (exemplos de erro) |
| B5 | Métricas por ranking no teste: ROC one-vs-rest com AUC macro, curva Precision-Recall com average precision (mais informativa para `df`/`vasc`, que são raras) e análise do limiar de `mel` (sensibilidade × especificidade). Pré-requisito: `evaluation.py` guardar as probabilidades do softmax, não só as predições | ⬜ pendente | src/evaluation.py; A1_vision_transformers.ipynb |

## Restrições (o que NÃO pode)
- O professor não proibiu bibliotecas nem uso de IA (confirmado pelo aluno).
- O enunciado diz "não há arquitetura prescrita", mas a rubrica exige um ViT
  construído sobre módulos próprios, treinado do zero, e um ViT pré-treinado com
  fine-tuning. A liberdade está nas escolhas dentro disso (tamanho, patch,
  profundidade, checkpoint, estratégia de fine-tuning) e na escolha do modelo final.
- Attention, multi-head attention, encoder block, patch embedding, CLS token e
  positional encoding do ViT do zero **não podem** vir prontos de biblioteca
  (`nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, `timm`, `transformers`).
  Blocos básicos do PyTorch (`nn.Linear`, `nn.LayerNorm`, `nn.Conv2d`, `nn.Dropout`)
  podem ser usados.
- O ViT pré-treinado precisa ter o classification head substituído e passar por
  fine-tuning (não basta feature extraction).
- Tudo precisa rodar no Colab T4: nada de dependência de GPU/arquivos locais que o
  Colab não tenha, e os modelos/batches precisam caber na memória da T4.

## Critérios de avaliação
Rubrica binária por item ("Não demonstrou" / "Demonstrou o item de rubrica").

**Seção 2 — Construir arquiteturas Transformer, do mecanismo de atenção ao
fine-tuning de BERT (A1):**
- Scaled dot-product attention e multi-head attention do zero como módulos PyTorch
  testáveis, com projeções independentes por head e concatenação → R11, R12, R13
- Heatmap de attention weights para ao menos um exemplo do domínio, com
  interpretação escrita → R24, R25
- `TransformerEncoderBlock` completo (FFN de 2 camadas, LayerNorm, residual), usado
  como base do ViT → R14, R19
- Positional encoding na sequência de tokens do ViT e explicação de por que
  attention sem PE não preserva posição → R17, R18
- Diferenças entre pré-treinamento de BERT e de ViT e o que cada um maximiza → R29

**Seção 3 — Projetar Vision Transformers para classificação de imagens (A1):**
- Patch embedding, CLS token aprendível e PE, formando um ViT completo
  (imagem → logits) → R15, R16, R17, R19
- ViT treinado do zero no domínio, com attention maps de ao menos um head e regiões
  emergentes identificadas por escrito → R20, R26
- Fine-tuning de ViT pré-treinado no mesmo domínio, head substituído, comparação
  com o ViT do zero em tabela quantitativa → R21, R22
- Análise de DeiT e Swin Transformer → R27
- Quando ViTs superam CNNs e quando CNNs são preferíveis, no domínio escolhido → R28
- Tabela comparativa ViT do zero × pré-treinado e escolha de arquitetura justificada
  com base nos dados → R22, R23

**Seções que pertencem às outras atividades** (fora do escopo deste arquivo):
- Seção 1 (transfer learning com CNNs, augmentation, feature extraction ×
  fine-tuning) → A3
- Seção 4 (CLIP: alinhamento, ranking, busca semântica, comparação com a
  tokenização do BERT) → A2
- Seção 5 (GANs, raio-X, tráfego) → A4

## Entregáveis
- `A1_vision_transformers.ipynb`, executável no Colab T4, com tempo estimado de
  execução e uso de memória no início.
- A seção da A1 no relatório técnico único
  `nome_sobrenome_deep-learning-and-vision_computer-vision.pdf` (junto com A2–A4).
- Tudo compactado em `nomedoaluno_nomedadisciplina_pd.ZIP` (junto com A2–A4).

## Dúvidas para o professor
- A análise de pré-treinamento BERT × ViT (R29) deve ficar na seção da A1 do
  relatório? A rubrica não diz a qual atividade ela pertence.
- O heatmap de R24/R25 pode ser do ViT pré-treinado, ou precisa ser do ViT do zero
  (que já é exigido em R26)?
- Para o positional encoding do ViT do zero, pode ser aprendível (como no ViT
  original) ou a rubrica espera o senoidal?
- Há métrica ou protocolo de avaliação preferido?
