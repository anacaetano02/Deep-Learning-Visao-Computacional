# Checklist — A1 Vision Transformers

Ordem pensada para o **caminho crítico**: cada fase destrava a seguinte. Os IDs (R1…R31)
são os do [REQUISITOS.md](REQUISITOS.md). Meta: fechar a A1 até **30/09** para sobrar
01–05/10 para A2, A3, A4 e o relatório.

Estado em 26/09: só `src/data.py` tem conteúdo (versão antiga, a refatorar). Demais
módulos, `requirements.txt` e o notebook estão vazios.

---

## Fase 0 — Setup (26/09, ~1h)
- [ ] Mandar ao professor as dúvidas da A1 + gerais + as bloqueantes da A2/A4 (uma mensagem só)
- [x] `git init` na raiz `Deep-Learning-Visao-Computacional/`, `.gitignore` (dados, checkpoints, outputs pesados), primeiro commit, repositório público — https://github.com/anacaetano02/Deep-Learning-Visao-Computacional
- [ ] Notebook: célula de setup (clone local **ou** Drive, `sys.path` para `A1_vision_transformers/`, checagem de GPU T4) — R7
  - [x] Clone (sparse, só `A1_vision_transformers/`), `sys.path`, import do `src` e GPU T4 confirmados no Colab
  - [x] ~~Autoreload~~ não funciona no Colab (Python 3.13 + IPython antigo: `No module named 'imp'`) → substituído por `importlib.reload` dos módulos `src.*` (célula após o clone; reexecutar os imports depois)
  - [x] Mover a célula de `git pull` para **depois** do clone (hoje ela usa `REPO_DIR` antes de ele existir)
  - [ ] Testar o modo `SALVAR_NO_DRIVE = True`
  - [x] Commit + push do notebook
- [x] `requirements.txt` com as versões do Colab T4 (torch sem o rótulo local `+cu128`) — R7
- [ ] Testar o setup num runtime **limpo** do Colab

**Pronto quando:** o notebook clona o repo e importa `src` num Colab novo.

## Fase 1 — Dados (26–27/09) — destrava tudo
- [x] `data.py` → `extrair_metadados` (3 splits empilhados + `split_original` + `idx_original`)
- [x] Contagem direta no notebook: 13.354 linhas, 10.015 `image_id`, 7.470 `lesion_id` (iguais antes/depois), 3.339 duplicatas removidas
- [x] Checar se cópias do mesmo `image_id` têm o mesmo `dx` (e o mesmo `lesion_id`), antes de deduplicar
- [x] `deduplicar_por_imagem`
- [x] `montar_split_por_lesao` (agrupado por `lesion_id`, estratificado por `dx`, seed fixa, asserts de nulos/altura)
- [x] `checar_vazamento(df, coluna)` + `validar_split` → `assert` de zero nos pares, por `image_id` e por `lesion_id` ("Split OK" no Colab)
- [x] Fixar `revision=` no `load_dataset` (`bdd59e10…`, último commit do dataset em 25/01/2023)
- [x] Split em CSV versionado no Git (`split_lesoes.csv`, commit 4d9fad1) — no Colab: "Split versionado confere com o recalculado"
- [ ] Guardar as tabelas de vazamento "antes" (split do HF) — 🟡 geradas e conferidas (84,47/79,77/16,26 e 91,17/88,09/30,04); falta persistir (Drive/download, de preferência em `DIR_REPORT_ASSETS`)
- [x] Tabela `dx × split` com nº de imagens e de lesões (antes e depois), exibida sem cortar colunas
- [x] Mapeamento fixo de rótulos (`CLASSES` + `CLASSE_PARA_INDICE`/`INDICE_PARA_CLASSE`, nomes completos do `dx`)
- [x] `computar_pesos` (só no treino, por imagem, "balanced"; confere com `compute_class_weight` do sklearn)
- [x] `Ham10000Dataset` (busca por `split_original` + `idx_original`, confere `image_id`) + `montar_transforms` (augmentation D4: flips + rotações de 90°, sem fill) + `preparar_dataloaders`
- [x] Teste de sanidade com 1 batch de treino e validação (shape, dtype, faixa, valores negativos, rótulos) + figuras em `report_assets`
- [x] `/revisar` do `data.py` e do notebook com Dataset/DataLoaders

**Pronto quando:** existem DataLoaders de treino/val/teste sem vazamento, e o CSV do split está no Git.

## Fase 2 — EDA mínima (27/09, ~2h) — só o essencial
- [ ] `eda.py`: `distribuicao_classes` + `plot_distribuicao_classes` (split novo) — R3
- [ ] `imagens_por_lesao` — sustenta o split no relatório
- [ ] `grade_exemplos` (4 por classe) — justifica a augmentation — R5
- [ ] Salvar as figuras em `outputs/`
- [ ] ~~Demografia, `dx_type`, tamanhos~~ → só se sobrar tempo (B1)

## Fase 3 — ViT do zero (27–28/09) — maior risco de implementação
- [ ] `transformer.py`: scaled dot-product attention (devolvendo **os pesos**) — R11
- [ ] Multi-head attention com projeções por head + concatenação — R12
- [ ] `TransformerEncoderBlock` (FFN de 2 camadas, LayerNorm, residual) — R14
- [ ] Patch embedding — R15
- [ ] CLS token aprendível — R16
- [ ] Positional encoding — R17
- [ ] ViT completo: imagem → logits, montado sobre o `TransformerEncoderBlock` — R19
- [ ] Notebook: seção de testes (shapes, pesos de atenção somando 1, forward de uma imagem) — R13
- [ ] `/revisar A1_vision_transformers/src/transformer.py`

**Pronto quando:** os testes passam e um batch real passa pelo modelo sem erro.

## Fase 4 — Treino (28–29/09) — deixar a GPU trabalhando enquanto você escreve
- [ ] `training.py`: adaptar do projeto 2 (`fixar_seeds`, loop, `registrar_experimento` com persistência no Drive) — lembrar que o ViT do HF devolve `.logits`
- [ ] Checkpoint por época no Drive (sobrevive à desconexão do Colab)
- [ ] `models.py`: carregar o ViT pré-treinado, trocar o head, definir o congelamento — R21
- [ ] **ViT pré-treinado:** 2–4 experimentos no máximo (ex.: LR × camadas descongeladas) — R5, R21
- [ ] **ViT do zero:** modelo pequeno, 1 configuração principal (+1 variação, se der tempo) — R20
- [ ] Anotar o tempo de cada treino e o pico de memória da GPU — R8, R9

**Pronto quando:** existem 2 modelos treinados salvos e a tabela de experimentos exportada.

## Fase 5 — Avaliação (29/09)
- [ ] `evaluation.py`: F1 macro/weighted, precision/recall por classe, matriz de confusão, curvas de loss — R2, R3
- [ ] Avaliar **no teste** só os modelos finais (uma vez)
- [ ] Tabela comparativa ViT do zero × pré-treinado — R22

## Fase 6 — Attention (29/09)
- [ ] `attention.py`: extrair a attention dos 2 modelos (HF: `output_attentions=True`; o seu: pesos devolvidos pelo módulo)
- [ ] Heatmap de 1 head sobre a imagem (linha do CLS → grid de patches → upsample) — R24
- [ ] Attention maps do **ViT do zero** — R26
- [ ] Escolher 2–3 imagens (ex.: `mel` acertado, `nv` acertado, 1 erro)
- [ ] ~~Rollout, comparar heads/camadas, acerto × erro~~ → bônus (B3, B4)

## Fase 7 — Texto (em paralelo desde a Fase 4; fechar em 30/09)
Escrever **enquanto os treinos rodam**. Cada item vira um markdown no notebook e um trecho do relatório.
- [ ] Por que o HAM10000 e por que o split do HF foi descartado (tabelas de vazamento) — R1, R10
- [ ] Por que attention sem PE não preserva posição — R18
- [ ] Por que essa arquitetura para esse domínio — R4
- [ ] Por que esses hiperparâmetros (citando a tabela de experimentos) — R5
- [ ] O que o modelo aprende a ponderar (interpretação dos heatmaps) — R25, R26
- [ ] Escolha final da arquitetura com base na comparação — R23
- [ ] O que os resultados revelam — R30
- [ ] O que você mudaria — R31
- [ ] DeiT e Swin: o que resolvem que o ViT não resolve — R27 (pode escrever já)
- [ ] ViT × CNN no domínio de dermatoscopia — R28 (pode escrever já)
- [ ] Pré-treino BERT × ViT: o que cada um maximiza — R29 (pode escrever já)

## Fase 8 — Fechamento (30/09)
- [ ] Célula inicial com o tempo estimado de execução e o uso de memória — R8, R9
- [ ] Flag para **pular os treinos** e carregar checkpoints (o professor não precisa esperar horas), mantendo a opção de treinar
- [ ] "Restart & Run All" num Colab T4 **limpo**, nos dois modos (local e Drive) — R6, R7
- [ ] Salvar o notebook com os outputs
- [ ] Tag no Git da versão entregue (o notebook clona essa tag)
- [ ] `/revisar A1_vision_transformers/A1_vision_transformers.ipynb` (visão geral contra o REQUISITOS.md)
- [ ] Atualizar a coluna "Status" do REQUISITOS.md

---

## Atalhos para ganhar tempo
- **Reaproveite o projeto 2** (`training.py`, métricas, matriz de confusão): copie e adapte.
- **Mixed precision (`torch.autocast` + `GradScaler`) na T4:** os treinos ficam bem mais rápidos e usam menos memória.
- **ViT do zero pequeno:** o que conta é a implementação e a comparação, não o F1. Imagem menor, poucas camadas e poucas épocas bastam.
- **Poucos experimentos, bem registrados:** 3–4 linhas na tabela com justificativa valem mais que uma busca grande sem análise.
- **As análises teóricas (R27–R29) não dependem de código:** escreva-as em momentos ociosos, como na espera de um treino.
- **Corte pelos bônus, nunca pelos obrigatórios:** B1–B4 só entram depois de R1–R31.
