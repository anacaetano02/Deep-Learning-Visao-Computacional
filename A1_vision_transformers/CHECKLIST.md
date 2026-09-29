# Checklist — A1 Vision Transformers

Ordem pensada para o **caminho crítico**: cada fase destrava a seguinte. Os IDs (R1…R31)
são os do [REQUISITOS.md](REQUISITOS.md). Meta: fechar a A1 até **30/09** para sobrar
01–05/10 para A2, A3, A4 e o relatório.

Estado em 28/09 (noite): Fases 0, 1 e 3 concluídas (Fase 2 reduzida ao essencial). Fase 4 em
andamento: `training.py` validado no Colab (smoke test), treino do ViT do zero disparado, `models.py`
escrito e corrigido (falta rodar). Próximo: `CONFIG_PRE` + `smoke_pre` e os treinos do pré-treinado.
Meta: A1 funcionalmente fechada até 30/09 à noite (A2, A3 e A4 ainda inteiras).

---

## Fase 0 — Setup (26/09, ~1h)
- [ ] Mandar ao professor as dúvidas da A1 + gerais + as bloqueantes da A2/A4 (uma mensagem só)
- [x] `git init` na raiz `Deep-Learning-Visao-Computacional/`, `.gitignore` (dados, checkpoints, outputs pesados), primeiro commit, repositório público — https://github.com/anacaetano02/Deep-Learning-Visao-Computacional
- [x] Notebook: célula de setup (clone local **ou** Drive, `sys.path` para `A1_vision_transformers/`, checagem de GPU T4) — R7
  - [x] Clone (sparse, só `A1_vision_transformers/`), `sys.path`, import do `src` e GPU T4 confirmados no Colab
  - [x] ~~Autoreload~~ não funciona no Colab (Python 3.13 + IPython antigo: `No module named 'imp'`) → substituído por `importlib.reload` dos módulos `src.*` (célula após o clone; reexecutar os imports depois)
  - [x] Mover a célula de `git pull` para **depois** do clone (hoje ela usa `REPO_DIR` antes de ele existir)
  - [x] Testar o modo `SALVAR_NO_DRIVE = True` (Drive montado, artefatos + zip no Drive)
  - [x] Commit + push do notebook
- [x] `requirements.txt` com as versões do Colab T4 (torch sem o rótulo local `+cu128`) — R7
- [x] Testar o setup num runtime **limpo** do Colab (27/09, exec 1→12 sem erro)

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
- [x] Tabelas de vazamento "antes" (84,47/79,77/16,26 e 91,17/88,09/30,04) salvas em `report_assets` no Drive
- [x] Tabela `dx × split` com nº de imagens e de lesões (antes e depois), exibida sem cortar colunas
- [x] Mapeamento fixo de rótulos (`CLASSES` + `CLASSE_PARA_INDICE`/`INDICE_PARA_CLASSE`, nomes completos do `dx`)
- [x] `computar_pesos` (só no treino, por imagem, "balanced"; confere com `compute_class_weight` do sklearn)
- [x] `Ham10000Dataset` (busca por `split_original` + `idx_original`, confere `image_id`) + `montar_transforms` (augmentation D4: flips + rotações de 90°, sem fill) + `preparar_dataloaders`
- [x] Teste de sanidade com 1 batch de treino e validação (shape, dtype, faixa, valores negativos, rótulos) + figuras em `report_assets`
- [x] `/revisar` do `data.py` e do notebook com Dataset/DataLoaders

**Pronto quando:** existem DataLoaders de treino/val/teste sem vazamento, e o CSV do split está no Git.

## Fase 2 — EDA mínima (27/09, ~2h) — só o essencial
- [ ] `eda.py`: `distribuicao_classes` + `plot_distribuicao_classes` (split novo) — R3
- [ ] ~~`imagens_por_lesao`~~ → cortado (a tabela `dx × split` já traz imagens × lesões)
- [ ] ~~`grade_exemplos`~~ → cortado (figuras de batch já servem de amostra visual); só se sobrar tempo
- [ ] Salvar as figuras em `outputs/`
- [ ] ~~Demografia, `dx_type`, tamanhos~~ → só se sobrar tempo (B1)

## Fase 3 — ViT do zero (27–28/09) — maior risco de implementação
- [x] `transformer.py`: scaled dot-product attention (função + módulo `ScaledDotProductAttention`, devolvendo **os pesos**) — R11
- [x] Multi-head attention com projeções **explícitas** por head (`CabecaAtencao` em `nn.ModuleList`) + concatenação + `W_O` — R12
- [x] `TransformerEncoderBlock` (pré-norm, FFN de 2 camadas, LayerNorm, 2 residuais) — R14
- [x] `PatchEmbedding` (Conv2d kernel=stride=16; teste de equivalência com unfold + Linear) — R15
- [x] CLS token aprendível — R16
- [x] Positional encoding: `tipo_pe="aprendivel"` (padrão) ou `"senoidal"` (buffer) — R17
- [x] ViT completo (128 px, 64+1 tokens, d=192, h=3, 6 camadas, 2,83M parâmetros): imagem → logits + pesos por camada — R19
- [x] Notebook: seção de testes (R11–R19, R18 com patches embaralhados, overfit de 16 imagens: loss 1,899 → 0,036 em 81 passos) executada no Colab T4 — R13
- [ ] Textos R12 e R18 nos markdowns da seção de testes (marcadores "A escrever") → movido para a Fase 7
- [x] `/revisar` do `transformer.py` e do notebook

**Pronto quando:** os testes passam e um batch real passa pelo modelo sem erro.

## Fase 4 — Treino (28–29/09) — deixar a GPU trabalhando enquanto você escreve
- [x] `training.py` (adaptado do projeto 2, revisado pelo mentor e corrigido): AMP fp16 + GradScaler, clipping, warmup + cosseno por passo, weight decay seletivo (sem decay em bias/LayerNorm/CLS/posição), `extrair_logits` (`.logits` do HF ou tupla do ViT próprio), melhor modelo e early stopping pelo F1 macro de validação, loss média ponderada exata, `registrar_experimento` em CSV no Drive
  - [x] `fixar_seeds` (`random`, `numpy`, `torch`, `cuda`, flags do cudnn) **+** `seed=SEED` → `generator` no DataLoader de treino (`data.py`); estado do gerador salvo e restaurado na retomada
  - [x] Hiperparâmetros obrigatórios e só por nome (`epochs`, `lr`, `weight_decay`, `config_modelo`); config salva inclui arquitetura, critério e pesos de classe, normalizada via JSON, e é conferida ao retomar/pular
- [x] Checkpoints com gravação atômica: `ultimo.pt` (retomada; `dir_ultimo` permite deixá-lo fora do Drive), `melhor.pt` + `concluido.json` (pular o treino sem o `ultimo.pt`); testado na CPU (retomada reproduz o treino contínuo)
- [x] Smoke test do ViT do zero no Colab (`max_passos=20`, nome `smoke_zero`): arquivos, CSV, modelo em eval OK
- [x] Diagnóstico do gargalo: DataLoader 0,435 s/batch × GPU 0,089 s/passo → limitado pela CPU, GPU ociosa ~80% (anotado no `relatorio.md`, seção 8) → batch 64 mantido (escolhido pela otimização), cache de imagens adiado
  - [x] Registrar no `relatorio.md` a decisão final sobre o cache e o porquê (não feito; confirmado pelos 35 min do treino)
- [x] `CONFIG_ZERO` único na célula de setup (usado nos testes, no smoke e no treino)
- [x] **ViT do zero:** treino `vit_zero_v1` — R20 — 36/50 épocas (early stopping), melhor F1 macro val 0,4907 (época 26), 58 s/época, 35 min; análise no `relatorio.md` (seção 9)
  - [ ] (Opcional, só com GPU livre no fim) `vit_zero_v2` testando uma única hipótese: cosseno completo (paciência ≥ épocas)
- [x] `models.py`: `carregar_vit_pretreinado(revisao, seed)` → `(modelo, metadados)` (head 1000→7 com `id2label`/`label2id`/`ignore_mismatched_sizes`, fp32, SHA obrigatório, semente antes do head) + `congelar(modelo, blocos_treinaveis)` → dict para o config — R21 (revisado e corrigido; **ainda não rodou**)
  - [ ] R4: justificar `-224` (21k + ajuste 1k, head 1000) × `-in21k` (usado na aula 5, head 21.843) no markdown
- [ ] Notebook: `REVISAO_CHECKPOINT_PRE` (SHA via `HfApi().model_info(...).sha`) + `CONFIG_PRE` no setup (`"modelo"`: checkpoint, revisão, `blocos_treinaveis`; `"treino"`: lr ~5e-5, wd 0,01, ~10 épocas, paciência ~3)
- [ ] `smoke_pre` (`max_passos=20`, `dir_ultimo=DIR_CHECKPOINTS_LOCAL`): conferir fp32, `metadados["img"] == TAMANHO_PRE`, contagem de treináveis, `cls_token`/`position_embeddings` sem decay, s/passo e VRAM com batch 32; depois olhar a lixeira do Drive (`melhor.pt` ~344 MB)
- [ ] **ViT pré-treinado:** 2–4 experimentos no máximo (ex.: só head × últimos N blocos × tudo) — R5, R21 — ponto de partida da aula 5: `lr=5e-5`, `weight_decay=0.01`; pensar se o linear probe precisa de lr maior
- [ ] Anotar o tempo de cada treino e o pico de memória da GPU — R8, R9 (`with cronometrar(...)`, já com VRAM alocada e reservada) + memória total da GPU (`torch.cuda.get_device_properties(0).total_memory`, como na aula 5) no topo do notebook

**Pronto quando:** existem 2 modelos treinados salvos e a tabela de experimentos exportada.

## Fase 5 — Avaliação (29/09)
- [ ] `evaluation.py`: F1 macro/weighted, precision/recall por classe, matriz de confusão, curvas de loss — R2, R3
- [ ] Avaliar **no teste** só os modelos finais (uma vez)
- [ ] R2: registrar por que F1 macro e não só acurácia (a aula 5 usa só acurácia, adequado ao EuroSAT balanceado, não ao HAM10000 com `nv` = 67%)
- [ ] Tabela comparativa ViT do zero × pré-treinado — R22

## Fase 6 — Attention (29/09)
- [ ] `attention.py`: extrair a attention dos 2 modelos (HF: `output_attentions=True`; o seu: pesos devolvidos pelo módulo)
  - [ ] ViT do HF: carregar o checkpoint ajustado com `attn_implementation="eager"` (a SDPA padrão do transformers 5.x não devolve os pesos) — padrão da aula 4
- [ ] Heatmap de 1 head sobre a imagem (linha do CLS → grid de patches → upsample) — R24
- [ ] Attention maps do **ViT do zero** — R26
- [ ] Escolher 2–3 imagens (ex.: `mel` acertado, `nv` acertado, 1 erro) + 1 com vinheta escura forte e a da régua (checar atalhos)
- [ ] Figura por imagem: original + attention map + barras de probabilidade top-3 (softmax), como a inferência da aula 5 — R25
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
- [ ] Flag para **pular os treinos** e carregar checkpoints (o professor não precisa esperar horas), mantendo a opção de treinar — base pronta: com `melhor.pt` + `concluido.json`, `treinar_modelo` só carrega; falta decidir onde publicar esses arquivos (HF Hub, Release ou Drive compartilhado) e os `historico/*.csv`
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
