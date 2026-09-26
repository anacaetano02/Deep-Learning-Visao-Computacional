# Requisitos — Atividade 2: Reconhecimento Semântico em Publicidade Visual com CLIP

**Disciplina:** Deep Learning and Vision — Computer Vision
**Professor(a):** [não informado]
**Prazo:** 05/10/2026
**Formato:** individual

> Atividade 2 de 4 do Projeto da Disciplina. Regras comuns (T4, ZIP, PDF, tempo e
> memória no topo do notebook) estão em [../REQUISITOS_GERAIS.md](../REQUISITOS_GERAIS.md).
> Fontes: "Requisitos do Projeto.pdf", enunciado completo da Atividade 2 e rubrica
> do professor (seção 4).

## Objetivo do trabalho
Usar um CLIP pré-treinado, **sem treinar nenhum modelo**, para extrair inteligência
semântica de um corpus de anúncios (ADS-16): ranquear os objetos/conceitos mais
frequentes por similaridade imagem-texto e implementar busca de imagens por consulta
em linguagem natural, analisando criticamente o que o modelo recupera.

## Dados e experimento
- **Tipo de tarefa:** recuperação imagem-texto zero-shot (sem treino supervisionado)
- **Dataset:** ADS-16 (Computational Advertising Dataset), imagens de anúncios em
  16 categorias de produto. Pode ser o corpus inteiro ou um subconjunto de **≥ 500
  imagens selecionadas de forma representativa**. [origem/forma de download a definir]
- **Modelo:** CLIP pré-treinado (embeddings de imagem e texto). [checkpoint e
  biblioteca a definir pelo aluno]
- **Framework:** PyTorch + [Hugging Face `transformers` ou `open_clip` — a definir]
- **Ambiente:** Google Colab com GPU T4 (ver requisitos gerais)
- **Métrica(s) exigida(s):** cosine similarity imagem-texto; score médio de
  similaridade e frequência de ocorrência acima do threshold (ranking)
- **Meta de desempenho (se houver):** nenhuma
- **Protocolo de avaliação exigido (se houver):** nenhum formal; a avaliação da busca
  é qualitativa (análise crítica de cada consulta)

## Requisitos obrigatórios

### 2.1 — Ranking de objetos por frequência semântica
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R1 | Usar o ADS-16 inteiro ou um subconjunto de ≥ 500 imagens | ⬜ pendente | |
| R2 | Justificar que o subconjunto (se usado) é representativo | ⬜ pendente | A2_clip_ads16.ipynb (markdown); relatório |
| R3 | Pipeline CLIP que calcula a cosine similarity de cada imagem com ≥ 20 descrições distintas de objetos/conceitos | ⬜ pendente | |
| R4 | Definir um threshold de similaridade | ⬜ pendente | |
| R5 | Justificar o threshold escolhido | ⬜ pendente | A2_clip_ads16.ipynb (markdown); relatório |
| R6 | Ranking dos objetos mais frequentes, com score médio de similaridade e frequência de ocorrência acima do threshold | ⬜ pendente | |
| R7 | Visualizar os 5 objetos mais encontrados com exemplos de imagens do corpus que confirmem cada categoria | ⬜ pendente | |

### 2.2 — Busca semântica por consulta textual
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R8 | Busca de imagens por texto: dada uma consulta, retornar as top-5 imagens mais similares do corpus | ⬜ pendente | |
| R9 | Executar ≥ 8 consultas variando em especificidade (genérico → específico) e abstração (concreto → abstrato) | ⬜ pendente | |
| R10 | Documentar os resultados de cada consulta (top-5 exibidas) | ⬜ pendente | A2_clip_ads16.ipynb |
| R11 | Analisar, para cada consulta, se o modelo recupera o que ela descreve ou interpreta de forma inesperada | ⬜ pendente | A2_clip_ads16.ipynb (markdown); relatório |

### Análises escritas (rubrica, seção 4)
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R12 | Analisar o alinhamento das representações visuais e textuais no CLIP | ⬜ pendente | relatório (e/ou markdown no notebook) |
| R13 | Explicar por que o pré-treinamento contrastivo habilita recuperação semântica sem treino supervisionado | ⬜ pendente | relatório (e/ou markdown no notebook) |
| R14 | Comparar o mecanismo de consulta textual do CLIP com a tokenização de sequências no BERT, explicando o papel do padding e da attention mask | ⬜ pendente | relatório (e/ou markdown no notebook) |

### Entrega
| ID | Requisito | Status | Arquivo(s) |
|----|-----------|--------|------------|
| R15 | Notebook `A2_clip_ads16.ipynb`, rodando no Colab T4, com tempo estimado e uso de memória no início | ⬜ pendente | A2_clip_ads16.ipynb |
| R16 | Seção da A2 no relatório: definição do problema, justificativas técnicas, métricas e análise crítica | ⬜ pendente | relatório |

## Requisitos opcionais / bônus
Nenhum definido no enunciado nem na rubrica.

## Restrições (o que NÃO pode)
- **Não treinar nenhum modelo**: só embeddings pré-treinados do CLIP e consultas em
  linguagem natural (nem fine-tuning, nem classificador treinado sobre embeddings).
- Rodar no Colab T4 (ver requisitos gerais).

## Critérios de avaliação
Rubrica binária por item. **Seção 4 — Classificação zero-shot e busca semântica com CLIP:**
- Alinhamento das representações visuais e textuais; por que o pré-treino
  contrastivo habilita recuperação sem treino supervisionado → R12, R13
- Ranking por frequência semântica no ADS-16 com ≥ 20 descrições e visualização dos
  5 mais frequentes com exemplos → R3, R6, R7
- Busca semântica com ≥ 8 consultas variando em especificidade, documentada e
  analisada → R8, R9, R10, R11
- Consulta textual do CLIP × tokenização no BERT, papel do padding e da attention
  mask → R14

## Entregáveis
- `A2_clip_ads16.ipynb` e a seção da A2 no relatório único (ver requisitos gerais).

## Dúvidas para o professor
- De onde obter o ADS-16? Há uma fonte/versão recomendada?
- As ≥ 20 descrições podem incluir as 16 categorias de produto do próprio dataset,
  ou devem ser objetos/conceitos independentes delas (como nos exemplos "a car",
  "text and logo")?
- "Frequência de ocorrência acima do threshold": uma imagem pode contar para vários
  objetos ao mesmo tempo (multi-rótulo), ou só para o de maior similaridade?
