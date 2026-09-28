"""
Módulo src/training.py (A1 — Vision Transformers)
Loop de treinamento PyTorch: mixed precision (fp16) + GradScaler + clipping,
warmup + cosseno por passo, weight decay seletivo, seleção do melhor modelo
por F1 macro de validação, checkpoint retomável (ultimo.pt / melhor.pt com
marca de concluído, gravados de forma atômica) e registro de experimentos em
CSV no Drive.
"""
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from src.data import CLASSES, SPLIT_TREINO, SPLIT_VALIDACAO


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017). Mesma ideia de ponderação por classe de
    data.computar_pesos (usada aqui como alpha), somada a um fator
    (1 - p_t)^gamma que reduz a contribuição de exemplos já bem
    classificados (p_t alto) e concentra o gradiente nos exemplos difíceis.
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        perda = -(1 - pt).pow(self.gamma) * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            perda = alpha_t * perda
            # Normaliza por sum(alpha_t), como nn.CrossEntropyLoss(weight=...)
            if self.reduction == "mean":
                return perda.sum() / alpha_t.sum()
            elif self.reduction == "sum":
                return perda.sum()
            return perda

        if self.reduction == "mean":
            return perda.mean()
        elif self.reduction == "sum":
            return perda.sum()
        return perda


def fixar_seeds(seed: int = 42, priorizar_velocidade: bool = True) -> None:
    """
    Fixa inicialização de pesos, ordem de shuffle e máscaras de dropout.
    priorizar_velocidade=True mantém cudnn.benchmark (diferenças numéricas
    mínimas entre execuções); False dá reprodutibilidade bit a bit, mais lenta.
    As duas flags são definidas nos dois ramos: sem isso, um deterministic=True
    de uma chamada anterior continuaria valendo na mesma sessão.
    A ordem do shuffle do treino depende também do `seed` passado a
    data.preparar_dataloaders (gerador próprio do DataLoader).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = priorizar_velocidade
    torch.backends.cudnn.deterministic = not priorizar_velocidade

    print(f"Semente {seed} configurada (priorizar_velocidade={priorizar_velocidade}).")


# ---------------------------------------------------------------------------
# Peças do loop
# ---------------------------------------------------------------------------

def extrair_logits(saida) -> torch.Tensor:
    """
    Uniformiza a saída do modelo:
    * ViT do Hugging Face: objeto com .logits;
    * VisionTransformer próprio: tupla (logits, pesos_de_atencao);
    * qualquer outro modelo: o próprio tensor de logits.
    """
    if hasattr(saida, "logits"):
        return saida.logits
    if isinstance(saida, (tuple, list)):
        return saida[0]
    return saida


# Parâmetros sem weight decay além de bias e LayerNorm (ndim <= 1):
# CLS e embeddings de posição, no modelo próprio e no do Hugging Face.
_SEM_DECAY_POR_NOME = {"cls", "pos", "cls_token", "position_embeddings"}


def grupos_weight_decay(modelo: nn.Module, weight_decay: float) -> list[dict]:
    """
    Separa os parâmetros em dois grupos para o AdamW:
    * com decay: matrizes de peso (Linear, Conv);
    * sem decay: bias, LayerNorm, CLS e posição. Encolher esses para zero não
      regulariza nada útil: bias e LayerNorm são deslocamentos/escalas, e CLS e
      posição são embeddings que o modelo precisa manter distintos.
    Parâmetros congelados (requires_grad=False) ficam fora do otimizador.
    """
    com_decay, sem_decay = [], []
    for nome, p in modelo.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or nome.split(".")[-1] in _SEM_DECAY_POR_NOME:
            sem_decay.append(p)
        else:
            com_decay.append(p)

    print(f"Weight decay: {len(com_decay)} tensores com decay, {len(sem_decay)} sem decay.")
    return [
        {"params": com_decay, "weight_decay": weight_decay},
        {"params": sem_decay, "weight_decay": 0.0},
    ]


def scheduler_warmup_cosseno(otimizador, passos_totais: int, passos_warmup: int):
    """
    LR por PASSO (batch), não por época: sobe linearmente de ~0 até o lr
    base nos primeiros passos_warmup passos e depois desce em cosseno até 0.
    Transformers treinados do zero são instáveis no início sem warmup.
    """
    def fator(passo: int) -> float:
        if passo < passos_warmup:
            return (passo + 1) / passos_warmup
        progresso = (passo - passos_warmup) / max(1, passos_totais - passos_warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progresso, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(otimizador, fator)


def _salvar_atomico(obj, caminho: Path) -> None:
    """
    Grava num .tmp e troca pelo arquivo final com os.replace. Se o Colab cair
    no meio da gravação, o arquivo final continua sendo o anterior inteiro (e
    não um arquivo truncado que o torch.load não consegue abrir).
    """
    tmp = caminho.with_name(caminho.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, caminho)


def _escrever_csv_atomico(df: pl.DataFrame, caminho: Path) -> None:
    """Mesma ideia de _salvar_atomico, para CSV."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_name(caminho.name + ".tmp")
    df.write_csv(tmp)
    os.replace(tmp, caminho)


def _resolver_device(device) -> torch.device:
    """
    Aceita "cuda", "cuda:0", torch.device(...) ou None. O .to() recebe o
    dispositivo; autocast e GradScaler recebem só o TIPO (device.type == "cuda").
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _carregar_pesos_melhor(modelo: nn.Module, caminho_melhor: Path, device: torch.device) -> dict:
    """Carrega o melhor.pt em `modelo` (em modo eval) e devolve o checkpoint."""
    ckpt = torch.load(caminho_melhor, map_location=device)
    modelo.load_state_dict(ckpt["modelo"])
    modelo.to(device).eval()
    return ckpt


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def treinar_modelo(
    modelo: nn.Module,
    dataloaders: dict,
    pesos_classe: torch.Tensor,
    nome_experimento: str,
    checkpoint_dir: str | Path,
    dir_resultados: str | Path,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    frac_warmup: float = 0.1,
    paciencia_early_stopping: int = 7,
    min_delta: float = 0.0,
    device=None,
    forcar_do_zero: bool = False,
    clip_grad_norm_max: float = 1.0,
    criterio: nn.Module = None,
    max_passos: int | None = None,
) -> dict:
    """
    Treina 'modelo' escolhendo o melhor pela F1 macro de validação (não pela
    val_loss: a loss ponderada pode cair sem que as classes raras melhorem).
    Só conta como melhora um F1 maior que o melhor anterior + min_delta.

    Checkpoints em checkpoint_dir/nome_experimento/ (gravação atômica):
    * ultimo.pt: estado completo e a config, salvo a cada época, com a marca "concluido";
    * melhor.pt: pesos da época de maior F1 macro.
    Rodar de novo o mesmo experimento retoma do ultimo.pt; se ele estiver
    marcado como concluído, não treina de novo. Retomar com uma config
    diferente da salva é erro (mude o nome ou use forcar_do_zero=True).

    Ao retornar, `modelo` SEMPRE tem os pesos do melhor.pt e está em modo eval,
    tanto depois de treinar quanto no caminho "já concluído".

    max_passos: smoke test. Limita cada época de treino e de validação a N
    batches; o nome do experimento precisa começar com "smoke_" para não
    marcar como concluído (nem sobrescrever) o experimento real.

    criterio: por padrão CrossEntropyLoss(weight=pesos_classe). Use o mesmo
    critério nos modelos que serão comparados. As losses por época são a média
    ponderada exata (soma de loss·peso / soma dos pesos) para critérios que
    normalizam pela soma dos pesos dos alvos, como a CE com weight e a
    FocalLoss com alpha=pesos_classe.

    O histórico por época vai para dir_resultados/historico/<nome>.csv.
    """
    assert set(dataloaders) >= {SPLIT_TREINO, SPLIT_VALIDACAO}, (
        f"dataloaders precisa ter as chaves {SPLIT_TREINO!r} e {SPLIT_VALIDACAO!r}; "
        f"recebeu {sorted(dataloaders)}"
    )
    if max_passos is not None:
        assert nome_experimento.startswith("smoke_"), (
            "Com max_passos (smoke test), use um nome começando com 'smoke_' para não "
            "marcar o experimento real como concluído"
        )
    loader_treino, loader_val = dataloaders[SPLIT_TREINO], dataloaders[SPLIT_VALIDACAO]

    dir_exp = Path(checkpoint_dir) / nome_experimento
    dir_exp.mkdir(parents=True, exist_ok=True)
    caminho_ultimo = dir_exp / "ultimo.pt"
    caminho_melhor = dir_exp / "melhor.pt"
    caminho_historico = Path(dir_resultados) / "historico" / f"{nome_experimento}.csv"

    device = _resolver_device(device)
    usa_amp = device.type == "cuda"  # fp16 + GradScaler: a T4 não tem bf16
    modelo = modelo.to(device)

    pesos_classe = pesos_classe.to(device)
    criterio = criterio if criterio is not None else nn.CrossEntropyLoss(weight=pesos_classe)
    criterio = criterio.to(device)  # o weight da CE é buffer: vai junto para a GPU
    otimizador = torch.optim.AdamW(grupos_weight_decay(modelo, weight_decay), lr=lr)

    passos_por_epoca = len(loader_treino) if max_passos is None else min(max_passos, len(loader_treino))
    passos_totais = epochs * passos_por_epoca
    scheduler = scheduler_warmup_cosseno(otimizador, passos_totais, int(frac_warmup * passos_totais))
    scaler = torch.amp.GradScaler(device.type) if usa_amp else None

    # Tudo o que muda a trajetória do treino: salvo no checkpoint e conferido na retomada
    config = {
        "epochs": epochs, "lr": lr, "weight_decay": weight_decay, "frac_warmup": frac_warmup,
        "paciencia_early_stopping": paciencia_early_stopping, "min_delta": min_delta,
        "clip_grad_norm_max": clip_grad_norm_max, "batch_size": loader_treino.batch_size,
        "max_passos": max_passos,
    }

    historico = {
        "epoca": [], "train_loss": [], "val_loss": [], "val_f1_macro": [],
        "grad_norm": [], "batches_overflow": [], "lr": [], "segundos": [],
    }
    melhor_f1 = -1.0
    epocas_sem_melhora = 0
    epoca_inicial = 1

    if caminho_ultimo.exists() and not forcar_do_zero:
        estado = torch.load(caminho_ultimo, map_location=device)
        diferencas = {
            k: (estado["config"].get(k), v) for k, v in config.items() if estado["config"].get(k) != v
        }
        if diferencas:
            raise ValueError(
                f"'{nome_experimento}' já tem checkpoint com outra config "
                f"(chave: (salvo, atual)) {diferencas}. Mude nome_experimento ou use forcar_do_zero=True."
            )

        if estado["concluido"]:
            ckpt = _carregar_pesos_melhor(modelo, caminho_melhor, device)
            print(f"'{nome_experimento}' já concluído: melhor.pt carregado (época {ckpt['epoca']}, "
                  f"F1 macro de validação={ckpt['val_f1_macro']:.4f}). "
                  f"Use forcar_do_zero=True para treinar de novo.")
            return estado["historico"]

        modelo.load_state_dict(estado["modelo"])
        otimizador.load_state_dict(estado["otimizador"])
        scheduler.load_state_dict(estado["scheduler"])
        if scaler and estado.get("scaler"):
            scaler.load_state_dict(estado["scaler"])
        historico = estado["historico"]
        melhor_f1 = estado["melhor_f1"]
        epocas_sem_melhora = estado["epocas_sem_melhora"]
        epoca_inicial = estado["epoca"] + 1
        print(f"Retomando '{nome_experimento}' da época {epoca_inicial}.")

    print(f"Treino de '{nome_experimento}' em {device} (AMP {'ativo' if usa_amp else 'inativo'}), "
          f"até {epochs} épocas, {passos_por_epoca} passos por época.")

    def salvar_ultimo(epoca: int, concluido: bool) -> None:
        _salvar_atomico(
            {
                "modelo": modelo.state_dict(),
                "otimizador": otimizador.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "historico": historico,
                "melhor_f1": melhor_f1,
                "epocas_sem_melhora": epocas_sem_melhora,
                "epoca": epoca,
                "concluido": concluido,
                "config": config,
            },
            caminho_ultimo,
        )

    for epoca in range(epoca_inicial, epochs + 1):
        t0 = time.time()

        # ---- treino ----
        # Acumuladores na GPU: um .item() por passo forçaria sincronizar CPU e GPU
        # a cada batch; aqui a conversão acontece uma vez por época.
        modelo.train()
        soma_loss = torch.zeros((), device=device)
        soma_pesos = torch.zeros((), device=device)
        soma_normas = torch.zeros((), device=device)
        n_normas = torch.zeros((), device=device)
        n_overflow = torch.zeros((), device=device)

        for passo, (imgs, rotulos) in enumerate(loader_treino):
            if max_passos is not None and passo >= max_passos:
                break
            imgs = imgs.to(device, non_blocking=True)
            rotulos = rotulos.to(device, non_blocking=True)

            otimizador.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=usa_amp):
                logits = extrair_logits(modelo(imgs))
                loss = criterio(logits, rotulos)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(otimizador)  # antes de medir/limitar a norma real
            else:
                loss.backward()

            max_norm = clip_grad_norm_max if clip_grad_norm_max is not None else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=max_norm)

            # inf/NaN = overflow do fp16: o scaler pula esse passo do otimizador
            finito = torch.isfinite(grad_norm)
            soma_normas += torch.where(finito, grad_norm, torch.zeros_like(grad_norm))
            n_normas += finito
            n_overflow += ~finito

            if scaler:
                scaler.step(otimizador)
                scaler.update()
            else:
                otimizador.step()
            # Por passo. Quando o scaler pula um passo por overflow, o scheduler avança
            # mesmo assim e o PyTorch avisa "lr_scheduler.step() before optimizer.step()"
            # nos primeiros passos: é esperado (ver a coluna batches_overflow).
            scheduler.step()

            # Loss média ponderada: a CE com weight divide pela soma dos pesos do batch
            peso_lote = pesos_classe[rotulos].sum()
            soma_loss += loss.detach().float() * peso_lote
            soma_pesos += peso_lote

        train_loss = (soma_loss / soma_pesos).item()
        n_normas_int = int(n_normas.item())
        grad_norm_medio = (soma_normas / n_normas).item() if n_normas_int else float("nan")
        n_batches_overflow = int(n_overflow.item())

        # ---- validação ----
        modelo.eval()
        soma_loss_val = torch.zeros((), device=device)
        soma_pesos_val = torch.zeros((), device=device)
        preds, alvos = [], []
        with torch.no_grad():
            for passo, (imgs, rotulos) in enumerate(loader_val):
                if max_passos is not None and passo >= max_passos:
                    break
                imgs = imgs.to(device, non_blocking=True)
                rotulos = rotulos.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=usa_amp):
                    logits = extrair_logits(modelo(imgs))
                    loss = criterio(logits, rotulos)

                peso_lote = pesos_classe[rotulos].sum()
                soma_loss_val += loss.float() * peso_lote
                soma_pesos_val += peso_lote
                preds.append(logits.argmax(dim=1))
                alvos.append(rotulos)

        val_loss = (soma_loss_val / soma_pesos_val).item()
        val_f1 = float(f1_score(
            torch.cat(alvos).cpu().numpy(), torch.cat(preds).cpu().numpy(),
            average="macro", labels=list(range(len(CLASSES))), zero_division=0,
        ))

        dt = time.time() - t0
        historico["epoca"].append(epoca)
        historico["train_loss"].append(train_loss)
        historico["val_loss"].append(val_loss)
        historico["val_f1_macro"].append(val_f1)
        historico["grad_norm"].append(grad_norm_medio)
        historico["batches_overflow"].append(n_batches_overflow)
        historico["lr"].append(otimizador.param_groups[0]["lr"])
        historico["segundos"].append(dt)

        overflow_str = f" overflow={n_batches_overflow}batches" if n_batches_overflow else ""
        print(
            f"[{nome_experimento}] época {epoca}/{epochs} — "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1_macro={val_f1:.4f} "
            f"grad_norm={grad_norm_medio:.4f}{overflow_str} ({dt:.1f}s)"
        )

        if val_f1 > melhor_f1 + min_delta:
            melhor_f1 = val_f1
            epocas_sem_melhora = 0
            _salvar_atomico(
                {"modelo": modelo.state_dict(), "epoca": epoca, "val_f1_macro": melhor_f1, "config": config},
                caminho_melhor,
            )
        else:
            epocas_sem_melhora += 1

        parar = epocas_sem_melhora >= paciencia_early_stopping
        # Estado completo a cada época (é o que permite retomar), marcado como
        # concluído na última época ou no early stopping
        salvar_ultimo(epoca, concluido=parar or epoca == epochs)
        _escrever_csv_atomico(pl.DataFrame(historico), caminho_historico)

        if parar:
            print(f"Early stopping em '{nome_experimento}' na época {epoca} "
                  f"(sem melhora por {paciencia_early_stopping} épocas).")
            break

    # O modelo em memória tem os pesos da ÚLTIMA época; a seleção foi pelo melhor F1
    ckpt = _carregar_pesos_melhor(modelo, caminho_melhor, device)
    print(f"Treino de '{nome_experimento}' concluído. Melhor F1 macro={melhor_f1:.4f} "
          f"(época {ckpt['epoca']}); modelo com os pesos de '{caminho_melhor}'.")
    return historico


def carregar_melhor_modelo(modelo: nn.Module, checkpoint_dir: str | Path,
                           nome_experimento: str, device=None) -> nn.Module:
    """Recarrega os pesos do melhor.pt salvo por treinar_modelo (modelo em modo eval)."""
    caminho = Path(checkpoint_dir) / nome_experimento / "melhor.pt"
    ckpt = _carregar_pesos_melhor(modelo, caminho, _resolver_device(device))
    print(f"Modelo recarregado de '{caminho}' (época {ckpt['epoca']}, "
          f"F1 macro de validação={ckpt['val_f1_macro']:.4f}).")
    return modelo


# ---------------------------------------------------------------------------
# Registro de experimentos (CSV no Drive; sobrevive a importlib.reload)
# ---------------------------------------------------------------------------

_CHAVES_METRICA_NAO_TABULAVEIS = {"relatorio_por_classe", "matriz_confusao"}


def registrar_experimento(nome: str, tipo: str, hiperparametros: dict, metricas: dict,
                          dir_resultados: str | Path, notas: str = "") -> None:
    """
    Grava uma linha por experimento em dir_resultados/experimentos.csv.
    Registrar de novo um experimento com o mesmo nome substitui a linha
    antiga, então re-rodar a célula não duplica resultados.
    tipo: 'vit_zero' | 'vit_pre' | ... ou 'busca_<tipo>' para tentativas
    comparadas só por validação.
    """
    caminho = Path(dir_resultados) / "experimentos.csv"

    linha = {
        "data": str(np.datetime64("now")),
        "nome": nome,
        "tipo": tipo,
        **{f"hp_{k}": str(v) for k, v in hiperparametros.items()},
        **{f"metrica_{k}": str(v) for k, v in metricas.items() if k not in _CHAVES_METRICA_NAO_TABULAVEIS},
        "notas": notas,
    }
    # Tudo como texto: experimentos diferentes têm colunas e tipos diferentes,
    # e o concat diagonal só junta sem conflito se os tipos forem iguais
    nova = pl.DataFrame([linha], schema={k: pl.String for k in linha})

    if caminho.exists():
        antigo = pl.read_csv(caminho, infer_schema=False).filter(pl.col("nome") != nome)
        nova = pl.concat([antigo, nova], how="diagonal")

    _escrever_csv_atomico(nova, caminho)

    # Métricas não tabuláveis (matriz de confusão, relatório por classe) em JSON à parte
    extras = {k: v for k, v in metricas.items() if k in _CHAVES_METRICA_NAO_TABULAVEIS}
    if extras:
        caminho_extras = Path(dir_resultados) / "metricas_detalhadas" / f"{nome}.json"
        caminho_extras.parent.mkdir(parents=True, exist_ok=True)
        caminho_extras.write_text(json.dumps(extras, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"Experimento '{nome}' registrado em '{caminho}'.")


def carregar_experimentos(dir_resultados: str | Path, apenas_busca: bool | None = None) -> pl.DataFrame:
    """Lê experimentos.csv (tipos inferidos); apenas_busca filtra os 'busca_*'."""
    caminho = Path(dir_resultados) / "experimentos.csv"
    if not caminho.exists():
        return pl.DataFrame()

    df = pl.read_csv(caminho)
    if apenas_busca is True:
        df = df.filter(pl.col("tipo").str.starts_with("busca_"))
    elif apenas_busca is False:
        df = df.filter(~pl.col("tipo").str.starts_with("busca_"))
    return df
