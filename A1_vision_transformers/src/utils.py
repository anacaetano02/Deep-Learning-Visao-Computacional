"""Utilitários transversais: cronômetro de etapas com pico de memória (R8, R9) e testes."""
import time
from contextlib import contextmanager

import polars as pl
import torch

try:  # só existe em Linux/macOS (o Colab é Linux); no Windows a RAM fica sem medida
    import resource
except ImportError:
    resource = None


def _pico_ram_gb() -> float | None:
    """Pico de RAM do processo desde o início (ru_maxrss vem em KB no Linux)."""
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


@contextmanager
def cronometrar(etapa: str, registro: list):
    """
    Mede o tempo de parede de um bloco e o pico de memória da GPU durante ele, e anexa
    uma linha em `registro`.

    O que cada número mede:
    * pico_vram_gb: memória ALOCADA pelos tensores do PyTorch (max_memory_allocated);
    * pico_vram_reservada_gb: memória RESERVADA pelo alocador do PyTorch
      (max_memory_reserved), mais próxima do que o nvidia-smi mostra; nenhum dos dois
      inclui o contexto CUDA (~0,3-0,5 GB);
    * pico_ram_processo_gb: pico ACUMULADO do processo principal desde o início do
      notebook (não só desta etapa) e sem os processos dos workers do DataLoader. O registro é uma lista do notebook (e não do módulo) para
    não se perder quando o módulo for recarregado com importlib.reload.

    Uso:
        TEMPOS = []
        with cronometrar("carregar dataset", TEMPOS):
            ...
    """
    usa_gpu = torch.cuda.is_available()
    if usa_gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    inicio = time.perf_counter()
    try:
        yield
    finally:
        if usa_gpu:
            torch.cuda.synchronize()  # espera os kernels assíncronos terminarem
        segundos = time.perf_counter() - inicio
        pico_gpu = torch.cuda.max_memory_allocated() / 1024**3 if usa_gpu else None
        pico_reservada = torch.cuda.max_memory_reserved() / 1024**3 if usa_gpu else None
        pico_ram = _pico_ram_gb()
        registro.append({
            "etapa": etapa,
            "segundos": round(segundos, 1),
            "pico_vram_gb": None if pico_gpu is None else round(pico_gpu, 2),
            "pico_vram_reservada_gb": None if pico_reservada is None else round(pico_reservada, 2),
            "pico_ram_processo_gb": None if pico_ram is None else round(pico_ram, 2),
        })
        vram = "-" if pico_gpu is None else f"{pico_gpu:.2f} GB (reservada {pico_reservada:.2f} GB)"
        ram = "-" if pico_ram is None else f"{pico_ram:.2f} GB"
        print(f"[tempo] {etapa}: {segundos:.1f}s | pico VRAM {vram} | pico RAM do processo {ram}")


def tabela_tempos(registro: list) -> pl.DataFrame:
    """Registro de etapas como tabela, com uma linha de total ao final."""
    schema = {
        "etapa": pl.String,
        "segundos": pl.Float64,
        "pico_vram_gb": pl.Float64,
        "pico_vram_reservada_gb": pl.Float64,
        "pico_ram_processo_gb": pl.Float64,
    }
    df = pl.DataFrame(registro, schema=schema)
    total = pl.DataFrame(
        [{
            "etapa": "TOTAL",
            "segundos": df["segundos"].sum(),
            "pico_vram_gb": df["pico_vram_gb"].max(),
            "pico_vram_reservada_gb": df["pico_vram_reservada_gb"].max(),
            "pico_ram_processo_gb": df["pico_ram_processo_gb"].max(),
        }],
        schema=schema,
    )
    return pl.concat([df, total])


def testar(nome: str, condicao) -> None:
    """Falha o notebook se a condição for falsa; senão imprime OK."""
    assert bool(condicao), f"FALHOU: {nome}"
    print(f"OK  {nome}")
