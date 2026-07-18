#!/usr/bin/env python3
"""Mistura de cores no multilayer hierarchical k-SEP.

Cada partícula recebe a cor da sua camada inicial e conserva essa cor ao se
mover. O script mede a aproximação da distribuição de cada cor à medida
uniforme sobre as camadas.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import platform
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

try:
    from backend_numba import NUMBA_DISPONIVEL, advance_events_numba
except ImportError:
    NUMBA_DISPONIVEL = False
    advance_events_numba = None


@dataclass
class Parametros:
    N: int = 200
    k: int = 4
    densidade_inicial: float = 0.35
    lambda_vertical: float = 1.0
    tempo_final: float = 8.0
    epsilon_equilibrio: float = 0.10
    janela_equilibrio: float = 0.50
    seed: int = 20260719

    def validar(self) -> None:
        if self.N < 4:
            raise ValueError("Use N >= 4.")
        if self.k < 1:
            raise ValueError("Use k >= 1.")
        if not 0 < self.densidade_inicial < 1:
            raise ValueError("A densidade deve estar estritamente entre 0 e 1.")
        if self.lambda_vertical < 0:
            raise ValueError("lambda_vertical deve ser não negativo.")
        if self.tempo_final <= 0:
            raise ValueError("tempo_final deve ser positivo.")
        if not 0 < self.epsilon_equilibrio < 1:
            raise ValueError("epsilon_equilibrio deve estar entre 0 e 1.")
        if self.janela_equilibrio < 0:
            raise ValueError("janela_equilibrio deve ser não negativa.")


@dataclass(frozen=True)
class EventoVertical:
    origem: int
    destino: int
    x: int
    sucesso: bool


def _carregar_backend() -> Optional[ctypes.CDLL]:
    diretorio = Path(__file__).resolve().parent
    fonte = diretorio / "backend_multilayer.c"
    sistema = platform.system()
    extensao = ".dylib" if sistema == "Darwin" else ".so"
    biblioteca = diretorio / f"backend_multilayer{extensao}"

    try:
        precisa_compilar = (
            not biblioteca.exists()
            or biblioteca.stat().st_mtime < fonte.stat().st_mtime
        )
        if precisa_compilar:
            compilador = next(
                (caminho for nome in ("cc", "clang", "gcc")
                 if (caminho := shutil.which(nome)) is not None),
                None,
            )
            if compilador is None:
                raise OSError("nenhum compilador C foi encontrado")
            opcoes_biblioteca = ["-dynamiclib"] if sistema == "Darwin" else ["-shared"]
            subprocess.run(
                [
                    compilador,
                    "-O3",
                    "-std=c11",
                    *opcoes_biblioteca,
                    "-fPIC",
                    str(fonte),
                    "-o",
                    str(biblioteca),
                ],
                check=True,
                capture_output=True,
            )

        backend = ctypes.CDLL(str(biblioteca))
        funcao = backend.advance_events
        funcao.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
        ]
        funcao.restype = ctypes.c_int
        return backend
    except (OSError, subprocess.SubprocessError):
        return None


class MultilayerMistura:
    """k-SEP com uma cor passiva e conservada por partícula."""

    def __init__(self, parametros: Parametros):
        parametros.validar()
        self.par = parametros
        self.N = parametros.N
        self.k = parametros.k
        self.rng = np.random.default_rng(parametros.seed)

        # -1 representa vazio; 0,...,k-1 são as cores das camadas iniciais.
        self.estado = np.full((self.k, self.N), -1, dtype=np.int16)
        quantidade = int(round(parametros.densidade_inicial * self.N))
        for camada in range(self.k):
            sitios = self.rng.choice(self.N, size=quantidade, replace=False)
            self.estado[camada, sitios] = camada

        self.massas_cores = np.bincount(
            self.estado[self.estado >= 0],
            minlength=self.k,
        ).astype(int)
        self.massa_total = int(self.massas_cores.sum())

        self.taxa_horizontal = 2.0 * self.k * self.N * self.N**2
        self.taxa_vertical = (
            self.N * self.k * (self.k - 1) * parametros.lambda_vertical
        )
        self.taxa_total = self.taxa_horizontal + self.taxa_vertical
        self.prob_vertical = self.taxa_vertical / self.taxa_total

        self.tempo = 0.0
        self.estatisticas = np.zeros(6, dtype=np.uint64)
        self._rng_c = ctypes.c_uint64(
            (parametros.seed ^ 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        )
        self._backend = _carregar_backend()
        self._usar_numba = self._backend is None and NUMBA_DISPONIVEL
        if self._backend is None and not self._usar_numba:
            warnings.warn(
                "Backend C e Numba indisponíveis. Usando o fallback Python, "
                "adequado apenas para N pequeno.",
                RuntimeWarning,
            )
        self.tempo_convergencia: Optional[float] = None

    def matriz_mistura(self) -> np.ndarray:
        """C[a,i] = número de partículas da cor a na camada atual i."""
        matriz = np.zeros((self.k, self.k), dtype=int)
        for camada in range(self.k):
            presentes = self.estado[camada]
            presentes = presentes[presentes >= 0]
            matriz[:, camada] = np.bincount(
                presentes,
                minlength=self.k,
            )
        return matriz

    def distancia_equilibrio(self) -> float:
        matriz = self.matriz_mistura().astype(float)
        proporcoes = matriz / self.massas_cores[:, None]
        uniforme = 1.0 / self.k
        tv_por_cor = 0.5 * np.abs(proporcoes - uniforme).sum(axis=1)
        return float(tv_por_cor.mean())

    def _avancar_python(
        self,
        numero_eventos: int,
        max_registros: int,
    ) -> list[EventoVertical]:
        eventos: list[EventoVertical] = []
        for _ in range(numero_eventos):
            if self.rng.random() < self.prob_vertical:
                self.estatisticas[2] += 1
                origem = int(self.rng.integers(self.k))
                destino = int(self.rng.integers(self.k - 1))
                if destino >= origem:
                    destino += 1
                x = int(self.rng.integers(self.N))
                if self.estado[origem, x] < 0:
                    self.estatisticas[4] += 1
                    continue
                inferior, superior = sorted((origem, destino))
                ocupado = self.estado[inferior : superior + 1, x] >= 0
                permitido = int(ocupado.sum()) == 1
                if permitido:
                    self.estado[destino, x] = self.estado[origem, x]
                    self.estado[origem, x] = -1
                    self.estatisticas[3] += 1
                else:
                    self.estatisticas[5] += 1
                if len(eventos) < max_registros:
                    eventos.append(EventoVertical(origem, destino, x, permitido))
            else:
                self.estatisticas[0] += 1
                camada = int(self.rng.integers(self.k))
                x = int(self.rng.integers(self.N))
                y = (x + (1 if self.rng.random() < 0.5 else -1)) % self.N
                if self.estado[camada, x] >= 0 and self.estado[camada, y] < 0:
                    self.estado[camada, y] = self.estado[camada, x]
                    self.estado[camada, x] = -1
                    self.estatisticas[1] += 1
        return eventos

    def avancar(self, delta_t: float, max_registros: int = 256) -> list[EventoVertical]:
        numero_eventos = int(self.rng.poisson(self.taxa_total * delta_t))
        if self._backend is None and not self._usar_numba:
            eventos = self._avancar_python(numero_eventos, max_registros)
        elif self._usar_numba:
            origens = np.empty(max_registros, dtype=np.int16)
            destinos = np.empty(max_registros, dtype=np.int16)
            xs = np.empty(max_registros, dtype=np.int32)
            sucessos = np.empty(max_registros, dtype=np.uint8)
            novo_rng, quantidade = advance_events_numba(
                self.estado,
                self.k,
                self.N,
                numero_eventos,
                self.prob_vertical,
                int(self._rng_c.value),
                self.estatisticas,
                origens,
                destinos,
                xs,
                sucessos,
                max_registros,
            )
            self._rng_c.value = int(novo_rng)
            eventos = [
                EventoVertical(
                    int(origens[i]),
                    int(destinos[i]),
                    int(xs[i]),
                    bool(sucessos[i]),
                )
                for i in range(quantidade)
            ]
        else:
            origens = np.empty(max_registros, dtype=np.int16)
            destinos = np.empty(max_registros, dtype=np.int16)
            xs = np.empty(max_registros, dtype=np.int32)
            sucessos = np.empty(max_registros, dtype=np.uint8)
            quantidade = self._backend.advance_events(
                self.estado.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                self.k,
                self.N,
                numero_eventos,
                self.prob_vertical,
                ctypes.byref(self._rng_c),
                self.estatisticas.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                origens.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                destinos.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                xs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                sucessos.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                max_registros,
            )
            eventos = [
                EventoVertical(
                    int(origens[i]),
                    int(destinos[i]),
                    int(xs[i]),
                    bool(sucessos[i]),
                )
                for i in range(quantidade)
            ]

        self.tempo += delta_t
        if int((self.estado >= 0).sum()) != self.massa_total:
            raise RuntimeError("A massa total deixou de ser conservada.")
        return eventos


@dataclass
class ParametrosGrandeEscala:
    """Parâmetros da aproximação rápida usada pela interface."""

    N: int = 572
    k: int = 4
    particulas_por_camada: int = 200
    lambda_vertical: float = 1.0
    tempo_final: float = 8.0
    epsilon_equilibrio: float = 0.10
    janela_equilibrio: float = 0.50
    seed: int = 20260719

    def validar(self) -> None:
        if not 1 <= self.k <= 15:
            raise ValueError("O número de camadas deve estar entre 1 e 15.")
        if not 100 <= self.particulas_por_camada <= 500:
            raise ValueError("Use entre 100 e 500 partículas por camada.")
        if self.N < self.particulas_por_camada:
            raise ValueError("N deve ser maior ou igual ao número de partículas.")
        if self.lambda_vertical < 0:
            raise ValueError("lambda_vertical deve ser não negativo.")
        if self.tempo_final <= 0:
            raise ValueError("tempo_final deve ser positivo.")
        if not 0 < self.epsilon_equilibrio < 1:
            raise ValueError("epsilon_equilibrio deve estar entre 0 e 1.")
        if self.janela_equilibrio < 0:
            raise ValueError("janela_equilibrio deve ser não negativa.")


class MultilayerMisturaGrandeEscala:
    """Dinâmica vertical média após equilíbrio horizontal condicional.

    Como a parte horizontal do gerador é multiplicada por N², em grande escala
    ela é muito mais rápida que os saltos verticais. Este motor usa essa
    separação de escalas: entre dois quadros, as posições horizontais são
    amostradas da medida uniforme, condicionadas à composição de cada camada.
    """

    def __init__(self, parametros: ParametrosGrandeEscala):
        parametros.validar()
        self.par = parametros
        self.N = parametros.N
        self.k = parametros.k
        self.rng = np.random.default_rng(parametros.seed)
        self.tempo = 0.0
        self.tempo_convergencia: Optional[float] = None
        self.estatisticas = np.zeros(6, dtype=np.uint64)

        # C[a,i] conta partículas da cor inicial a na camada atual i.
        self._contagens = np.zeros((self.k, self.k), dtype=np.int64)
        np.fill_diagonal(self._contagens, parametros.particulas_por_camada)
        self.massas_cores = self._contagens.sum(axis=1).astype(int)
        self.massa_total = int(self.massas_cores.sum())
        self.estado = np.full((self.k, self.N), -1, dtype=np.int16)
        self._reamostrar_posicoes()

    def matriz_mistura(self) -> np.ndarray:
        return self._contagens.copy()

    def distancia_equilibrio(self) -> float:
        proporcoes = self._contagens / self.massas_cores[:, None]
        tv_por_cor = 0.5 * np.abs(proporcoes - 1.0 / self.k).sum(axis=1)
        return float(tv_por_cor.mean())

    def _reamostrar_posicoes(self) -> None:
        self.estado.fill(-1)
        for camada in range(self.k):
            quantidades = self._contagens[:, camada]
            total = int(quantidades.sum())
            if total == 0:
                continue
            sitios = self.rng.choice(self.N, size=total, replace=False)
            rotulos = np.repeat(np.arange(self.k, dtype=np.int16), quantidades)
            self.rng.shuffle(rotulos)
            self.estado[camada, sitios] = rotulos

    def avancar(self, delta_t: float, max_registros: int = 256) -> list[EventoVertical]:
        eventos: list[EventoVertical] = []
        if self.k > 1 and self.par.lambda_vertical > 0:
            pares = [(i, j) for i in range(self.k) for j in range(self.k) if i != j]
            self.rng.shuffle(pares)

            for origem, destino in pares:
                propostas = int(
                    self.rng.poisson(self.N * self.par.lambda_vertical * delta_t)
                )
                if propostas == 0:
                    continue
                self.estatisticas[2] += propostas

                ocupacao_origem = int(self._contagens[:, origem].sum())
                presentes = int(self.rng.binomial(propostas, ocupacao_origem / self.N))
                self.estatisticas[4] += propostas - presentes

                inferior, superior = sorted((origem, destino))
                prob_caminho_livre = 1.0
                for camada in range(inferior, superior + 1):
                    if camada == origem:
                        continue
                    ocupacao = int(self._contagens[:, camada].sum())
                    prob_caminho_livre *= (self.N - ocupacao) / self.N

                sucessos = int(self.rng.binomial(presentes, prob_caminho_livre))
                capacidade_destino = self.N - int(self._contagens[:, destino].sum())
                sucessos = min(sucessos, ocupacao_origem, capacidade_destino)
                bloqueios = presentes - sucessos
                self.estatisticas[3] += sucessos
                self.estatisticas[5] += bloqueios

                if sucessos:
                    por_cor = self.rng.multivariate_hypergeometric(
                        self._contagens[:, origem], sucessos
                    )
                    self._contagens[:, origem] -= por_cor
                    self._contagens[:, destino] += por_cor

                vagas = max_registros - len(eventos)
                for _ in range(min(sucessos, vagas)):
                    eventos.append(
                        EventoVertical(origem, destino, int(self.rng.integers(self.N)), True)
                    )
                vagas = max_registros - len(eventos)
                for _ in range(min(bloqueios, vagas)):
                    eventos.append(
                        EventoVertical(origem, destino, int(self.rng.integers(self.N)), False)
                    )

        self._reamostrar_posicoes()
        self.tempo += delta_t
        if not np.array_equal(self._contagens.sum(axis=1), self.massas_cores):
            raise RuntimeError("A quantidade de partículas de alguma cor mudou.")
        if int(self._contagens.sum()) != self.massa_total:
            raise RuntimeError("A massa total deixou de ser conservada.")
        return eventos

def nome_camada(indice: int) -> str:
    return chr(ord("A") + indice) if indice < 26 else str(indice + 1)


def paleta(k: int) -> list[str]:
    # Cores neon com matizes bem separados sobre o fundo escuro.
    base = [
        "#00E5FF", "#FF9F0A", "#39FF14", "#FF2DAA", "#FFE600",
        "#9D4EDD", "#FF3131", "#00FFB3", "#4D7CFE", "#FF6B6B",
        "#B8FF00", "#D946EF", "#00BFFF", "#FF7A00", "#7CFF6B",
    ]
    if k <= len(base):
        return base[:k]
    cmap = plt.get_cmap("turbo")
    return [matplotlib.colors.to_hex(cmap(i / (k - 1))) for i in range(k)]


def produzir_video(
    simulacao: MultilayerMistura,
    caminho: Path,
    frames: int = 450,
    fps: int = 15,
    progresso: Optional[Callable[[int, int], None]] = None,
) -> tuple[list[float], list[float]]:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    par = simulacao.par
    cores = paleta(par.k)
    fundo = "#060914"
    delta_t = par.tempo_final / max(1, frames - 1)
    janela_frames = max(1, math.ceil(par.janela_equilibrio / delta_t))

    # 10.4 x 7.4 a 110 dpi produz 1144 x 814 px. As duas dimensões pares
    # mantêm compatibilidade com o formato yuv420p usado pelo H.264.
    fig = plt.figure(figsize=(10.4, 7.4), dpi=110, facecolor=fundo)
    grade = fig.add_gridspec(
        2,
        2,
        width_ratios=[4.8, 1.45],
        height_ratios=[4.8, 1.25],
        wspace=0.11,
        hspace=0.22 + 0.08 * max(0, math.ceil(par.k / 5) - 1),
    )
    ax = fig.add_subplot(grade[0, 0])
    ax_comp = fig.add_subplot(grade[0, 1])
    ax_dist = fig.add_subplot(grade[1, :])
    for eixo in (ax, ax_comp, ax_dist):
        eixo.set_facecolor(fundo)

    alturas = np.arange(par.k - 1, -1, -1, dtype=float)
    xs = np.arange(par.N)
    for altura in alturas:
        ax.plot([-0.5, par.N - 0.5], [altura, altura], color="#1f2b44", lw=0.7)

    gx, gc = np.meshgrid(xs, np.arange(par.k))
    gy = alturas[gc]
    ax.scatter(
        gx.ravel(),
        gy.ravel(),
        s=max(1.8, min(5.5, 2800 / par.N)),
        facecolors=fundo,
        edgecolors="#2e3b58",
        linewidths=0.35,
        zorder=1,
    )

    tamanho_particula = max(3.0, min(12.5, 5000 / par.N))
    pontos_cores = [
        ax.scatter(
            [], [], s=tamanho_particula, c=cor, edgecolors="#020617",
            linewidths=0.16,
            zorder=4,
        )
        for cor in cores
    ]
    # Branco para não confundir um salto aceito com a cor de uma partícula.
    linhas_ok = LineCollection([], colors="#F8FAFC", linewidths=1.4, alpha=0.78, zorder=5)
    linhas_block = LineCollection([], colors="#fb7185", linewidths=0.9, alpha=0.45, zorder=3)
    ax.add_collection(linhas_ok)
    ax.add_collection(linhas_block)
    cruzes = ax.scatter([], [], marker="x", s=20, c="#fb7185", linewidths=0.9, zorder=6)

    ax.set_xlim(-2, par.N + 1)
    ax.set_ylim(-0.7, par.k - 0.3)
    ax.set_xticks([0, par.N // 2, par.N - 1])
    ax.set_xticklabels(["0", str(par.N // 2), r"$N-1$"], color="#64748b")
    ax.set_yticks(alturas)
    ax.set_yticklabels(
        [f"camada {nome_camada(i)}" for i in range(par.k)],
        color="#e2e8f0",
        fontsize=9,
    )
    ax.tick_params(length=0)
    for borda in ax.spines.values():
        borda.set_visible(False)

    legenda = [
        Line2D([0], [0], marker="o", linestyle="", color=cores[i],
               label=f"cor inicial {nome_camada(i)}", markersize=5)
        for i in range(par.k)
    ]
    ax.legend(
        handles=legenda,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=min(par.k, 5),
        frameon=False,
        labelcolor="#cbd5e1",
        fontsize=8,
    )
    subtitulo = ax.set_title("", color="#a5b4fc", fontsize=10, pad=8)

    # Barras empilhadas: cada barra é uma camada atual e cada segmento é uma cor inicial.
    barras: list[list[matplotlib.patches.Rectangle]] = []
    for cor in range(par.k):
        conjunto = ax_comp.barh(
            np.arange(par.k),
            np.zeros(par.k),
            left=np.zeros(par.k),
            color=cores[cor],
            height=0.62,
            alpha=0.92,
        )
        barras.append(list(conjunto))
    # Sistemas com muitas camadas precisam de uma faixa superior maior para
    # as estatísticas não cobrirem as primeiras barras.
    topo_composicao = -3.2 if par.k >= 8 else -1.45
    ax_comp.set_ylim(par.k - 0.5, topo_composicao)
    ax_comp.set_xlim(0, par.N)
    ax_comp.set_yticks(np.arange(par.k))
    ax_comp.set_yticklabels(
        [f"{nome_camada(i)}" for i in range(par.k)],
        color="#e2e8f0",
    )
    ax_comp.set_xticks([0, par.N])
    ax_comp.set_xticklabels(["0", str(par.N)], color="#64748b", fontsize=8)
    ax_comp.tick_params(length=0)
    for borda in ax_comp.spines.values():
        borda.set_visible(False)
    painel = ax_comp.text(
        0,
        0.99,
        "",
        transform=ax_comp.transAxes,
        color="#cbd5e1",
        fontsize=8.2,
        va="top",
        linespacing=1.4,
    )
    ax_comp.text(
        0,
        0.86 if par.k >= 8 else 0.77,
        "composição por camada",
        transform=ax_comp.transAxes,
        color="#f1f5f9",
        fontsize=9.2,
    )

    linha_distancia, = ax_dist.plot([], [], color="#67e8f9", lw=1.8)
    ax_dist.axhline(
        par.epsilon_equilibrio,
        color="#facc15",
        lw=1.1,
        ls="--",
        label=rf"tolerância $\varepsilon={par.epsilon_equilibrio:g}$",
    )
    linha_convergencia = ax_dist.axvline(0, color="#4ade80", lw=1.2, alpha=0.0)
    ax_dist.set_xlim(0, par.tempo_final)
    ax_dist.set_ylim(0, max(0.15, 1 - 1 / par.k + 0.05))
    ax_dist.set_xlabel("tempo macroscópico", color="#94a3b8", fontsize=8.5)
    ax_dist.set_ylabel(r"$D(t)$", color="#94a3b8", fontsize=9, rotation=0, labelpad=12)
    ax_dist.tick_params(colors="#64748b", labelsize=8, length=2)
    for borda in ax_dist.spines.values():
        borda.set_color("#25314b")
    ax_dist.legend(loc="upper right", frameon=False, labelcolor="#cbd5e1", fontsize=8)
    texto_convergencia = ax_dist.text(
        0.01,
        0.08,
        "",
        transform=ax_dist.transAxes,
        color="#f8fafc",
        fontsize=9,
        va="bottom",
    )

    fig.suptitle(
        "Mistura de cores no multilayer hierarchical k-SEP",
        color="#f8fafc",
        fontsize=15,
        y=0.97,
    )

    escritor = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=3000,
        metadata={"title": "Mistura no multilayer k-SEP"},
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    tempos: list[float] = []
    distancias: list[float] = []
    with escritor.saving(fig, str(caminho), dpi=110):
        for frame in range(frames):
            eventos: list[EventoVertical] = []
            if frame > 0:
                eventos = simulacao.avancar(delta_t)

            matriz = simulacao.matriz_mistura()
            distancia = simulacao.distancia_equilibrio()
            tempos.append(simulacao.tempo)
            distancias.append(distancia)

            if (
                simulacao.tempo_convergencia is None
                and len(distancias) >= janela_frames
                and max(distancias[-janela_frames:]) <= par.epsilon_equilibrio
            ):
                simulacao.tempo_convergencia = tempos[-janela_frames]

            for cor, colecao in enumerate(pontos_cores):
                camadas, sitios = np.where(simulacao.estado == cor)
                offsets = np.column_stack([sitios, alturas[camadas]])
                colecao.set_offsets(offsets if len(offsets) else np.empty((0, 2)))

            sucessos = [evento for evento in eventos if evento.sucesso][-14:]
            bloqueios = [evento for evento in eventos if not evento.sucesso][-8:]
            linhas_ok.set_segments([
                [(e.x, alturas[e.origem]), (e.x, alturas[e.destino])]
                for e in sucessos
            ])
            linhas_block.set_segments([
                [(e.x, alturas[e.origem]), (e.x, alturas[e.destino])]
                for e in bloqueios
            ])
            centros = np.array([
                (e.x, 0.5 * (alturas[e.origem] + alturas[e.destino]))
                for e in bloqueios
            ])
            cruzes.set_offsets(centros if len(centros) else np.empty((0, 2)))

            for camada in range(par.k):
                esquerda = 0.0
                for cor in range(par.k):
                    retangulo = barras[cor][camada]
                    largura = matriz[cor, camada]
                    retangulo.set_x(esquerda)
                    retangulo.set_width(largura)
                    esquerda += largura

            linha_distancia.set_data(tempos, distancias)
            if simulacao.tempo_convergencia is None:
                texto_tau = rf"$\tau_{{mix}}>{simulacao.tempo:.2f}$"
            else:
                tau = simulacao.tempo_convergencia
                texto_tau = rf"$\tau_{{mix}}\approx {tau:.2f}$"
                linha_convergencia.set_xdata([tau, tau])
                linha_convergencia.set_alpha(0.8)
            texto_convergencia.set_text(
                texto_tau
                + rf"   ($D\leq {par.epsilon_equilibrio:g}$ por "
                + rf"$\Delta t={par.janela_equilibrio:g}$)"
            )

            subtitulo.set_text(
                rf"$N={par.N}$, $k={par.k}$, $\lambda={par.lambda_vertical:g}$"
                rf"   •   $t={simulacao.tempo:.2f}$"
            )
            if par.k >= 8:
                painel.set_text(
                    f"massa: {simulacao.massa_total}   D(t): {distancia:.3f}\n"
                    f"verticais: {int(simulacao.estatisticas[3])}   "
                    f"bloqueadas: {int(simulacao.estatisticas[5])}"
                )
            else:
                painel.set_text(
                    f"massa: {simulacao.massa_total}\n"
                    f"D(t): {distancia:.3f}\n"
                    f"verticais: {int(simulacao.estatisticas[3])}\n"
                    f"bloqueadas: {int(simulacao.estatisticas[5])}"
                )
            escritor.grab_frame(facecolor=fig.get_facecolor())
            if progresso is not None:
                progresso(frame + 1, frames)

    plt.close(fig)
    return tempos, distancias


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mistura de cores no multilayer k-SEP.")
    parser.add_argument("--saida", type=Path, default=Path("mistura_multilayer_30s.mp4"))
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--densidade", type=float, default=0.35)
    parser.add_argument("--lambda-vertical", type=float, default=1.0)
    parser.add_argument("--tempo-final", type=float, default=8.0)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--janela", type=float, default=0.50)
    parser.add_argument("--frames", type=int, default=450)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser


def main() -> None:
    args = construir_parser().parse_args()
    parametros = Parametros(
        N=args.N,
        k=args.k,
        densidade_inicial=args.densidade,
        lambda_vertical=args.lambda_vertical,
        tempo_final=args.tempo_final,
        epsilon_equilibrio=args.epsilon,
        janela_equilibrio=args.janela,
        seed=args.seed,
    )
    simulacao = MultilayerMistura(parametros)
    _, distancias = produzir_video(simulacao, args.saida, args.frames, args.fps)
    print(f"Vídeo salvo em: {args.saida.resolve()}")
    print(f"Massa total: {simulacao.massa_total}")
    print(f"Matriz final (cores x camadas):\n{simulacao.matriz_mistura()}")
    print(f"Distância final: {distancias[-1]:.6f}")
    print(f"Tempo de mistura: {simulacao.tempo_convergencia}")
    print(f"Estatísticas: {simulacao.estatisticas.tolist()}")


if __name__ == "__main__":
    main()
