"""Backend portátil do simulador microscópico exato.

O Numba compila esta função na primeira execução e funciona com as mesmas
regras do backend em C, sem exigir Xcode no macOS ou Visual Studio no Windows.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    NUMBA_DISPONIVEL = True
except ImportError:  # pragma: no cover - usado somente no fallback mínimo
    njit = None
    NUMBA_DISPONIVEL = False


if NUMBA_DISPONIVEL:

    @njit(cache=True, inline="always")
    def _rng_next(estado):
        x = np.uint64(estado)
        if x == np.uint64(0):
            x = np.uint64(0x9E3779B97F4A7C15)
        x ^= x >> np.uint64(12)
        x ^= x << np.uint64(25)
        x ^= x >> np.uint64(27)
        return x, x * np.uint64(2685821657736338717)


    @njit(cache=True, inline="always")
    def _rng_uniform(estado):
        estado, valor = _rng_next(estado)
        uniforme = float(valor >> np.uint64(11)) * 1.1102230246251565e-16
        return estado, uniforme


    @njit(cache=True, inline="always")
    def _rng_bounded(estado, limite):
        estado, valor = _rng_next(estado)
        indice = np.int64(valor % np.uint64(limite))
        return estado, indice


    @njit(cache=True)
    def advance_events_numba(
        estado,
        k,
        N,
        numero_eventos,
        prob_vertical,
        rng_state,
        stats,
        eventos_origem,
        eventos_destino,
        eventos_x,
        eventos_sucesso,
        max_eventos_registrados,
    ):
        """Executa exatamente os relógios uniformizados do gerador."""
        rng = np.uint64(rng_state)
        registrados = 0

        for _ in range(numero_eventos):
            rng, uniforme = _rng_uniform(rng)
            if uniforme < prob_vertical:
                stats[2] += np.uint64(1)
                rng, origem = _rng_bounded(rng, k)
                rng, destino = _rng_bounded(rng, k - 1)
                if destino >= origem:
                    destino += 1
                rng, x = _rng_bounded(rng, N)

                if estado[origem, x] < 0:
                    stats[4] += np.uint64(1)
                    continue

                inferior = min(origem, destino)
                superior = max(origem, destino)
                permitido = True
                for camada in range(inferior, superior + 1):
                    if camada != origem and estado[camada, x] >= 0:
                        permitido = False
                        break

                if permitido:
                    estado[destino, x] = estado[origem, x]
                    estado[origem, x] = -1
                    stats[3] += np.uint64(1)
                else:
                    stats[5] += np.uint64(1)

                if registrados < max_eventos_registrados:
                    eventos_origem[registrados] = origem
                    eventos_destino[registrados] = destino
                    eventos_x[registrados] = x
                    eventos_sucesso[registrados] = 1 if permitido else 0
                    registrados += 1
            else:
                stats[0] += np.uint64(1)
                rng, camada = _rng_bounded(rng, k)
                rng, x = _rng_bounded(rng, N)
                rng, sorteio = _rng_next(rng)
                direcao = 1 if (sorteio & np.uint64(1)) else -1
                y = x + direcao
                if y < 0:
                    y += N
                elif y >= N:
                    y -= N

                if estado[camada, x] >= 0 and estado[camada, y] < 0:
                    estado[camada, y] = estado[camada, x]
                    estado[camada, x] = -1
                    stats[1] += np.uint64(1)

        return rng, registrados

else:

    def advance_events_numba(*_args, **_kwargs):
        raise RuntimeError("Numba não está instalado.")
