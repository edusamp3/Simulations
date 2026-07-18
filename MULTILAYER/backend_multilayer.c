#include <stdint.h>
#include <stddef.h>

/* Gerador xorshift64*: simples, rápido e suficiente para a simulação. */
static inline uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) {
        x = UINT64_C(0x9E3779B97F4A7C15);
    }
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static inline double rng_uniform(uint64_t *state) {
    return (rng_next(state) >> 11) * 0x1.0p-53;
}

static inline int rng_bounded(uint64_t *state, int limite) {
    return (int)(rng_next(state) % (uint64_t)limite);
}

/*
stats:
0 propostas horizontais
1 saltos horizontais
2 propostas verticais
3 saltos verticais
4 verticais com origem vazia
5 verticais bloqueadas pela hierarquia
*/
int advance_events(
    int16_t *estado,
    int k,
    int N,
    uint64_t numero_eventos,
    double prob_vertical,
    uint64_t *rng_state,
    uint64_t *stats,
    int16_t *eventos_origem,
    int16_t *eventos_destino,
    int32_t *eventos_x,
    uint8_t *eventos_sucesso,
    int max_eventos_registrados
) {
    int registrados = 0;

    for (uint64_t evento = 0; evento < numero_eventos; ++evento) {
        if (rng_uniform(rng_state) < prob_vertical) {
            stats[2] += 1;
            int origem = rng_bounded(rng_state, k);
            int destino = rng_bounded(rng_state, k - 1);
            if (destino >= origem) {
                destino += 1;
            }
            int x = rng_bounded(rng_state, N);
            int indice_origem = origem * N + x;

            if (estado[indice_origem] < 0) {
                stats[4] += 1;
                continue;
            }

            int inferior = origem < destino ? origem : destino;
            int superior = origem > destino ? origem : destino;
            int permitido = 1;
            for (int camada = inferior; camada <= superior; ++camada) {
                if (camada != origem && estado[camada * N + x] >= 0) {
                    permitido = 0;
                    break;
                }
            }

            if (permitido) {
                int indice_destino = destino * N + x;
                estado[indice_destino] = estado[indice_origem];
                estado[indice_origem] = -1;
                stats[3] += 1;
            } else {
                stats[5] += 1;
            }

            if (registrados < max_eventos_registrados) {
                eventos_origem[registrados] = (int16_t)origem;
                eventos_destino[registrados] = (int16_t)destino;
                eventos_x[registrados] = (int32_t)x;
                eventos_sucesso[registrados] = (uint8_t)permitido;
                registrados += 1;
            }
        } else {
            stats[0] += 1;
            int camada = rng_bounded(rng_state, k);
            int x = rng_bounded(rng_state, N);
            int direcao = (rng_next(rng_state) & 1) ? 1 : -1;
            int y = x + direcao;
            if (y < 0) {
                y += N;
            } else if (y >= N) {
                y -= N;
            }

            int origem = camada * N + x;
            int destino = camada * N + y;
            if (estado[origem] >= 0 && estado[destino] < 0) {
                estado[destino] = estado[origem];
                estado[origem] = -1;
                stats[1] += 1;
            }
        }
    }

    return registrados;
}
