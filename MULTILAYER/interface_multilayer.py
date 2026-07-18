#!/usr/bin/env python3
"""Interface local, executada no navegador, para o multilayer k-SEP."""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import streamlit as st

# Permite iniciar o aplicativo de qualquer diretório.
PASTA = Path(__file__).resolve().parent
if str(PASTA) not in sys.path:
    sys.path.insert(0, str(PASTA))

from simulacao_mistura_multilayer import (
    MultilayerMistura,
    Parametros,
    nome_camada,
    paleta,
    produzir_video,
)


matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
PASTA_VIDEOS = PASTA / "videos"


st.set_page_config(
    page_title="Multilayer k-SEP",
    page_icon="🔵",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetricValue"] {color: #00E5FF;}
    .nota {
        padding: .8rem 1rem; border: 1px solid #263554; border-radius: .7rem;
        background: #0a1020; color: #cbd5e1; margin: .5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Mistura de cores no multilayer hierarchical k-SEP")
st.write(
    "Cada partícula conserva a cor da camada em que começou. "
    "Cada tentativa microscópica de exclusão é simulada explicitamente."
)

with st.sidebar:
    st.header("Configuração")
    k = st.slider("Número de camadas", 1, 15, 4, 1)
    particulas = st.slider("Partículas por camada", 100, 500, 100, 10)
    densidade = st.slider(
        "Densidade inicial", 0.20, 0.75, 0.35, 0.05,
        help="Determina o número de sítios: N = partículas/densidade.",
    )
    N = int(math.ceil(particulas / densidade))
    st.info(
        f"**{N:,} sítios por camada**  \n"
        f"{N * k:,} sítios e {particulas * k:,} partículas no total"
    )

    st.subheader("Dinâmica")
    lambda_vertical = st.slider("Taxa vertical λ", 0.0, 3.0, 1.0, 0.1)
    tempo_final = st.slider("Tempo macroscópico final", 2.0, 30.0, 8.0, 0.5)
    epsilon = st.slider("Tolerância ε", 0.02, 0.25, 0.10, 0.01)
    janela = st.slider("Janela de permanência Δt", 0.10, 2.00, 0.50, 0.10)

    st.subheader("Vídeo")
    duracao_video = st.slider("Duração do vídeo (segundos)", 10, 60, 30, 5)
    seed = int(st.number_input("Semente aleatória", 0, 2**31 - 1, 20260719, 1))

cores = paleta(k)
legenda_cores = " ".join(
    f'<span style="color:{cores[i]}; font-weight:700">● {nome_camada(i)}</span>'
    for i in range(k)
)
st.markdown(legenda_cores, unsafe_allow_html=True)

taxa_horizontal = 2.0 * k * N**3
taxa_vertical = N * k * (k - 1) * lambda_vertical
eventos_esperados = (taxa_horizontal + taxa_vertical) * tempo_final
minutos_estimados = eventos_esperados / 40_000_000 / 60

if eventos_esperados < 1_000_000:
    texto_eventos = f"{eventos_esperados / 1_000:.1f} mil"
elif eventos_esperados < 1_000_000_000:
    texto_eventos = f"{eventos_esperados / 1_000_000:.1f} milhões"
else:
    texto_eventos = f"{eventos_esperados / 1_000_000_000:.1f} bilhões"

st.markdown(
    f"""
    <div class="nota">
    <b>SEP microscópico exato.</b> Em cada camada, uma partícula propõe
    x → x ± 1 e o salto é aceito somente quando o vizinho está vazio.
    Esta configuração exige aproximadamente <b>{texto_eventos} de propostas</b>.
    Estimativa inicial de processamento: <b>{minutos_estimados:.1f} min</b>,
    além da renderização. O tempo real depende do computador.
    </div>
    """,
    unsafe_allow_html=True,
)

confirmar_longa = True
if eventos_esperados >= 20_000_000_000:
    st.warning(
        "Esta escolha é muito grande e pode levar dezenas de minutos ou horas. "
        "Isso é consequência direta da taxa horizontal N²."
    )
    confirmar_longa = st.checkbox("Entendo o custo e quero executar mesmo assim")

if st.button(
    "Gerar simulação exata",
    type="primary",
    width="stretch",
    disabled=not confirmar_longa,
):
    parametros = Parametros(
        N=N,
        k=k,
        densidade_inicial=particulas / N,
        lambda_vertical=lambda_vertical,
        tempo_final=tempo_final,
        epsilon_equilibrio=epsilon,
        janela_equilibrio=janela,
        seed=seed,
    )
    simulacao = MultilayerMistura(parametros)
    PASTA_VIDEOS.mkdir(exist_ok=True)
    instante = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome = f"mistura_{k}camadas_{particulas}particulas_{instante}.mp4"
    saida = PASTA_VIDEOS / nome
    frames = duracao_video * 15
    barra = st.progress(0.0, text="Preparando o vídeo…")

    def atualizar_progresso(atual: int, total: int) -> None:
        fracao = atual / total
        barra.progress(fracao, text=f"Renderizando: {100 * fracao:.0f}%")

    try:
        _, distancias = produzir_video(
            simulacao,
            saida,
            frames=frames,
            fps=15,
            progresso=atualizar_progresso,
        )
        video = saida.read_bytes()
        st.session_state["video"] = video
        st.session_state["nome_video"] = nome
        st.session_state["tau"] = simulacao.tempo_convergencia
        st.session_state["distancia_final"] = distancias[-1]
        st.session_state["matriz"] = simulacao.matriz_mistura()
        st.session_state["k_resultado"] = k
        st.session_state["massa"] = simulacao.massa_total
        st.session_state["estatisticas"] = simulacao.estatisticas.copy()
        barra.progress(1.0, text="Vídeo concluído")
    except Exception as erro:
        barra.empty()
        st.exception(erro)

if "video" in st.session_state:
    st.divider()
    st.subheader("Resultado")
    tau = st.session_state["tau"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Massa total", f'{st.session_state["massa"]:,}')
    col2.metric("Distância final", f'{st.session_state["distancia_final"]:.4f}')
    col3.metric("Tempo observado de mistura", "não atingido" if tau is None else f"{tau:.3f}")

    stats = st.session_state["estatisticas"]
    st.caption(
        f"Propostas horizontais executadas: {int(stats[0]):,} · "
        f"saltos horizontais aceitos: {int(stats[1]):,} · "
        f"saltos verticais aceitos: {int(stats[3]):,}"
    )

    st.video(st.session_state["video"], format="video/mp4")
    st.download_button(
        "Baixar vídeo MP4",
        data=st.session_state["video"],
        file_name=st.session_state["nome_video"],
        mime="video/mp4",
        type="primary",
        width="stretch",
    )

    with st.expander("Ver matriz final: cores iniciais × camadas atuais"):
        matriz = st.session_state["matriz"]
        kr = st.session_state["k_resultado"]
        nomes = [nome_camada(i) for i in range(kr)]
        tabela = {"cor inicial": nomes}
        tabela.update({f"camada {nomes[j]}": matriz[:, j] for j in range(kr)})
        st.dataframe(tabela, width="stretch", hide_index=True)
