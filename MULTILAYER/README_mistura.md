# Mistura de cores no multilayer k-SEP

Nesta versão, cada partícula recebe a cor da camada em que começou e conserva
essa cor durante todos os saltos horizontais e verticais. Assim, as cores são
rótulos passivos que permitem observar a mistura entre as camadas.

No caso padrão com quatro camadas, a paleta fixa é: A em ciano, B em laranja,
C em verde-limão e D em magenta. Os saltos verticais aceitos aparecem em
branco para não serem confundidos com as partículas.

Para cada cor inicial `a` e camada atual `i`, definimos

\[
C_{a,i}(t)=\#\{\text{partículas da cor }a\text{ na camada }i\text{ em }t\}.
\]

Se `M_a` é o número total de partículas da cor `a`, a medida de equilíbrio tem
valor esperado

\[
\frac{C_{a,i}}{M_a}=\frac1k.
\]

A distância mostrada no vídeo é a média das distâncias de variação total:

\[
D(t)=\frac1k\sum_{a=1}^k\frac12
\sum_{i=1}^k\left|\frac{C_{a,i}(t)}{M_a}-\frac1k\right|.
\]

Como uma cadeia finita continua flutuando em equilíbrio, o código define o
tempo observado de mistura como o primeiro instante em que

\[
D(t)\leq\varepsilon
\]

durante uma janela contínua de comprimento `janela_equilibrio`. Os valores
padrão são `epsilon=0.10` e `janela=0.50`.

## Sistema padrão

- `N=200` sítios por camada;
- `k=4` camadas;
- densidade `0.35` em cada camada;
- 70 partículas de cada cor e massa total 280;
- taxas verticais constantes e simétricas `lambda=1`;
- tempo macroscópico final `T=8`;
- vídeo de 30 segundos.

## Desempenho

O gerador horizontal possui taxa total de ordem \(N^3\). Para manter a cadeia
microscópica original com `N=200`, o script compila automaticamente o pequeno
arquivo `backend_multilayer.c` e o chama por `ctypes`. Se não houver compilador
C, `backend_numba.py` compila o mesmo laço automaticamente. O fallback Python
puro é usado somente quando ambos estão indisponíveis e é recomendado apenas
para valores pequenos de `N`.

## Executando

```bash
python simulacao_mistura_multilayer.py --saida mistura_multilayer_30s.mp4
```

Os parâmetros podem ser alterados:

```bash
python simulacao_mistura_multilayer.py \
    --N 300 \
    --k 4 \
    --densidade 0.35 \
    --lambda-vertical 1.0 \
    --tempo-final 10 \
    --epsilon 0.08 \
    --janela 0.50
```
