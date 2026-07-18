# Multilayer k-SEP microscópico exato — pacote universal

O mesmo código funciona em macOS, Windows e Linux. Escolha apenas o
inicializador correspondente ao sistema operacional.

## macOS

Execute `INICIAR_EXATO_NO_MAC.command`. Se o Gatekeeper bloquear:

1. clique em **Done**;
2. abra **System Settings > Privacy & Security**;
3. clique em **Open Anyway** e confirme.

Também é possível executar pelo Terminal:

```bash
bash INICIAR_EXATO_NO_MAC.command
```

## Windows

Execute `INICIAR_EXATO_NO_WINDOWS.bat`. Se o SmartScreen aparecer, escolha
**Mais informações > Executar assim mesmo**. Ao instalar Python pelo site
<https://www.python.org/downloads/windows/>, marque **Add Python to PATH**.

## Linux

Abra o terminal nesta pasta e execute:

```bash
bash INICIAR_EXATO_NO_LINUX.sh
```

Se a criação do ambiente virtual falhar no Ubuntu ou Debian:

```bash
sudo apt install python3-venv
```

## Requisitos e primeira execução

É necessário Python 3.10 ou superior e conexão com a internet na primeira
execução. Cada inicializador cria `.venv`, instala as dependências e abre a
interface no navegador. Não é necessário instalar FFmpeg separadamente.

## Dinâmica simulada

Esta interface usa o motor microscópico exato. Cada proposta horizontal
`x -> x+1` ou `x -> x-1` é executada, e o salto só ocorre quando o sítio
vizinho está vazio. Não há reamostragem pela medida de equilíbrio.

A pasta correta contém `backend_numba.py`, e o botão da interface se chama
**Gerar simulação exata**. Como a taxa horizontal é acelerada por N², o custo
esperado cresce como `k*N^3`; escolhas extremas podem levar horas.

