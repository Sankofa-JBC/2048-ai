# 2048 AI Training Project

Projeto em Python para implementar o jogo 2048, criar agentes de IA e treinar
algoritmos de aprendizagem posteriormente no Google Colab.

## Etapa atual

- Motor do 2048 em Python puro.
- Interface leve no terminal.
- Testes com `unittest`, sem dependencias externas.

## Rodar o jogo

```bash
python main.py
```

Controles:

- `W`: cima
- `A`: esquerda
- `S`: baixo
- `D`: direita
- `Q`: sair

## Rodar os testes

```bash
python -m unittest discover -s tests
```

## Rodar o agente aleatorio

Uma partida:

```bash
python run_random_agent.py
```

Varias partidas com estatisticas:

```bash
python run_random_agent.py --games 10 --seed 42
```

Mostrar cada movimento no terminal:

```bash
python run_random_agent.py --show --delay 0.15
```

## API base para agente

```python
from game2048 import ACTION_LEFT, Game2048

game = Game2048(seed=42)
state = game.reset()
next_state, reward, done, info = game.step(ACTION_LEFT)
```

## Git

Este projeto deve ser versionado como um repositorio proprio dentro desta pasta.
Isso evita misturar os arquivos do jogo com outros arquivos do usuario.
