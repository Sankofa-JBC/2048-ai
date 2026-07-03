# 2048 AI Training Project

Projeto em Python para implementar o jogo 2048, criar agentes de IA e treinar
algoritmos de aprendizagem posteriormente no Google Colab.

## Etapa atual

- Motor do 2048 em Python puro.
- Interface leve no terminal.
- Interface comum para agentes.
- Agentes baseline: aleatorio e heuristico.
- Avaliador de agentes com metricas e exportacao.
- Testes com `unittest`, sem dependencias externas.

## Instalar em modo desenvolvimento

Use este comando localmente ou no Google Colab depois de clonar o repositorio:

```bash
pip install -e .
```

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

## Avaliar agentes

Avaliar o agente aleatorio:

```bash
python evaluate_agents.py --agent random --games 100 --seed 42
```

Avaliar o agente heuristico:

```bash
python evaluate_agents.py --agent heuristic --games 100 --seed 42
```

Exportar metricas em JSON:

```bash
python evaluate_agents.py --agent random --games 100 --seed 42 --output results/random.json
```

Exportar metricas por partida em CSV:

```bash
python evaluate_agents.py --agent heuristic --games 100 --seed 42 --output results/heuristic.csv
```

## Google Colab

Notebook inicial de validacao:

```text
notebooks/2048_ai_colab_validacao.ipynb
```

Use esse notebook para validar que o Colab consegue:

- clonar o repositorio;
- instalar o projeto com `pip install -e .`;
- importar `game2048`;
- rodar os testes;
- avaliar os agentes `random` e `heuristic`;
- exportar metricas em JSON.

Notebook inicial de treino DQN:

```text
notebooks/2048_ai_dqn_training.ipynb
```

Esse notebook faz um treino curto de validacao, salva um checkpoint em
`models/dqn_smoke.pt` e avalia o agente treinado. O primeiro treino nao tem
objetivo de jogar bem ainda; ele valida que o pipeline de aprendizagem funciona.

## Treinar DQN

No Colab, depois de instalar o projeto com dependencias de aprendizagem:

```bash
pip install -e ".[learning]"
```

Treino curto:

```bash
python train_dqn.py --episodes 50 --batch-size 32 --min-replay-size 100 --output models/dqn_smoke.pt
```

Avaliar checkpoint:

```bash
python evaluate_dqn.py models/dqn_smoke.pt --games 20 --seed 42
```

## API base para agente

```python
from game2048 import ACTION_LEFT, Game2048, RandomAgent

game = Game2048(seed=42)
state = game.reset()
next_state, reward, done, info = game.step(ACTION_LEFT)

agent = RandomAgent(seed=42)
action = agent.choose_action(game.board, game.available_actions())
```

Todo agente deve expor este metodo:

```python
choose_action(board, available_actions) -> int
```

Isso permite comparar agentes diferentes usando o mesmo avaliador.

## Git

Este projeto deve ser versionado como um repositorio proprio dentro desta pasta.
Isso evita misturar os arquivos do jogo com outros arquivos do usuario.
