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

Depois da instalacao, voce pode rodar os modulos `game2048.cli.*` de qualquer
pasta com `python -m`, inclusive no Colab. Esse e o modo mais robusto.
Os comandos `game2048-compare-agents`, `game2048-train-dqn`,
`game2048-evaluate-agents`, `game2048-evaluate-dqn` e
`game2048-run-experiment` tambem sao instalados como atalhos.

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

Ou, de qualquer pasta depois de `pip install -e .`:

```bash
python -m game2048.cli.evaluate_agents --agent random --games 100 --seed 42
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

Fluxo mais direto, em uma unica celula:

```python
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/Sankofa-JBC/2048-ai.git"
PROJECT_DIR = Path("/content/2048-ai")

if PROJECT_DIR.exists():
    subprocess.run(["git", "-C", str(PROJECT_DIR), "pull"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, str(PROJECT_DIR)], check=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".[learning]"],
    cwd=PROJECT_DIR,
    check=True,
)

subprocess.run(
    [
        sys.executable,
        "-m",
        "game2048.cli.run_experiment",
        "--project-dir",
        str(PROJECT_DIR),
        "--episodes",
        "1000",
        "--games",
        "200",
    ],
    check=True,
)
```

Esse fluxo ja:

- clona ou atualiza o repositorio;
- instala o projeto com PyTorch;
- treina ou retoma o DQN;
- gera a tabela final com `random`, `heuristic` e `dqn`;
- salva `results/final_report.json` e `results/final_comparison.csv`;
- tenta gerar `results/comparison_scores.png` e `results/training_progress.png`.

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
No final, o notebook tambem mostra a tabela comparativa dos agentes e o
veredito salvo em `results/final_report.json`.

Notebook de fluxo unico:

```text
notebooks/2048_ai_colab_fluxo_unico.ipynb
```

Esse notebook faz o caminho mais curto em uma unica celula e usa retomada
automatica quando `models/dqn_2048_best.pt` ja existir.

## Treinar DQN

No Colab, depois de instalar o projeto com dependencias de aprendizagem:

```bash
pip install -e ".[learning]"
```

Treino inicial:

```bash
python train_dqn.py --episodes 300 --output models/dqn_2048.pt --best-output models/dqn_2048_best.pt
```

Fluxo mais simples no Colab: treinar ou retomar e ja gerar a comparacao final
em um unico comando:

```bash
python -m game2048.cli.run_experiment --project-dir /content/2048-ai --episodes 1000 --games 200
```

Se `models/dqn_2048_best.pt` ja existir, esse comando retoma automaticamente a
partir dele.
Na pratica, repetir o mesmo comando soma mais episodios ao treino anterior.

Se quiser continuar do melhor checkpoint atual:

```bash
python -m game2048.cli.run_experiment --project-dir /content/2048-ai --episodes 1000 --games 200 --resume-from models/dqn_2048_best.pt
```

Se quiser forcar um treino novo do zero:

```bash
python -m game2048.cli.run_experiment --project-dir /content/2048-ai --episodes 1000 --games 200 --force-fresh
```

Para apenas comparar o modelo atual sem novo treino:

```bash
python -m game2048.cli.run_experiment --project-dir /content/2048-ai --games 200 --skip-train
```

Continuar um treino a partir de um checkpoint ja salvo:

```bash
python train_dqn.py --episodes 1000 --resume-from models/dqn_2048_best.pt --output models/dqn_2048.pt --best-output models/dqn_2048_best.pt
```

Sem `--resume-from`, o treino cria uma rede nova do zero. Com `--resume-from`,
ele reutiliza os pesos do checkpoint anterior e continua a numeracao dos
episodios.

No Colab, depois de `pip install -e .`, voce tambem pode rodar esse treino de
qualquer pasta com:

```bash
python -m game2048.cli.train_dqn --episodes 1000 --resume-from /content/2048-ai/models/dqn_2048_best.pt --output /content/2048-ai/models/dqn_2048.pt --best-output /content/2048-ai/models/dqn_2048_best.pt
```

Avaliar checkpoint:

```bash
python evaluate_dqn.py models/dqn_2048_best.pt --games 100 --seed 42
```

Ou, de qualquer pasta depois de `pip install -e .`:

```bash
python -m game2048.cli.evaluate_dqn /content/2048-ai/models/dqn_2048_best.pt --games 100 --seed 42
```

Comparar o DQN treinado com os agentes aleatorio e heuristico:

```bash
python compare_agents.py --games 100 --seed 42 --dqn-model models/dqn_2048_best.pt
```

No Colab, depois de `pip install -e .`, o comando abaixo funciona sem precisar
entrar primeiro em `/content/2048-ai`:

```bash
python -m game2048.cli.compare_agents --games 100 --seed 42 --dqn-model /content/2048-ai/models/dqn_2048_best.pt --training-metrics /content/2048-ai/results/dqn_training_metrics.json --output-json /content/2048-ai/results/final_report.json --output-csv /content/2048-ai/results/final_comparison.csv
```

Esse comando imprime uma tabela com media, maior score e menor score de cada
agente. Ele tambem salva os arquivos:

- `results/final_report.json`: relatorio completo para analise;
- `results/final_comparison.csv`: tabela agregada para planilha ou Colab.

Ver o modelo treinado jogando no terminal:

```bash
python play_trained_agent.py models/dqn_2048_best.pt --delay 0.08
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
