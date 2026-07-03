"""Loop de treinamento DQN para o jogo 2048."""

from __future__ import annotations

import random
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import torch
from torch import nn

from game2048.constants import ACTIONS
from game2048.game import Game2048
from game2048.learning.dqn_model import DQNModel
from game2048.learning.replay_memory import ReplayMemory, Transition
from game2048.learning.state import (
    available_action_mask,
    board_to_features,
)


@dataclass(frozen=True)
class DQNTrainingConfig:
    """Configuração de treino com valores seguros para validação inicial."""

    episodes: int = 200
    seed: int = 42
    gamma: float = 0.99
    learning_rate: float = 0.001
    batch_size: int = 64
    memory_capacity: int = 20_000
    min_replay_size: int = 500
    target_update_interval: int = 20
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 800
    hidden_size: int = 128
    reward_scale: float = 1.0 / 2048.0
    max_steps_per_episode: int = 5_000
    save_path: str = "models/dqn_2048.pt"
    best_save_path: str | None = "models/dqn_2048_best.pt"
    metrics_path: str | None = "results/dqn_training_metrics.json"
    checkpoint_dir: str | None = "models/checkpoints"
    checkpoint_interval: int = 100
    rolling_window: int = 50
    double_dqn: bool = True
    device: str | None = None


@dataclass(frozen=True)
class TrainingEpisodeStats:
    """Métricas de um episódio de treino."""

    episode: int
    score: int
    steps: int
    max_tile: int
    epsilon: float
    average_loss: float | None
    rolling_average_score: float


@dataclass(frozen=True)
class TrainingResult:
    """Resultado final do treinamento."""

    config: DQNTrainingConfig
    episodes: tuple[TrainingEpisodeStats, ...]
    save_path: str
    best_save_path: str | None
    best_rolling_average_score: float


def train_dqn(config: DQNTrainingConfig) -> TrainingResult:
    """Treina um DQN básico e salva o checkpoint final."""
    _validate_config(config)
    _seed_everything(config.seed)

    device = _resolve_device(config.device)
    memory = ReplayMemory(config.memory_capacity, seed=config.seed)
    policy_net = DQNModel(hidden_size=config.hidden_size).to(device)
    target_net = DQNModel(hidden_size=config.hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    episode_stats: list[TrainingEpisodeStats] = []
    best_rolling_average_score = float("-inf")

    for episode in range(1, config.episodes + 1):
        epsilon = _epsilon_for_episode(config, episode)
        stats = _train_episode(
            episode=episode,
            epsilon=epsilon,
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            loss_fn=loss_fn,
            memory=memory,
            config=config,
            device=device,
        )
        episode_stats.append(_with_rolling_average(stats, episode_stats, config))

        if episode % config.target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        latest_stats = episode_stats[-1]
        if latest_stats.rolling_average_score > best_rolling_average_score:
            best_rolling_average_score = latest_stats.rolling_average_score
            if config.best_save_path is not None:
                save_checkpoint(
                    path=config.best_save_path,
                    model=policy_net,
                    config=config,
                    episode_stats=tuple(episode_stats),
                )

        if _should_save_periodic_checkpoint(config, episode):
            checkpoint_path = Path(config.checkpoint_dir or "models") / f"dqn_episode_{episode}.pt"
            save_checkpoint(
                path=checkpoint_path,
                model=policy_net,
                config=config,
                episode_stats=tuple(episode_stats),
            )

        _print_progress(episode_stats)

    save_checkpoint(
        path=config.save_path,
        model=policy_net,
        config=config,
        episode_stats=tuple(episode_stats),
    )
    if config.metrics_path is not None:
        save_training_metrics(
            path=config.metrics_path,
            result=TrainingResult(
                config=config,
                episodes=tuple(episode_stats),
                save_path=config.save_path,
                best_save_path=config.best_save_path,
                best_rolling_average_score=best_rolling_average_score,
            ),
        )
    return TrainingResult(
        config=config,
        episodes=tuple(episode_stats),
        save_path=config.save_path,
        best_save_path=config.best_save_path,
        best_rolling_average_score=best_rolling_average_score,
    )


def save_checkpoint(
    path: str | Path,
    model: DQNModel,
    config: DQNTrainingConfig,
    episode_stats: tuple[TrainingEpisodeStats, ...],
) -> None:
    """Salva pesos do modelo e metadados mínimos de treino."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "hidden_size": config.hidden_size,
            },
            "training_config": asdict(config),
            "episodes": [asdict(stats) for stats in episode_stats],
        },
        output_path,
    )


def _train_episode(
    episode: int,
    epsilon: float,
    policy_net: DQNModel,
    target_net: DQNModel,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    memory: ReplayMemory,
    config: DQNTrainingConfig,
    device: torch.device,
) -> TrainingEpisodeStats:
    """Executa um episódio e atualiza a rede usando replay memory."""
    game = Game2048(seed=config.seed + episode)
    rng = random.Random(config.seed + episode)
    losses: list[float] = []

    while not game.done and game.steps < config.max_steps_per_episode:
        available_actions = game.available_actions()
        if not available_actions:
            break

        state = board_to_features(game.board)
        action = _choose_training_action(
            board_features=state,
            available_actions=available_actions,
            epsilon=epsilon,
            policy_net=policy_net,
            device=device,
            rng=rng,
        )
        next_board, reward, done, _ = game.step(action)
        next_available_actions = tuple(game.available_actions()) if not done else ()
        memory.push(
            Transition(
                state=state,
                action=action,
                reward=reward * config.reward_scale,
                next_state=board_to_features(next_board),
                done=done,
                next_available_actions=next_available_actions,
            )
        )

        loss = _optimize_model(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            loss_fn=loss_fn,
            memory=memory,
            config=config,
            device=device,
        )
        if loss is not None:
            losses.append(loss)

    average_loss = mean(losses) if losses else None
    return TrainingEpisodeStats(
        episode=episode,
        score=game.score,
        steps=game.steps,
        max_tile=max(max(row) for row in game.board),
        epsilon=epsilon,
        average_loss=average_loss,
        rolling_average_score=float(game.score),
    )


def save_training_metrics(path: str | Path, result: TrainingResult) -> None:
    """Salva o historico de treino em JSON para analise no Colab."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "config": asdict(result.config),
                "save_path": result.save_path,
                "best_save_path": result.best_save_path,
                "best_rolling_average_score": result.best_rolling_average_score,
                "episodes": [asdict(stats) for stats in result.episodes],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _choose_training_action(
    board_features: tuple[float, ...],
    available_actions: list[int],
    epsilon: float,
    policy_net: DQNModel,
    device: torch.device,
    rng: random.Random,
) -> int:
    """Escolhe ação com exploração epsilon-greedy e máscara de ações válidas."""
    if rng.random() < epsilon:
        return rng.choice(available_actions)

    with torch.no_grad():
        state_tensor = torch.tensor([board_features], dtype=torch.float32, device=device)
        q_values = policy_net(state_tensor)[0]

    masked_q_values = torch.full_like(q_values, -1_000_000_000.0)
    for action in available_actions:
        masked_q_values[action] = q_values[action]
    return int(torch.argmax(masked_q_values).item())


def _optimize_model(
    policy_net: DQNModel,
    target_net: DQNModel,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    memory: ReplayMemory,
    config: DQNTrainingConfig,
    device: torch.device,
) -> float | None:
    """Executa um passo de otimização se já houver memória suficiente."""
    if len(memory) < max(config.batch_size, config.min_replay_size):
        return None

    transitions = memory.sample(config.batch_size)
    states = torch.tensor(
        [transition.state for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    actions = torch.tensor(
        [[transition.action] for transition in transitions],
        dtype=torch.long,
        device=device,
    )
    rewards = torch.tensor(
        [transition.reward for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    next_states = torch.tensor(
        [transition.next_state for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    dones = torch.tensor(
        [transition.done for transition in transitions],
        dtype=torch.bool,
        device=device,
    )
    next_action_masks = torch.tensor(
        [
            available_action_mask(transition.next_available_actions)
            for transition in transitions
        ],
        dtype=torch.bool,
        device=device,
    )

    current_q_values = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        has_next_action = next_action_masks.any(dim=1) & ~dones
        max_next_q_values = torch.zeros(config.batch_size, dtype=torch.float32, device=device)

        if config.double_dqn:
            policy_next_q_values = policy_net(next_states)
            masked_policy_next_q_values = policy_next_q_values.masked_fill(
                ~next_action_masks,
                -1_000_000_000.0,
            )
            best_next_actions = masked_policy_next_q_values.argmax(dim=1, keepdim=True)
            target_next_q_values = target_net(next_states)
            selected_target_q_values = target_next_q_values.gather(
                1,
                best_next_actions,
            ).squeeze(1)
            max_next_q_values[has_next_action] = selected_target_q_values[has_next_action]
        else:
            next_q_values = target_net(next_states)
            masked_next_q_values = next_q_values.masked_fill(
                ~next_action_masks,
                -1_000_000_000.0,
            )
            max_next_q_values[has_next_action] = (
                masked_next_q_values[has_next_action].max(dim=1).values
            )

        expected_q_values = rewards + config.gamma * max_next_q_values

    loss = loss_fn(current_q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def _epsilon_for_episode(config: DQNTrainingConfig, episode: int) -> float:
    """Calcula epsilon com decaimento linear."""
    if config.epsilon_decay_episodes <= 0:
        return config.epsilon_end

    progress = min(1.0, (episode - 1) / config.epsilon_decay_episodes)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def _resolve_device(configured_device: str | None) -> torch.device:
    """Escolhe GPU quando disponível, caso nenhum device seja informado."""
    if configured_device is not None:
        return torch.device(configured_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    """Define seeds para reduzir variação entre execuções."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config(config: DQNTrainingConfig) -> None:
    """Falha cedo para configurações inválidas de treino."""
    if config.episodes <= 0:
        raise ValueError("episodes deve ser maior que zero.")
    if config.batch_size <= 0:
        raise ValueError("batch_size deve ser maior que zero.")
    if config.min_replay_size <= 0:
        raise ValueError("min_replay_size deve ser maior que zero.")
    if config.memory_capacity < config.batch_size:
        raise ValueError("memory_capacity deve ser maior ou igual a batch_size.")
    if config.gamma < 0.0 or config.gamma > 1.0:
        raise ValueError("gamma deve ficar entre 0.0 e 1.0.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate deve ser maior que zero.")
    if config.reward_scale <= 0.0:
        raise ValueError("reward_scale deve ser maior que zero.")
    if config.target_update_interval <= 0:
        raise ValueError("target_update_interval deve ser maior que zero.")
    if config.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval nao pode ser negativo.")
    if config.rolling_window <= 0:
        raise ValueError("rolling_window deve ser maior que zero.")


def _with_rolling_average(
    stats: TrainingEpisodeStats,
    previous_stats: list[TrainingEpisodeStats],
    config: DQNTrainingConfig,
) -> TrainingEpisodeStats:
    """Atualiza a metrica de media movel do score."""
    previous_window_size = max(0, config.rolling_window - 1)
    scores = (
        [item.score for item in previous_stats[-previous_window_size:]]
        if previous_window_size
        else []
    )
    scores.append(stats.score)
    return TrainingEpisodeStats(
        episode=stats.episode,
        score=stats.score,
        steps=stats.steps,
        max_tile=stats.max_tile,
        epsilon=stats.epsilon,
        average_loss=stats.average_loss,
        rolling_average_score=mean(scores),
    )


def _should_save_periodic_checkpoint(
    config: DQNTrainingConfig,
    episode: int,
) -> bool:
    """Indica se um checkpoint intermediario deve ser salvo."""
    return (
        config.checkpoint_dir is not None
        and config.checkpoint_interval > 0
        and episode % config.checkpoint_interval == 0
        and episode != config.episodes
    )


def _print_progress(episode_stats: list[TrainingEpisodeStats]) -> None:
    """Imprime progresso compacto a cada 10 episódios e no episódio 1."""
    latest = episode_stats[-1]
    if latest.episode != 1 and latest.episode % 10 != 0:
        return

    loss_text = "n/a" if latest.average_loss is None else f"{latest.average_loss:.4f}"
    print(
        f"episodio={latest.episode} "
        f"score={latest.score} "
        f"media_movel={latest.rolling_average_score:.1f} "
        f"maior_bloco={latest.max_tile} "
        f"epsilon={latest.epsilon:.3f} "
        f"loss={loss_text}"
    )
