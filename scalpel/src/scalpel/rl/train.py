"""
Training script for the SCALPEL PPO retrieval agent.

Two-phase loop per iteration:
  1. Rollout collection — run queries through the RAG environment with the
     current policy (or random actions early on), score with the evaluator,
     store transitions in the buffer.
  2. PPO update — train policy on the collected rollouts.

Usage (CLI):
    scalpel train-rl

Usage (Python):
    from scalpel.rl.train import train
    train(queries=["What is attention?", ...], n_iterations=10)
"""

import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from scalpel.rl.environment import RAGEnvironment
from scalpel.rl.agent import PPOAgent, PPOConfig

console = Console()

# Seed queries covering different aspects of ML papers.
# Extend this list with domain-specific queries for your corpus.
DEFAULT_QUERIES = [
    "What is the main contribution of this paper?",
    "How was the model trained and what data was used?",
    "What are the key experimental results?",
    "What are the limitations acknowledged by the authors?",
    "How does attention mechanism work in transformers?",
    "What baseline models were compared against?",
    "What evaluation metrics were used?",
    "What datasets were used for evaluation?",
    "How does the proposed method improve over prior work?",
    "What future work do the authors suggest?",
]


def collect_rollouts(
    agent: PPOAgent,
    env: RAGEnvironment,
    queries: list[str],
    n_rollouts: int,
    explore_ratio: float = 0.3,
) -> dict[str, float]:
    """
    Collect rollouts into agent.buffer.

    Args:
        agent:         PPO agent (policy used for non-random steps).
        env:           RAG environment.
        queries:       Pool of queries to sample from.
        n_rollouts:    Number of steps to collect.
        explore_ratio: Fraction of steps using random actions.

    Returns:
        Dict with mean_reward and mean_eval_score.
    """
    import random

    rewards = []
    scores = []

    for i in range(n_rollouts):
        query = random.choice(queries)

        state = env.embed_query(query)

        if random.random() < explore_ratio:
            action = env.random_action()
            log_prob = 0.0
            value = 0.0
        else:
            action, log_prob, value = agent.select_action(state)

        console.print(
            f"  [dim]Step {i + 1}/{n_rollouts}[/dim]  query=[cyan]{query[:50]}[/cyan]",
            highlight=False,
        )

        try:
            result = env.step(query, action)
        except Exception as e:
            from scalpel.rl.environment import DailyLimitError
            if isinstance(e, DailyLimitError):
                console.print(f"\n[yellow]⚠ Daily request limit reached. "
                              f"Stopping rollout collection — resume tomorrow.[/yellow]")
                break
            console.print(f"    [yellow]⚠ Step failed, skipping:[/yellow] {e}")
            continue

        agent.buffer.store(
            state=result.state,
            action=result.action,
            reward=result.reward,
            log_prob=log_prob,
            value=value,
        )

        rewards.append(result.reward)
        scores.append(result.eval_score)

        cached_tag = "[dim](cached)[/dim] " if env.cache.get(query, action) is not None else ""
        console.print(
            f"    {cached_tag}reward=[bold]{result.reward:+.3f}[/bold]  "
            f"eval={result.eval_score:.1f}/10  "
            f"params={result.params}",
            highlight=False,
        )

    return {
        "mean_reward": sum(rewards) / len(rewards),
        "mean_eval_score": sum(scores) / len(scores),
    }


def _find_latest_checkpoint(save_dir: Path) -> tuple[Path, int] | None:
    """Return (checkpoint_path, iteration_number) for the latest saved checkpoint."""
    checkpoints = sorted(save_dir.glob("agent_iter_*.pt"))
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    try:
        iteration = int(latest.stem.split("_")[-1])
    except ValueError:
        return None
    return latest, iteration


def train(
    queries: list[str] | None = None,
    n_iterations: int = 10,
    n_rollouts_per_iter: int = 20,
    explore_ratio: float = 0.5,
    save_dir: Path | str = Path("models/rl"),
    config: PPOConfig | None = None,
    resume: bool = False,
) -> PPOAgent:
    """
    Full PPO training loop.

    Args:
        queries:              Query pool. Uses DEFAULT_QUERIES if None.
        n_iterations:         Number of collect → update cycles.
        n_rollouts_per_iter:  Rollouts collected each iteration.
        explore_ratio:        Fraction of random actions (anneals to 0.1).
        save_dir:             Directory to save checkpoints.
        config:               PPO hyperparameters.
        resume:               Load latest checkpoint and continue from next iteration.

    Returns:
        Trained PPOAgent.
    """
    queries = queries or DEFAULT_QUERIES
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    agent = PPOAgent(config=config)
    start_iteration = 1

    # Load existing history
    history_path = save_dir / "training_history.json"
    history: list[dict] = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = []

    if resume:
        found = _find_latest_checkpoint(save_dir)
        if found is None:
            console.print("[yellow]No checkpoint found — starting from scratch.[/yellow]")
        else:
            ckpt_path, completed = found
            agent.load(ckpt_path)
            start_iteration = completed + 1
            console.print(
                f"[green]✓[/green] Resumed from [cyan]{ckpt_path.name}[/cyan] "
                f"(iteration {completed} complete, continuing from {start_iteration})"
            )

    env = RAGEnvironment(verbose=False)

    remaining = n_iterations - (start_iteration - 1)
    console.print(Panel(
        f"[bold]PPO Retrieval Agent Training[/bold]\n\n"
        f"Queries: {len(queries)}  |  "
        f"Iterations: {start_iteration}→{start_iteration + remaining - 1}  |  "
        f"Rollouts/iter: {n_rollouts_per_iter}"
        + (f"  |  [green]Resumed[/green]" if resume and start_iteration > 1 else ""),
        title="[bold cyan]🔪 SCALPEL RL[/bold cyan]",
        border_style="cyan",
    ))

    for iteration in range(start_iteration, start_iteration + remaining):
        # Anneal exploration across total iterations (not reset on resume)
        total_iterations = start_iteration + remaining - 1
        current_explore = max(0.1, explore_ratio * (1 - iteration / total_iterations))

        console.print(f"\n[bold]Iteration {iteration}/{n_iterations}[/bold]  "
                      f"(explore={current_explore:.2f})\n")

        t0 = time.time()

        rollout_stats = collect_rollouts(
            agent=agent,
            env=env,
            queries=queries,
            n_rollouts=n_rollouts_per_iter,
            explore_ratio=current_explore,
        )

        ppo_metrics = agent.update() if len(agent.buffer) > 0 else {}

        elapsed = time.time() - t0

        record = {
            "iteration": iteration,
            **rollout_stats,
            **ppo_metrics,
            "elapsed_s": round(elapsed, 1),
        }
        history.append(record)

        _print_iteration_summary(record)

        # Save checkpoint every iteration
        ckpt_path = save_dir / f"agent_iter_{iteration:03d}.pt"
        agent.save(ckpt_path)

    # Save final model and training history
    agent.save(save_dir / "agent_final.pt")
    history_path = save_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    console.print(f"\n[green]✓[/green] Training complete. "
                  f"Model saved to [cyan]{save_dir / 'agent_final.pt'}[/cyan]\n")

    return agent


def _print_iteration_summary(record: dict):
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", width=20)
    table.add_column(style="bold")

    color = "green" if record["mean_reward"] > 0 else "red"

    table.add_row("Mean reward",     f"[{color}]{record['mean_reward']:+.3f}[/{color}]")
    table.add_row("Mean eval score", f"{record['mean_eval_score']:.2f}/10")
    if "policy_loss" in record:
        table.add_row("Policy loss",  f"{record['policy_loss']:.4f}")
        table.add_row("Value loss",   f"{record['value_loss']:.4f}")
        table.add_row("Entropy",      f"{record['entropy']:.4f}")
    table.add_row("Time",            f"{record['elapsed_s']}s")

    console.print(table)
