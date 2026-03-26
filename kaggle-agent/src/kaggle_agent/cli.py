"""CLI entry point for kaggle-agent."""

from __future__ import annotations

import click
from pathlib import Path
from rich.console import Console

console = Console()


@click.group()
def main():
    """KaggleAgent: Autonomous Kaggle competition research framework."""
    pass


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--max-experiments", "-n", type=int, default=None)
def run(config_path: str, max_experiments: int | None):
    """Run the experiment loop with strategies from config."""
    from kaggle_agent.config import load_config
    from kaggle_agent.loop import ExperimentLoop, Strategy

    config = load_config(config_path)
    loop = ExperimentLoop(config)

    # Load default strategies from config
    for name, params in config.models.items():
        model_type = params.pop("type", name)
        loop.add_strategy(Strategy(
            name=name,
            model_type=model_type,
            params=params,
            description=f"{model_type} default config",
        ))

    loop.run(max_experiments=max_experiments)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="submission.csv")
def submit(config_path: str, output: str):
    """Generate submission from best model."""
    from kaggle_agent.config import load_config
    from kaggle_agent.loop import ExperimentLoop

    config = load_config(config_path)
    loop = ExperimentLoop(config)
    loop.generate_best_submission(output)


@main.command()
@click.argument("results_path", type=click.Path(exists=True))
def summary(results_path: str):
    """Print experiment summary."""
    from kaggle_agent.tracking.experiments import ExperimentTracker

    tracker = ExperimentTracker(results_path)
    console.print(tracker.summary())


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def explore(config_path: str):
    """Run EDA on competition data."""
    from kaggle_agent.config import load_config
    from kaggle_agent.pipeline.data import load_competition_data, get_data_summary

    config = load_config(config_path)
    X_train, X_test, y_train, test_ids = load_competition_data(
        train_path=config.data.train_path,
        test_path=config.data.test_path,
        target_column=config.data.target_column,
        id_column=config.data.id_column,
    )

    console.print(get_data_summary(X_train, "Training Features"))
    console.print()
    console.print(get_data_summary(X_test, "Test Features"))
    console.print()

    # Target distribution
    console.print("[bold]Target Distribution:[/]")
    counts = y_train.value_counts()
    for val, count in counts.items():
        pct = count / len(y_train) * 100
        console.print(f"  {val}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
