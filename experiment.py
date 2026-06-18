# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

import numpy as np

from run_experiment import SEEDS, TrainConfig, load_daily_data, standardize_daily, train_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM, Transformer, and ConvTransformer, then print final metrics only."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        choices=[0, 90, 365],
        help="Forecast length. Use 0 to run both 90 and 365 days.",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to run.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs per model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = SEEDS[: max(1, min(args.seeds, len(SEEDS)))]
    daily = load_daily_data()
    print(f"Dataset days: {len(daily)}")
    print("Task: use past 90 days to forecast future global_active_power")
    print(f"Seeds: {seeds}")
    print()

    horizons = [90, 365] if args.horizon == 0 else [args.horizon]
    all_rows = []
    for horizon in horizons:
        epochs = args.epochs if args.epochs is not None else (10 if horizon == 365 else 14)
        data = standardize_daily(daily, horizon)
        print(f"===== Horizon: {horizon} days, epochs: {epochs} =====")
        for model_name in ["LSTM", "Transformer", "ConvTransformer"]:
            mses = []
            maes = []
            for seed in seeds:
                cfg = TrainConfig(model_name=model_name, horizon=horizon, seed=seed, epochs=epochs)
                result = train_one(cfg, data)
                mses.append(result["mse"])
                maes.append(result["mae"])
                print(
                    f"{model_name:15s} seed={seed} "
                    f"MSE={result['mse']:.4f} MAE={result['mae']:.4f} "
                    f"epochs={result['epochs_ran']}"
                )
            all_rows.append(
                (
                    horizon,
                    model_name,
                    np.mean(mses),
                    np.std(mses, ddof=1) if len(mses) > 1 else 0.0,
                    np.mean(maes),
                    np.std(maes, ddof=1) if len(maes) > 1 else 0.0,
                )
            )
        print()

    print("Final Summary")
    print("horizon  model             mse_mean      mse_std       mae_mean      mae_std")
    for horizon, model_name, mse_mean, mse_std, mae_mean, mae_std in all_rows:
        print(
            f"{horizon:7d}  {model_name:15s} "
            f"{mse_mean:12.4f} {mse_std:12.4f} {mae_mean:12.4f} {mae_std:12.4f}"
        )


if __name__ == "__main__":
    main()
