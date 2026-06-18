# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import math
import random
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
FIG_DIR = OUT_DIR / "figures"
RAW_ZIP = DATA_DIR / "household_power_consumption.zip"
RAW_TXT = DATA_DIR / "household_power_consumption.txt"
DAILY_CSV = DATA_DIR / "daily_power.csv"

UCI_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"

INPUT_LEN = 90
HORIZONS = [90, 365]
SEEDS = [2026, 2027, 2028, 2029, 2030]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    model_name: str
    horizon: int
    seed: int
    epochs: int
    batch_size: int = 16
    lr: float = 1e-3
    hidden: int = 32
    d_model: int = 32
    nhead: int = 4
    layers: int = 1
    dropout: float = 0.10


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        d_model: int = 32,
        nhead: int = 4,
        layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model, max_len=INPUT_LEN + 8)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.input_proj(x))
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


class ConvTransformerForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        d_model: int = 32,
        nhead: int = 4,
        layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(d_model, max_len=INPUT_LEN + 8)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h = self.encoder(self.pos(h))
        pooled = h.mean(dim=1)
        pooled = pooled * self.gate(pooled)
        return self.head(pooled)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_uci_if_needed() -> None:
    if RAW_TXT.exists():
        return
    if not RAW_ZIP.exists():
        print(f"Downloading UCI dataset to {RAW_ZIP} ...")
        urllib.request.urlretrieve(UCI_URL, RAW_ZIP)
    print("Extracting UCI dataset ...")
    with zipfile.ZipFile(RAW_ZIP, "r") as zf:
        target = "household_power_consumption.txt"
        with zf.open(target) as src, RAW_TXT.open("wb") as dst:
            dst.write(src.read())


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {
        "sub_metering_3": "sub_metering_3",
        "sub_metering_1": "sub_metering_1",
        "sub_metering_2": "sub_metering_2",
        "global_active_power": "global_active_power",
        "global_reactive_power": "global_reactive_power",
        "global_intensity": "global_intensity",
        "date": "date",
        "time": "time",
    }
    df = df.rename(columns=rename)
    return df


def load_provided_csvs() -> pd.DataFrame | None:
    candidates = []
    for name in ["train.csv", "test.csv", "tes.csv"]:
        path = DATA_DIR / name
        if path.exists():
            candidates.append(path)
    if not candidates:
        return None

    parts = []
    for path in candidates:
        part = pd.read_csv(path)
        part = normalize_column_names(part)
        part["source_split"] = path.stem
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce", dayfirst=True)
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    else:
        return None

    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    return aggregate_daily(df)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df)
    df = df.copy()
    df["date_only"] = pd.to_datetime(df["datetime"]).dt.date
    numeric_cols = [
        "global_active_power",
        "global_reactive_power",
        "voltage",
        "global_intensity",
        "sub_metering_1",
        "sub_metering_2",
        "sub_metering_3",
        "rr",
        "nbjrr1",
        "nbjrr5",
        "nbjrr10",
        "nbjbrou",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("?", np.nan), errors="coerce")

    agg: dict[str, str] = {}
    for col in ["global_active_power", "global_reactive_power", "sub_metering_1", "sub_metering_2", "sub_metering_3"]:
        if col in df.columns:
            agg[col] = "sum"
    for col in ["voltage", "global_intensity"]:
        if col in df.columns:
            agg[col] = "mean"
    for col in ["rr", "nbjrr1", "nbjrr5", "nbjrr10", "nbjbrou"]:
        if col in df.columns:
            agg[col] = "first"
    daily = df.groupby("date_only").agg(agg).reset_index().rename(columns={"date_only": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    return finalize_daily(daily)


def load_uci_daily() -> pd.DataFrame:
    if DAILY_CSV.exists():
        return pd.read_csv(DAILY_CSV, parse_dates=["date"])

    download_uci_if_needed()
    print("Reading raw minute-level power data ...")
    df = pd.read_csv(
        RAW_TXT,
        sep=";",
        na_values="?",
        low_memory=False,
    )
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    for col in df.columns:
        if col not in {"date", "time", "datetime"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    daily = aggregate_daily(df)
    daily.to_csv(DAILY_CSV, index=False)
    return daily


def finalize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy().sort_values("date").reset_index(drop=True)
    full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_dates).rename_axis("date").reset_index()

    for col in daily.columns:
        if col == "date":
            continue
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
        daily[col] = daily[col].interpolate(limit_direction="both")
        daily[col] = daily[col].fillna(daily[col].median())

    if "sub_metering_3" not in daily.columns:
        daily["sub_metering_3"] = 0.0
    if "sub_metering_remainder" not in daily.columns and "global_active_power" in daily.columns:
        daily["sub_metering_remainder"] = (
            daily["global_active_power"] * 1000.0 / 60.0
            - daily.get("sub_metering_1", 0.0)
            - daily.get("sub_metering_2", 0.0)
            - daily.get("sub_metering_3", 0.0)
        )
    daily["month"] = daily["date"].dt.month
    daily["dayofweek"] = daily["date"].dt.dayofweek
    daily["dayofyear"] = daily["date"].dt.dayofyear
    daily["is_weekend"] = (daily["dayofweek"] >= 5).astype(float)
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)
    daily["dow_sin"] = np.sin(2 * np.pi * daily["dayofweek"] / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["dayofweek"] / 7)
    return daily


def load_daily_data() -> pd.DataFrame:
    ensure_dirs()
    provided = load_provided_csvs()
    if provided is not None and len(provided) > INPUT_LEN + max(HORIZONS) + 10:
        daily = provided
        daily.to_csv(DAILY_CSV, index=False)
    else:
        daily = load_uci_daily()
    return daily


def make_windows(values: np.ndarray, target: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, starts = [], [], []
    for start in range(0, len(values) - INPUT_LEN - horizon + 1):
        xs.append(values[start : start + INPUT_LEN])
        ys.append(target[start + INPUT_LEN : start + INPUT_LEN + horizon])
        starts.append(start)
    return np.stack(xs), np.stack(ys), np.array(starts)


def train_val_test_split(x: np.ndarray, y: np.ndarray, starts: np.ndarray) -> dict[str, np.ndarray]:
    n = len(x)
    train_end = max(1, int(n * 0.70))
    val_end = max(train_end + 1, int(n * 0.85))
    return {
        "x_train": x[:train_end],
        "y_train": y[:train_end],
        "x_val": x[train_end:val_end],
        "y_val": y[train_end:val_end],
        "x_test": x[val_end:],
        "y_test": y[val_end:],
        "starts_test": starts[val_end:],
    }


def standardize_daily(daily: pd.DataFrame, horizon: int) -> dict[str, object]:
    feature_cols = [
        c
        for c in daily.columns
        if c != "date" and pd.api.types.is_numeric_dtype(daily[c])
    ]
    target_col = "global_active_power"
    if target_col not in feature_cols:
        raise ValueError("global_active_power is required as target column.")

    raw_features = daily[feature_cols].astype(float).values
    raw_target = daily[target_col].astype(float).values
    # Fit scaler only on the earliest part used by training windows.
    n_possible = len(daily) - INPUT_LEN - horizon + 1
    train_end = max(1, int(n_possible * 0.70))
    train_last_day = train_end + INPUT_LEN
    feat_mean = raw_features[:train_last_day].mean(axis=0)
    feat_std = raw_features[:train_last_day].std(axis=0)
    feat_std[feat_std == 0] = 1.0
    target_mean = float(raw_target[:train_last_day].mean())
    target_std = float(raw_target[:train_last_day].std() or 1.0)

    x_scaled = (raw_features - feat_mean) / feat_std
    y_scaled = (raw_target - target_mean) / target_std
    x, y, starts = make_windows(x_scaled, y_scaled, horizon)
    split = train_val_test_split(x, y, starts)
    split.update(
        {
            "feature_cols": feature_cols,
            "target_mean": target_mean,
            "target_std": target_std,
            "dates": daily["date"].tolist(),
            "raw_target": raw_target,
        }
    )
    return split


def build_model(cfg: TrainConfig, n_features: int) -> nn.Module:
    if cfg.model_name == "LSTM":
        return LSTMForecaster(n_features, cfg.horizon, hidden=cfg.hidden, dropout=cfg.dropout)
    if cfg.model_name == "Transformer":
        return TransformerForecaster(
            n_features,
            cfg.horizon,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            layers=cfg.layers,
            dropout=cfg.dropout,
        )
    if cfg.model_name == "ConvTransformer":
        return ConvTransformerForecaster(
            n_features,
            cfg.horizon,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            layers=cfg.layers,
            dropout=cfg.dropout,
        )
    raise ValueError(cfg.model_name)


def evaluate(model: nn.Module, loader: DataLoader, target_mean: float, target_std: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
    pred_scaled = np.concatenate(preds, axis=0)
    true_scaled = np.concatenate(trues, axis=0)
    pred = pred_scaled * target_std + target_mean
    true = true_scaled * target_std + target_mean
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    return mse, mae, pred, true


def train_one(cfg: TrainConfig, data: dict[str, object]) -> dict[str, object]:
    set_seed(cfg.seed)
    train_ds = WindowDataset(data["x_train"], data["y_train"])
    val_ds = WindowDataset(data["x_val"], data["y_val"])
    test_ds = WindowDataset(data["x_test"], data["y_test"])
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = build_model(cfg, n_features=data["x_train"].shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    patience = 4
    wait = 0
    history = []
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_mse_scaled, _, _, _ = evaluate(model, val_loader, 0.0, 1.0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mse_scaled": val_mse_scaled})
        if val_mse_scaled < best_val:
            best_val = val_mse_scaled
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    mse, mae, pred, true = evaluate(model, test_loader, data["target_mean"], data["target_std"])
    return {
        "model": cfg.model_name,
        "horizon": cfg.horizon,
        "seed": cfg.seed,
        "epochs_ran": len(history),
        "best_val_mse_scaled": best_val,
        "mse": mse,
        "mae": mae,
        "train_seconds": time.time() - start_time,
        "prediction": pred,
        "ground_truth": true,
        "history": history,
    }


def baseline_persistence(data: dict[str, object], horizon: int) -> dict[str, float]:
    x_test = data["x_test"]
    y_test = data["y_test"] * data["target_std"] + data["target_mean"]
    target_idx = data["feature_cols"].index("global_active_power")
    last_scaled = x_test[:, -1, target_idx]
    last = last_scaled * data["target_std"] + data["target_mean"]
    pred = np.repeat(last[:, None], horizon, axis=1)
    return {
        "mse": float(np.mean((pred - y_test) ** 2)),
        "mae": float(np.mean(np.abs(pred - y_test))),
    }


def save_dataset_overview(daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["date"], daily["global_active_power"], color="#2E74B5", linewidth=1.0)
    ax.set_title("Daily Global Active Power")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily sum of global_active_power")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "daily_power_overview.png", dpi=180)
    plt.close(fig)


def save_prediction_plot(
    pred: np.ndarray,
    true: np.ndarray,
    model: str,
    horizon: int,
    seed: int,
    dates: list[pd.Timestamp],
    starts_test: np.ndarray,
) -> str:
    # Plot the last test sample to show a contiguous future horizon.
    idx = -1
    sample_start = int(starts_test[idx])
    first_target = sample_start + INPUT_LEN
    plot_dates = pd.to_datetime(dates[first_target : first_target + horizon])
    if len(plot_dates) != horizon:
        plot_dates = pd.date_range("2010-01-01", periods=horizon, freq="D")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_dates, true[idx], label="Ground Truth", color="#1f77b4", linewidth=1.4)
    ax.plot(plot_dates, pred[idx], label="Prediction", color="#d62728", linewidth=1.3, alpha=0.85)
    ax.set_title(f"{model} - {horizon}-day Forecast (seed={seed})")
    ax.set_xlabel("Date")
    ax.set_ylabel("global_active_power")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    name = f"forecast_{model}_{horizon}d_seed{seed}.png"
    fig.savefig(FIG_DIR / name, dpi=180)
    plt.close(fig)
    return str(FIG_DIR / name)


def write_csvs(rows: list[dict[str, object]]) -> pd.DataFrame:
    results = pd.DataFrame(
        [
            {
                "model": r["model"],
                "horizon": r["horizon"],
                "seed": r["seed"],
                "epochs_ran": r["epochs_ran"],
                "best_val_mse_scaled": r["best_val_mse_scaled"],
                "mse": r["mse"],
                "mae": r["mae"],
                "train_seconds": r["train_seconds"],
            }
            for r in rows
        ]
    )
    results.to_csv(OUT_DIR / "results.csv", index=False, encoding="utf-8-sig")
    summary = (
        results.groupby(["model", "horizon"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            epochs_mean=("epochs_ran", "mean"),
            seconds_mean=("train_seconds", "mean"),
        )
        .sort_values(["horizon", "mse_mean"])
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    ensure_dirs()
    print(f"Using device: {DEVICE}")
    daily = load_daily_data()
    save_dataset_overview(daily)
    metadata = {
        "n_days": int(len(daily)),
        "start_date": str(daily["date"].min().date()),
        "end_date": str(daily["date"].max().date()),
        "input_len": INPUT_LEN,
        "horizons": HORIZONS,
        "seeds": SEEDS,
        "device": str(DEVICE),
        "data_source": "provided train/test csv" if (DATA_DIR / "train.csv").exists() else "UCI Individual household electric power consumption",
    }

    all_rows: list[dict[str, object]] = []
    best_for_plot: dict[tuple[str, int], dict[str, object]] = {}
    for horizon in HORIZONS:
        data = standardize_daily(daily, horizon)
        metadata[f"horizon_{horizon}_n_train_windows"] = int(len(data["x_train"]))
        metadata[f"horizon_{horizon}_n_val_windows"] = int(len(data["x_val"]))
        metadata[f"horizon_{horizon}_n_test_windows"] = int(len(data["x_test"]))
        metadata[f"horizon_{horizon}_baseline"] = baseline_persistence(data, horizon)
        for model_name in ["LSTM", "Transformer", "ConvTransformer"]:
            for seed in SEEDS:
                epochs = 14 if horizon == 90 else 10
                cfg = TrainConfig(model_name=model_name, horizon=horizon, seed=seed, epochs=epochs)
                print(f"Training {model_name}, horizon={horizon}, seed={seed}")
                row = train_one(cfg, data)
                row["plot_path"] = save_prediction_plot(
                    row["prediction"],
                    row["ground_truth"],
                    model_name,
                    horizon,
                    seed,
                    data["dates"],
                    data["starts_test"],
                )
                all_rows.append(row)
                key = (model_name, horizon)
                if key not in best_for_plot or row["mse"] < best_for_plot[key]["mse"]:
                    best_for_plot[key] = row

    summary = write_csvs(all_rows)
    metadata["feature_columns"] = standardize_daily(daily, 90)["feature_cols"]
    metadata["summary_records"] = summary.to_dict(orient="records")
    with (OUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    best_rows = []
    for (model, horizon), row in sorted(best_for_plot.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        best_rows.append(
            {
                "model": model,
                "horizon": horizon,
                "seed": row["seed"],
                "mse": row["mse"],
                "mae": row["mae"],
                "plot_path": row["plot_path"],
            }
        )
    pd.DataFrame(best_rows).to_csv(OUT_DIR / "best_plots.csv", index=False, encoding="utf-8-sig")
    print("Finished. Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
