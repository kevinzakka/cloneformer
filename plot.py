"""Generate pretty plots from CSV files downloaded from Tensorboard."""

import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUCCESS_RATE = {
    "lstm": "/home/kevin/Downloads/run-lstm_2_train_logs-tag-valid_success.csv",
    "mlp": "/home/kevin/Downloads/run-alldata_largepolicy_dropout_small_train_logs-tag-valid_success.csv",
}
LOSSES = {
    "lstm": "/home/kevin/Downloads/run-lstm_2_train_logs-tag-train_loss.csv",
    "mlp": "/home/kevin/Downloads/run-alldata_largepolicy_dropout_small_train_logs-tag-train_loss.csv",
}
SIZE = 30
DPI = 200
STD_SCALAR = 0.5
ALGO_TO_COLOR = {
    "MLP": "tab:blue",
    "LSTM": "tab:red",
}


def load_csv(filename: str) -> np.ndarray:
    df = pd.read_csv(filename)
    return np.vstack([[0, 0], df.to_numpy()[:, 1:]])


def update_matplotlib_params():
    params = {
        "legend.fontsize": SIZE * 0.7,
        "axes.titlesize": SIZE * 0.9,
        "axes.labelsize": SIZE * 0.9,
        "xtick.labelsize": SIZE * 0.7,
        "ytick.labelsize": SIZE * 0.7,
    }
    plt.rcParams.update(params)


def plot_success_rate():
    data = {
        "MLP": load_csv(SUCCESS_RATE["mlp"]),
        "LSTM": load_csv(SUCCESS_RATE["lstm"]),
    }
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(12, 9))
    for method_name, method_data in data.items():
        x = method_data[:, 0] / 1000  # Plot in thousands.
        y = method_data[:, 1]
        ax.plot(x, y, lw=2, label=method_name, color=ALGO_TO_COLOR[method_name])
        ax.set_xlabel("Steps (thousands)")
        ax.set_ylabel("Success Rate")
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", linewidth=0.5)
    plt.savefig(osp.join("images", "success_rates.png"), format="png", dpi=DPI)
    plt.close()


def plot_losses():
    data = {
        "MLP": load_csv(LOSSES["mlp"])[1:],
        "LSTM": load_csv(LOSSES["lstm"])[1:],
    }
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(12, 9))
    for method_name, method_data in data.items():
        x = method_data[:, 0] / 1000  # Plot in thousands.
        y = method_data[:, 1]
        ax.plot(x, y, lw=2, label=method_name, color=ALGO_TO_COLOR[method_name])
        ax.set_xlabel("Steps (thousands)")
        ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--", linewidth=0.5)
    plt.savefig(osp.join("images", "losses.png"), format="png", dpi=DPI)
    plt.close()


if __name__ == "__main__":
    update_matplotlib_params()
    plot_success_rate()
    plot_losses()
