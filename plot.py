"""Generate pretty plots from CSV files downloaded from Tensorboard."""

import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================================== #
# Data structures.
# =========================================================================== #
SUCCESS_RATE = {
    "lstm": "/home/kevin/Downloads/run-lstm_2_train_logs-tag-valid_success.csv",
    "mlp": "/home/kevin/Downloads/run-alldata_largepolicy_dropout_small_train_logs-tag-valid_success.csv",
}

# =========================================================================== #


def load_csv(filename):
    df = pd.read_csv(filename)
    return np.vstack([[0, 0], df.to_numpy()[:, 1:]])


def align(list_arrs):
    min_len = list_arrs[0].shape[0]
    for arr in list_arrs[1:]:
        if arr.shape[0] < min_len:
            min_len = arr.shape[0]
    return truncate(list_arrs, min_len)


def truncate(list_arrs, trunc_len):
    return [arr[:trunc_len] for arr in list_arrs]


DPI = 100
STD_SCALAR = 0.5
ALGO_TO_COLOR = {
    "MLP": "tab:blue",
    "LSTM": "tab:red",
}
size = 30
params = {
    "legend.fontsize": size * 0.7,
    "axes.titlesize": size * 0.9,
    "axes.labelsize": size * 0.9,
    "xtick.labelsize": size * 0.7,
    "ytick.labelsize": size * 0.7,
}
plt.rcParams.update(params)


def plot_success_rate():
    mlp_data = load_csv(SUCCESS_RATE["mlp"])
    lstm_data = load_csv(SUCCESS_RATE["lstm"])

    data = {
        "MLP": mlp_data,
        "LSTM": lstm_data,
    }

    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(12, 9))
    for method_name, method_data in data.items():
        # Plot in thousands.
        sr_x = method_data[:, 0] / 1000
        sr_y = method_data[:, 1]
        ax.plot(sr_x, sr_y, lw=2, label=method_name, color=ALGO_TO_COLOR[method_name])

        ax.set_xlabel("Steps (thousands)")
        ax.set_ylabel("Success Rate")

    ax.legend(loc="lower right")
    ax.grid(linestyle="--", linewidth=0.5)

    plt.savefig(osp.join("images", "success_rates.png"), format="png", dpi=DPI)
    plt.close()


if __name__ == "__main__":
    plot_success_rate()
