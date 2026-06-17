import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("artifacts/outputs/whulx_reproduction")
HISTORY_PATH = OUT_DIR / "training_history.csv"
SUMMARY_PATH = OUT_DIR / "summary_metrics.csv"
RESULT_PATH = OUT_DIR / "result.json"


def save_current_figure(filename):
    path = OUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    history_df = pd.read_csv(HISTORY_PATH)
    evaluation_df = pd.read_csv(SUMMARY_PATH)
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    history_plot = history_df.copy()
    history_plot["reward_smooth"] = history_plot["reward"].rolling(window=50, min_periods=1).mean()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(history_plot["episode"], history_plot["reward"], alpha=0.25, label="Reward tung episode")
    axes[0].plot(history_plot["episode"], history_plot["reward_smooth"], linewidth=2, label="Reward rolling mean")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Qua trinh huan luyen DQN")
    axes[0].legend()

    axes[1].plot(history_plot["episode"], history_plot["epsilon"], color="tab:orange")
    axes[1].set_ylabel("Epsilon")

    axes[2].plot(history_plot["episode"], history_plot["avg_loss"], color="tab:green")
    axes[2].set_ylabel("Avg loss")
    axes[2].set_xlabel("Episode")
    save_current_figure("training_history_plot.png")

    # Filter for DQN and Human, and add DQN Gốc (WHU-LX)
    evaluation_df = evaluation_df[evaluation_df["controller"].isin(["DQN", "Human"])].copy()
    evaluation_df.loc[evaluation_df["controller"] == "DQN", "controller"] = "DQN Cải tiến (Nhóm)"
    evaluation_df.loc[evaluation_df["controller"] == "Human", "controller"] = "Human (Thực tế)"
    
    dqn_goc = {
        "controller": "DQN Gốc (WHU-LX)",
        "ac_on_pct": 0.0,
        "comfort_pct": 67.99,
        "mean_indoor_temp": 24.8,
        "total_reward": -14.51,
        "window_open_pct": 34.09
    }
    evaluation_df = pd.concat([evaluation_df, pd.DataFrame([dqn_goc])], ignore_index=True)

    metric_cols = ["comfort_pct", "ac_on_pct", "window_open_pct"]
    comparison_df = evaluation_df.set_index("controller")[metric_cols]
    comparison_df = comparison_df.reindex(["DQN Gốc (WHU-LX)", "Human (Thực tế)", "DQN Cải tiến (Nhóm)"])
    
    ax = comparison_df.plot(kind="bar", figsize=(10, 5), width=0.65)
    ax.set_title("So sanh comfort, AC on va window open")
    ax.set_ylabel("Ti le (%)")
    ax.set_xlabel("Controller")
    ax.legend(["Comfort", "AC on", "Window open"], loc="upper right")
    plt.xticks(rotation=0)
    save_current_figure("controller_comparison_metrics.png")

    reward_df = evaluation_df.dropna(subset=["total_reward"]).set_index("controller")["total_reward"]
    reward_df = reward_df.reindex(["DQN Gốc (WHU-LX)", "DQN Cải tiến (Nhóm)"])
    ax = reward_df.plot(kind="barh", figsize=(10, 4), color="tab:purple", width=0.4)
    ax.set_title("Total reward theo controller")
    ax.set_xlabel("Total reward trung binh")
    ax.set_ylabel("Controller")
    save_current_figure("controller_total_reward.png")

    action_dist = result.get("evaluation", {}).get("action_distribution_pct", {}).get("DQN", {})
    if action_dist:
        action_dist_df = (
            pd.Series({int(k): v for k, v in action_dist.items()}, name="percentage")
            .rename_axis("action")
            .reset_index()
            .sort_values("action")
        )
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(action_dist_df["action"].astype(str), action_dist_df["percentage"], color="tab:blue")
        ax.set_title("Phan phoi action cua DQN tren tap danh gia")
        ax.set_xlabel("Action")
        ax.set_ylabel("Ti le chon (%)")
        save_current_figure("dqn_action_distribution.png")
    else:
        print("No DQN action distribution found in result.json")


if __name__ == "__main__":
    main()
