import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

# Add scripts directory to path to import components
sys.path.append(str(Path(__file__).parent))

from reproduce_whulx_pipeline import (
    DQN, load_and_prepare_data, train_xgboost, choose_day,
    comfort_bounds, map_action_to_dataframe, normalize_state, epsilon_greedy_policy
)

def original_dqn_policy(q_network, state):
    norm_state = normalize_state(state)
    q_values = q_network(np.array([norm_state], dtype=np.float32)).numpy()[0]
    # Force AC to be off by masking all actions except 0 (off/closed) and 12 (off/open)
    masked_q = np.full_like(q_values, -np.inf)
    masked_q[0] = q_values[0]
    masked_q[12] = q_values[12]
    return int(np.argmax(masked_q))

def rollout_day_custom(model_xgb, q_network, data_test, xgboost_test, policy_type="improved"):
    data_pre_test = data_test.copy()
    xgboost_pre_test = xgboost_test.copy()
    num_features = 8
    actions = []
    rewards = []

    for step in range(6):
        xgboost_pre_test.loc[
            step,
            ["Target_Temp", "AC_Status", "Window_Status", "CLast_Time", "WLast_Time", "CLast_Time_T", "WLast_Time_T"],
        ] = 0
        hour_row_df = pd.DataFrame(xgboost_pre_test.iloc[step]).T
        next_differ_temp = model_xgb.predict(hour_row_df)[0]
        next_in_temp = xgboost_pre_test.iloc[step]["Indoor_Temp"] + next_differ_temp
        xgboost_pre_test.at[step + 1, "Indoor_Temp"] = next_in_temp
        data_pre_test.at[step + 1, "Indoor_Temp"] = next_in_temp

    state = data_pre_test.iloc[6, :num_features].values
    for step in range(6, 18):
        if policy_type == "improved":
            action = epsilon_greedy_policy(q_network, state, 0.0, 24)
        elif policy_type == "original":
            action = original_dqn_policy(q_network, state)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
            
        target_temp, ac_status, window_status, c_last_time, w_last_time = map_action_to_dataframe(action)
        actions.append(action)

        xgboost_pre_test.at[step, "Target_Temp"] = target_temp
        xgboost_pre_test.at[step, "AC_Status"] = ac_status
        xgboost_pre_test.at[step, "Window_Status"] = window_status
        xgboost_pre_test.at[step, "CLast_Time"] = c_last_time
        xgboost_pre_test.at[step, "WLast_Time"] = w_last_time
        xgboost_pre_test.at[step, "CLast_Time_T"] = (
            xgboost_pre_test.iloc[step - 1]["CLast_Time_T"] + c_last_time if c_last_time > 0 else 0
        )
        xgboost_pre_test.at[step, "WLast_Time_T"] = (
            xgboost_pre_test.iloc[step - 1]["WLast_Time_T"] + w_last_time if w_last_time > 0 else 0
        )

        hour_row_df = pd.DataFrame(xgboost_pre_test.iloc[step]).T
        next_differ_temp = model_xgb.predict(hour_row_df)[0]
        next_in_temp = xgboost_pre_test.iloc[step]["Indoor_Temp"] + next_differ_temp
        xgboost_pre_test.at[step + 1, "Indoor_Temp"] = next_in_temp
        data_pre_test.at[step + 1, "Indoor_Temp"] = next_in_temp

        next_state = data_pre_test.iloc[step + 1, :num_features].values
        
        # Calculate reward
        indoor_temp = float(next_state[0])
        outdoor_temp = float(next_state[2])
        _, lower, upper, _ = comfort_bounds(outdoor_temp)
        if lower <= indoor_temp <= upper:
            comfort_penalty = 0.0
        elif indoor_temp < lower:
            comfort_penalty = -((indoor_temp - lower) ** 2)
        else:
            comfort_penalty = -((indoor_temp - upper) ** 2)

        if action in {0, 12}:
            energy_penalty = 0.0
        elif action > 12:
            energy_penalty = -2 * (60 * 0.87 * 1)
        else:
            energy_penalty = -(60 * 0.87 * 1)
            
        reward = 120.0 * comfort_penalty + energy_penalty
        rewards.append(reward)
        state = next_state

    return data_pre_test, actions, rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="data")
    parser.add_argument("--day-index", type=int, default=0)
    parser.add_argument("--xgb-device", default="cpu")
    parser.add_argument("--weights", default="artifacts/outputs/whulx_reproduction/whulx_dqn_reproduction.weights.h5")
    parser.add_argument("--plot-out", default="artifacts/outputs/whulx_reproduction/performance_comparison.png")
    args = parser.parse_args()

    print(f"Loading data and training environment model (XGBoost)...")
    raw, data = load_and_prepare_data(Path(args.repo_dir))
    model_xgb, _ = train_xgboost(data, device=args.xgb_device)

    print(f"Loading DQN agent weights from {args.weights}...")
    num_actions = 24
    q_network = DQN(num_actions)
    q_network(np.zeros((1, 8), dtype=np.float32))
    if os.path.exists(args.weights):
        q_network.load_weights(args.weights)
        print(f"Weights loaded successfully.")
    else:
        print(f"ERROR: Weights not found at {args.weights}.")
        sys.exit(1)

    print(f"Running rollouts for Day Index {args.day_index}...")
    data_test, xgboost_test = choose_day(args.day_index * 24, data)

    results = {}
    
    # 1. DQN Cải tiến (Nhóm)
    eval_day, actions, rewards = rollout_day_custom(model_xgb, q_network, data_test, xgboost_test, policy_type="improved")
    ac_status, window_status = [], []
    for act in actions:
        _, ac, win, _, _ = map_action_to_dataframe(act)
        ac_status.append(ac)
        window_status.append(win)
    results["DQN Cải tiến (Nhóm)"] = {
        "indoor_temp": eval_day.iloc[6:19]["Indoor_Temp"].values,
        "outdoor_temp": eval_day.iloc[6:19]["Outdoor_Temp"].values,
        "ac_status": np.array(ac_status),
        "window_status": np.array(window_status),
        "reward": rewards,
    }

    # 2. DQN Gốc (WHU-LX)
    eval_day, actions, rewards = rollout_day_custom(model_xgb, q_network, data_test, xgboost_test, policy_type="original")
    ac_status, window_status = [], []
    for act in actions:
        _, ac, win, _, _ = map_action_to_dataframe(act)
        ac_status.append(ac)
        window_status.append(win)
    results["DQN Gốc (WHU-LX)"] = {
        "indoor_temp": eval_day.iloc[6:19]["Indoor_Temp"].values,
        "outdoor_temp": eval_day.iloc[6:19]["Outdoor_Temp"].values,
        "ac_status": np.array(ac_status),
        "window_status": np.array(window_status),
        "reward": rewards,
    }

    # 3. Human
    ac_status = xgboost_test.iloc[6:18]["AC_Status"].values
    window_status = xgboost_test.iloc[6:18]["Window_Status"].values
    rewards = []
    for idx, step in enumerate(range(6, 18)):
        ac = ac_status[idx]
        win = window_status[idx]
        if ac == 0 and win == 0:
            penalty = 0.0
        elif ac == 1 and win == 1:
            penalty = -2 * (60 * 0.87 * 1)
        elif ac == 1:
            penalty = -(60 * 0.87 * 1)
        else:
            penalty = 0.0
        
        next_state = data_test.iloc[step + 1, :8].values
        indoor_temp = next_state[0]
        outdoor_temp = next_state[2]
        _, lower, upper, _ = comfort_bounds(outdoor_temp)
        if lower <= indoor_temp <= upper:
            comfort_penalty = 0.0
        elif indoor_temp < lower:
            comfort_penalty = -((indoor_temp - lower) ** 2)
        else:
            comfort_penalty = -((indoor_temp - upper) ** 2)
        rewards.append(120.0 * comfort_penalty + penalty)
        
    results["Human (Thực tế)"] = {
        "indoor_temp": data_test.iloc[6:19]["Indoor_Temp"].values,
        "outdoor_temp": data_test.iloc[6:19]["Outdoor_Temp"].values,
        "ac_status": ac_status,
        "window_status": window_status,
        "reward": rewards,
    }

    # Plotting
    hours = np.arange(6, 18)
    hours_temp = np.arange(6, 19)
    
    plt.figure(figsize=(12, 10))
    
    colors = {
        "DQN Cải tiến (Nhóm)": "blue",
        "DQN Gốc (WHU-LX)": "red",
        "Human (Thực tế)": "green"
    }
    styles = {
        "DQN Cải tiến (Nhóm)": "-",
        "DQN Gốc (WHU-LX)": "--",
        "Human (Thực tế)": "-."
    }
    
    # 1. Indoor Temperature
    plt.subplot(3, 1, 1)
    for name in results:
        plt.plot(hours_temp, results[name]["indoor_temp"], label=name, color=colors[name], linestyle=styles[name], linewidth=2.5)
    plt.plot(hours_temp, results["DQN Cải tiến (Nhóm)"]["outdoor_temp"], label="Outdoor Temp", color="black", alpha=0.3, linestyle=":")
    
    # Comfort limits
    lower_lims, upper_lims = [], []
    for step in range(6, 19):
        o_temp = data_test.iloc[step]["Outdoor_Temp"]
        _, lower, upper, _ = comfort_bounds(o_temp)
        lower_lims.append(lower)
        upper_lims.append(upper)
    plt.plot(hours_temp, lower_lims, label="Lower Comfort Bound (ASHRAE)", color="green", linestyle="--", alpha=0.5)
    plt.plot(hours_temp, upper_lims, label="Upper Comfort Bound (ASHRAE)", color="green", linestyle="--", alpha=0.5)
    
    plt.title(f"So sánh Nhiệt độ Phòng (Day Index {args.day_index})", fontsize=13, fontweight="bold")
    plt.ylabel("Nhiệt độ (°C)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", ncol=2, fontsize=10)
    
    # 2. AC Status / Energy Equivalent
    plt.subplot(3, 1, 2)
    for name in results:
        penalties = []
        for idx in range(12):
            ac = results[name]["ac_status"][idx]
            win = results[name]["window_status"][idx]
            if ac == 0 and win == 0:
                pen = 0.0
            elif ac == 1 and win == 1:
                pen = 2 * (60 * 0.87 * 1)
            elif ac == 1:
                pen = (60 * 0.87 * 1)
            else:
                pen = 0.0
            penalties.append(pen)
        plt.plot(hours, penalties, label=name, color=colors[name], linestyle=styles[name], drawstyle="steps-post", linewidth=2.5)
    plt.title("Tiêu thụ Năng lượng Tương đương (Energy Penalty Equivalent)", fontsize=13, fontweight="bold")
    plt.ylabel("Energy Penalty (W-min equivalent)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10)
    
    # 3. Comfort Deviation
    plt.subplot(3, 1, 3)
    for name in results:
        deviations = []
        temps = results[name]["indoor_temp"][1:]
        for idx, temp in enumerate(temps):
            o_temp = results[name]["outdoor_temp"][idx+1]
            _, lower, upper, _ = comfort_bounds(o_temp)
            if lower <= temp <= upper:
                dev = 0.0
            elif temp < lower:
                dev = temp - lower
            else:
                dev = temp - upper
            deviations.append(dev)
        plt.plot(hours, deviations, label=name, color=colors[name], linestyle=styles[name], linewidth=2.5)
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.title("Độ lệch ngoài Vùng tiện nghi (Comfort Deviation)", fontsize=13, fontweight="bold")
    plt.ylabel("Độ lệch (°C)", fontsize=11)
    plt.xlabel("Giờ trong ngày (Hour of Day)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)
    
    plt.tight_layout()
    plot_out = Path(args.plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_out, dpi=200)
    plt.close()
    print(f"Saved WHU-LX performance comparison to {plot_out}")

if __name__ == "__main__":
    main()
