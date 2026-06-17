import argparse
import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def comfort_bounds(outdoor_temp):
    # ASHRAE standard bands
    l11, l12, u12, u11 = [17.4, 18.4, 23.4, 24.4]
    l21, l22, u22, u21 = [23.6, 24.6, 29.6, 30.6]
    t1, t2 = [10, 30]
    if outdoor_temp <= t1:
        return l11, l12, u12, u11
    if outdoor_temp >= t2:
        return l21, l22, u22, u21
    increase_l = (outdoor_temp - t1) * (l21 - l11) / (t2 - t1)
    increase_u = (outdoor_temp - t1) * (u21 - u11) / (t2 - t1)
    return l11 + increase_l, l12 + increase_l, u12 + increase_u, u11 + increase_u


def comfort_ok(indoor_temp, outdoor_temp):
    _, lower, upper, _ = comfort_bounds(float(outdoor_temp))
    return lower <= float(indoor_temp) <= upper


def calculate_reward_original(state, action, next_state):
    """Original reward calculation with comfort_weight = 1.0"""
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

    # original comfort weight is 1.0
    reward = 1.0 * comfort_penalty + energy_penalty
    return float(reward)


def map_action_to_dataframe(action):
    action = int(action)
    target_temp, ac_status, window_status, c_last_time, w_last_time = 0, 0, 0, 0, 0
    if action == 0:
        pass
    elif 0 < action < 12:
        target_temp = 19 + action
        ac_status = 1
        c_last_time = 60
    elif action == 12:
        window_status = 1
        w_last_time = 60
    else:
        target_temp = action + 7
        ac_status = 1
        window_status = 1
        c_last_time = 60
        w_last_time = 60
    return target_temp, ac_status, window_status, c_last_time, w_last_time


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.buffer.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.output_layer = tf.keras.layers.Dense(num_actions, activation="linear")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


def update_q_network_original(q_network, replay_buffer, optimizer, loss_fn, gamma, num_actions, batch_size):
    """Updates using Q_network for next states (bug in original paper) and NO normalization"""
    states, actions, next_states, rewards = zip(*replay_buffer.sample(batch_size))
    
    # NO state normalization
    states_tensor = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
    next_states_tensor = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
    rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)

    with tf.GradientTape() as tape:
        q_values = q_network(states_tensor)
        # Bug: Using q_network instead of target_q_network
        target_q_values = q_network(next_states_tensor)
        target_q_values = rewards + gamma * tf.reduce_max(target_q_values, axis=1)
        mask = tf.one_hot(actions, num_actions)
        q_action = tf.reduce_sum(q_values * mask, axis=1)
        loss = loss_fn(target_q_values, q_action)

    grads = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
    return float(loss.numpy())


def epsilon_greedy_policy_original(q_network, state, epsilon, num_actions):
    if np.random.rand() < epsilon:
        return int(np.random.randint(num_actions))
    # NO state normalization
    q_values = q_network(np.array([state], dtype=np.float32))
    return int(np.argmax(q_values[0]))


def load_and_prepare_data(repo_dir):
    raw = pd.read_csv(repo_dir / "Cleaned_data.csv", encoding="gbk")
    data = raw.copy()
    data["Date_Time"] = pd.to_datetime(data["Date_Time"])
    data = data.dropna()
    data = data[data != -999].dropna()
    for col in data.columns:
        if data[col].dtype == "object":
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])
    return raw, data


def train_xgboost(data, device="cpu"):
    x_data = data.drop(
        ["Next_Indoor_Temp", "Next_Indoor_RH", "Date_Time", "Study_ID", "Differ_Indoor_Temp", "ID"],
        axis=1,
    )
    y_data = data["Differ_Indoor_Temp"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=2022
    )
    model = xgb.XGBRegressor(
        random_state=2000,
        verbosity=0,
        n_jobs=-1,
        tree_method="hist",
        device=device,
        max_depth=5,
        learning_rate=0.23474,
        n_estimators=500,
    )
    model.fit(x_train, y_train)
    return model


def choose_day(start_index, data):
    data_test_00 = data.iloc[start_index : start_index + 23].copy()
    data_test_0a = data_test_00.reset_index(drop=True)
    data_test_0 = data_test_00.reset_index(drop=True)
    data_test_0.at[0, "CLast_Time_T"] = (
        data_test_0a.iloc[0]["CLast_Time_T"] - data_test_0a.iloc[0]["AC_Status"] * 60
    )
    data_test_0.at[0, "WLast_Time_T"] = (
        data_test_0a.iloc[0]["WLast_Time_T"] - data_test_0a.iloc[0]["Window_Status"] * 60
    )
    data_test = data_test_0[
        [
            "Indoor_Temp",
            "Indoor_RH",
            "Outdoor_Temp",
            "Outdoor_RH",
            "Rain",
            "Cloud",
            "Windspeed",
            "Hour",
            "Next_Outdoor_Temp",
            "Next_Outdoor_RH",
        ]
    ].copy()
    xgboost_test = data_test_0.drop(
        ["Next_Indoor_Temp", "Next_Indoor_RH", "Date_Time", "Study_ID", "Differ_Indoor_Temp", "ID"],
        axis=1,
    )
    return data_test, xgboost_test


def rollout_day_original(
    model_xgb,
    q_network,
    data_test,
    xgboost_test,
    epsilon,
    train=False,
    replay_buffer=None,
    optimizer=None,
    loss_fn=None,
    gamma=0.9,
    num_actions=24,
    batch_size=32,
):
    data_pre_test = data_test.copy()
    xgboost_pre_test = xgboost_test.copy()
    num_features = 8
    actions = []
    rewards = []
    losses = []

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
        action = epsilon_greedy_policy_original(q_network, state, epsilon, num_actions)
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
        reward = calculate_reward_original(state, action, next_state)
        rewards.append(reward)

        if train and replay_buffer is not None:
            replay_buffer.push(state, action, next_state, reward)
            if len(replay_buffer) >= batch_size:
                losses.append(
                    update_q_network_original(
                        q_network,
                        replay_buffer,
                        optimizer,
                        loss_fn,
                        gamma,
                        num_actions,
                        batch_size,
                    )
                )
        state = next_state

    return data_pre_test, actions, rewards, losses


def summarize_day(data_day, actions):
    control = data_day.iloc[6:18].copy()
    comfort = [
        comfort_ok(row["Indoor_Temp"], row["Outdoor_Temp"])
        for _, row in control.iterrows()
    ]
    ac_on = sum(1 for a in actions if 0 < a < 12 or a > 12)
    window_open = sum(1 for a in actions if a >= 12)
    return {
        "comfort_pct": float(np.mean(comfort) * 100),
        "ac_on_pct": float(ac_on / len(actions) * 100) if actions else 0.0,
        "window_open_pct": float(window_open / len(actions) * 100) if actions else 0.0,
        "mean_indoor_temp": float(control["Indoor_Temp"].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="data")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--xgb-device", default="cpu")
    parser.add_argument("--out-dir", default="artifacts/outputs/whulx_original_full_days")
    args = parser.parse_args()

    np.random.seed(2022)
    random.seed(2022)
    tf.random.set_seed(2022)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    raw, data = load_and_prepare_data(Path(args.repo_dir))
    
    print("Training XGBoost environment surrogate...")
    model_xgb = train_xgboost(data, device=args.xgb_device)

    num_actions = 24
    q_network = DQN(num_actions)
    q_network(np.zeros((1, 8), dtype=np.float32))

    replay_buffer = ReplayBuffer(10000)
    optimizer = tf.optimizers.Adam(0.001)
    loss_fn = tf.losses.MeanSquaredError()
    epsilon = 1.0
    min_epsilon = 0.1
    epsilon_decay = 0.995
    gamma = 0.9

    max_complete_days = max(1, (len(data) - 24) // 24)
    print(f"Starting Multi-day Training with ORIGINAL buggy config for {args.episodes} episodes...")

    for episode in range(args.episodes):
        # Choose a random day for training
        train_day_index = random.randint(0, max_complete_days - 1)
        train_data_test, train_xgboost_test = choose_day(train_day_index * 24, data)

        _, actions, rewards, losses = rollout_day_original(
            model_xgb,
            q_network,
            train_data_test,
            train_xgboost_test,
            epsilon,
            train=True,
            replay_buffer=replay_buffer,
            optimizer=optimizer,
            loss_fn=loss_fn,
            gamma=gamma,
            num_actions=num_actions,
            batch_size=32,
        )

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{args.episodes}, last reward: {np.sum(rewards):.2f}, epsilon: {epsilon:.3f}")

    print("Evaluating trained ORIGINAL agent on all 1763 days...")
    eval_comforts = []
    eval_ac_ons = []
    eval_window_opens = []
    eval_rewards = []

    for day_idx in range(max_complete_days):
        data_test, xgboost_test = choose_day(day_idx * 24, data)
        eval_day, actions, rewards, _ = rollout_day_original(
            model_xgb,
            q_network,
            data_test,
            xgboost_test,
            epsilon=0.0,
            train=False,
            num_actions=num_actions,
        )
        metrics = summarize_day(eval_day, actions)
        eval_comforts.append(metrics["comfort_pct"])
        eval_ac_ons.append(metrics["ac_on_pct"])
        eval_window_opens.append(metrics["window_open_pct"])
        eval_rewards.append(np.sum(rewards))

    summary = {
        "controller": "DQN Gốc chạy Full Ngày (WHU-LX Original Full Days)",
        "comfort_pct": float(np.mean(eval_comforts)),
        "ac_on_pct": float(np.mean(eval_ac_ons)),
        "window_open_pct": float(np.mean(eval_window_opens)),
        "total_reward": float(np.mean(eval_rewards)),
    }

    print("\n=== EVALUATION RESULTS ===")
    print(json.dumps(summary, indent=2))

    (out_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    q_network.save_weights(out_dir / "whulx_original_full_days.weights.h5")
    print("Done!")


if __name__ == "__main__":
    main()
