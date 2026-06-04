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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


STANDARD_BANDS = {
    "ASHRAE": [[17.4, 18.4, 23.4, 24.4], [23.6, 24.6, 29.6, 30.6], [10, 30]]
}


def comfort_bounds(outdoor_temp, standard="ASHRAE"):
    l11, l12, u12, u11 = STANDARD_BANDS[standard][0]
    l21, l22, u22, u21 = STANDARD_BANDS[standard][1]
    t1, t2 = STANDARD_BANDS[standard][2]
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


def calculate_reward(state, action, next_state):
    indoor_temp = float(next_state[0])
    outdoor_temp = float(next_state[2])
    _, lower, upper, _ = comfort_bounds(outdoor_temp)

    if lower <= indoor_temp <= upper:
        reward = 0.0
    elif indoor_temp < lower:
        reward = -((indoor_temp - lower) ** 2)
    else:
        reward = -((indoor_temp - upper) ** 2)

    if action in {0, 12}:
        reward += 0.0
    elif action > 12:
        reward -= 2 * (60 * 0.87 * 1)
    else:
        reward -= 60 * 0.87 * 1
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


def update_q_network(q_network, replay_buffer, optimizer, loss_fn, gamma, num_actions, batch_size):
    states, actions, next_states, rewards = zip(*replay_buffer.sample(batch_size))
    states = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
    rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)

    with tf.GradientTape() as tape:
        q_values = q_network(states)
        target_q_values = q_network(next_states)
        target_q_values = rewards + gamma * tf.reduce_max(target_q_values, axis=1)
        mask = tf.one_hot(actions, num_actions)
        q_action = tf.reduce_sum(q_values * mask, axis=1)
        loss = loss_fn(target_q_values, q_action)

    grads = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
    return float(loss.numpy())


def epsilon_greedy_policy(q_network, state, epsilon, num_actions):
    if np.random.rand() < epsilon:
        return int(np.random.randint(num_actions))
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
    pred = model.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    metrics = {
        "x_shape": list(x_data.shape),
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_test, pred)),
    }
    return model, metrics


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


def rollout_day(
    model_xgb,
    q_network,
    data_test,
    xgboost_test,
    epsilon,
    train=False,
    replay_buffer=None,
    train_cfg=None,
    fixed_action=None,
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
        if fixed_action is None:
            action = epsilon_greedy_policy(q_network, state, epsilon, train_cfg["num_actions"])
        else:
            action = int(fixed_action)
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
        reward = calculate_reward(state, action, next_state)
        rewards.append(reward)

        if train and replay_buffer is not None:
            replay_buffer.push(state, action, next_state, reward)
            if len(replay_buffer) >= train_cfg["batch_size"]:
                losses.append(
                    update_q_network(
                        q_network,
                        replay_buffer,
                        train_cfg["optimizer"],
                        train_cfg["loss_fn"],
                        train_cfg["gamma"],
                        train_cfg["num_actions"],
                        train_cfg["batch_size"],
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


def summarize_human(data_test):
    control = data_test.iloc[6:18].copy()
    comfort = [
        comfort_ok(row["Indoor_Temp"], row["Outdoor_Temp"])
        for _, row in control.iterrows()
    ]
    return {
        "comfort_pct": float(np.mean(comfort) * 100),
        "mean_indoor_temp": float(control["Indoor_Temp"].mean()),
    }


def summarize_human_with_actions(data_test, xgboost_test):
    metrics = summarize_human(data_test)
    control = xgboost_test.iloc[6:18].copy()
    metrics["ac_on_pct"] = float(control["AC_Status"].mean() * 100)
    metrics["window_open_pct"] = float(control["Window_Status"].mean() * 100)
    return metrics


def aggregate_metric_rows(rows):
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row if isinstance(row.get(key), (int, float, np.number))})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def evaluate_many_days(model_xgb, q_network, data, day_indices, num_actions):
    controllers = {
        "DQN": None,
        "Human": "human",
        "Off_Closed": 0,
        "Window_Open": 12,
        "AC_25_Closed": 6,
        "AC_26_Closed": 7,
        "AC_27_Closed": 8,
    }
    rows_by_controller = {name: [] for name in controllers}
    actions_by_controller = {name: [] for name in controllers}

    for day_index in day_indices:
        data_test, xgboost_test = choose_day(day_index * 24, data)
        for name, fixed_action in controllers.items():
            if fixed_action == "human":
                rows_by_controller[name].append(summarize_human_with_actions(data_test, xgboost_test))
                continue

            eval_day, actions, rewards, _ = rollout_day(
                model_xgb,
                q_network,
                data_test,
                xgboost_test,
                epsilon=0.0,
                train=False,
                train_cfg={"num_actions": num_actions},
                fixed_action=fixed_action,
            )
            metrics = summarize_day(eval_day, actions)
            metrics["total_reward"] = float(np.sum(rewards))
            rows_by_controller[name].append(metrics)
            actions_by_controller[name].extend(int(action) for action in actions)

    summary = []
    action_distributions = {}
    for name, rows in rows_by_controller.items():
        metrics = aggregate_metric_rows(rows)
        summary.append({"controller": name, **metrics})
        actions = actions_by_controller.get(name, [])
        if actions:
            counts = pd.Series(actions).value_counts().sort_index()
            action_distributions[name] = {
                int(action): float(count / len(actions) * 100)
                for action, count in counts.items()
            }
    return summary, action_distributions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="data")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--day-index", type=int, default=0)
    parser.add_argument("--eval-days", type=int, default=0, help="0 means evaluate all complete days.")
    parser.add_argument("--xgb-device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--out-dir", default="artifacts/outputs/whulx_reproduction")
    args = parser.parse_args()

    np.random.seed(2022)
    random.seed(2022)
    tf.random.set_seed(2022)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, data = load_and_prepare_data(Path(args.repo_dir))
    model_xgb, xgb_metrics = train_xgboost(data, device=args.xgb_device)

    data_test, xgboost_test = choose_day(args.day_index * 24, data)

    num_actions = 24
    q_network = DQN(num_actions)
    q_network(np.zeros((1, 8), dtype=np.float32))
    replay_buffer = ReplayBuffer(10000)
    optimizer = tf.optimizers.Adam(0.001)
    loss_fn = tf.losses.MeanSquaredError()
    epsilon = 1.0
    min_epsilon = 0.1
    epsilon_decay = 0.995

    train_cfg = {
        "num_actions": num_actions,
        "batch_size": 32,
        "gamma": 0.9,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
    }
    history = []
    for episode in range(args.episodes):
        _, actions, rewards, losses = rollout_day(
            model_xgb,
            q_network,
            data_test,
            xgboost_test,
            epsilon,
            train=True,
            replay_buffer=replay_buffer,
            train_cfg=train_cfg,
        )
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
        history.append(
            {
                "episode": episode + 1,
                "reward": float(np.sum(rewards)),
                "epsilon": float(epsilon),
                "avg_loss": float(np.mean(losses)) if losses else np.nan,
            }
        )
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{args.episodes}, reward={np.sum(rewards):.2f}, epsilon={epsilon:.3f}")

    eval_day, eval_actions, eval_rewards, _ = rollout_day(
        model_xgb,
        q_network,
        data_test,
        xgboost_test,
        epsilon=0.0,
        train=False,
        train_cfg={"num_actions": num_actions},
    )
    human_metrics = summarize_human(data_test)
    dqn_metrics = summarize_day(eval_day, eval_actions)

    action_counts = pd.Series(eval_actions).value_counts().sort_index()
    action_distribution = {
        int(action): float(count / len(eval_actions) * 100)
        for action, count in action_counts.items()
    }
    result = {
        "episodes": args.episodes,
        "day_index": args.day_index,
        "raw_shape": list(raw.shape),
        "after_clean_shape": list(data.shape),
        "xgboost": xgb_metrics,
        "human_baseline": human_metrics,
        "dqn": {
            **dqn_metrics,
            "total_reward": float(np.sum(eval_rewards)),
            "actions": [int(a) for a in eval_actions],
            "action_distribution_pct": action_distribution,
        },
        "reported_whulx_readme": {
            "comfort_duration_increase_pct": 24.0,
            "ac_usage_decrease_pct": 24.7,
        },
    }

    max_complete_days = max(1, (len(data) - 24) // 24)
    eval_days_count = max_complete_days if args.eval_days == 0 else min(args.eval_days, max_complete_days)
    eval_indices = list(range(eval_days_count))
    evaluation_summary, evaluation_action_distribution = evaluate_many_days(
        model_xgb,
        q_network,
        data,
        eval_indices,
        num_actions,
    )
    result["evaluation"] = {
        "eval_days": eval_days_count,
        "controllers": evaluation_summary,
        "action_distribution_pct": evaluation_action_distribution,
    }

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame(evaluation_summary).to_csv(out_dir / "summary_metrics.csv", index=False)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    q_network.save_weights(out_dir / "whulx_dqn_reproduction.weights.h5")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
