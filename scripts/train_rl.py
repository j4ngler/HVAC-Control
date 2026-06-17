import argparse
import os
import random
import sys
from collections import deque
from datetime import date, timedelta
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import sinergym
import tensorflow as tf
from pythermalcomfort.models import pmv_ppd_iso

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


if "EPLUS_PATH" not in os.environ:
    os.environ["EPLUS_PATH"] = "/usr/local/EnergyPlus-25-2-0"
if os.environ["EPLUS_PATH"] not in sys.path:
    sys.path.insert(0, os.environ["EPLUS_PATH"])


STATE_KEYS = [
    "month",
    "hour",
    "outdoor_temperature",
    "outdoor_humidity",
    "air_temperature",
    "air_humidity",
    "people_occupant",
    "HVAC_electricity_demand_rate",
]

STATE_DIM = 8
POWER_SCALE_W = 15000.0

ACTION_HEATING_SETPOINTS = [21.0]
ACTION_COOLING_SETPOINTS = [25.0, 25.5, 26.0, 26.5, 27.0]
ACTION_GRID = [
    (heating, cooling)
    for heating in ACTION_HEATING_SETPOINTS
    for cooling in ACTION_COOLING_SETPOINTS
]


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

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


class DQNAgent:
    def __init__(
        self,
        state_dim,
        num_actions,
        learning_rate=0.001,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = DQN(num_actions)
        self.target_model = DQN(num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        dummy_state = tf.zeros((1, state_dim), dtype=tf.float32)
        self.model(dummy_state)
        self.target_model(dummy_state)
        self.update_target_network()

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        state = np.array([state], dtype=np.float32)
        q_values = self.model(state)
        return int(np.argmax(q_values[0]))

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        next_q_values = self.target_model(next_states)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        target_q = rewards + (1 - dones) * self.gamma * max_next_q

        with tf.GradientTape() as tape:
            current_q_values = self.model(states)
            masks = tf.one_hot(actions, self.num_actions)
            current_q = tf.reduce_sum(current_q_values * masks, axis=1)
            loss = self.loss_fn(target_q, current_q)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.numpy())


def map_action(action_idx, is_tropical=True):
    heating, cooling = ACTION_GRID[int(action_idx) % len(ACTION_GRID)]
    if is_tropical:
        heating = 15.0
    return [heating, cooling]


def build_indices(obs_names):
    missing = [name for name in STATE_KEYS if name not in obs_names]
    if missing:
        raise KeyError(f"Missing observation variables: {missing}. Available: {obs_names}")
    return {name: obs_names.index(name) for name in STATE_KEYS}


def normalize_state_values(values):
    month, hour, outdoor_temp, outdoor_rh, indoor_temp, indoor_rh, occupancy, power = values
    return np.array(
        [
            float(month) / 12.0,
            float(hour) / 23.0,
            float(outdoor_temp) / 40.0,
            float(outdoor_rh) / 100.0,
            float(indoor_temp) / 40.0,
            float(indoor_rh) / 100.0,
            float(occupancy) / 20.0,
            float(power) / POWER_SCALE_W,
        ],
        dtype=np.float32,
    )


def state_from_obs(obs, indices):
    values = [obs[indices[name]] for name in STATE_KEYS]
    return normalize_state_values(values)


def clothing_by_month(month, is_tropical=True):
    if is_tropical:
        return 0.5
    month = int(month)
    return 0.8 if month in {11, 12, 1, 2} else 0.5


def calculate_pmv(temp, rh, month, air_velocity=0.1, met_rate=1.2, is_tropical=True):
    clo = clothing_by_month(month, is_tropical=is_tropical)
    try:
        result = pmv_ppd_iso(
            tdb=float(temp),
            tr=float(temp),
            vr=float(air_velocity),
            rh=float(rh),
            met=float(met_rate),
            clo=float(clo),
            limit_inputs=False,
        )
        pmv = float(result.pmv)
    except Exception:
        pmv = (temp - 23.0) * 0.5 + (rh - 50.0) * 0.01
    return float(np.clip(pmv, -3.0, 3.0))


def shape_reward(
    raw_reward,
    power,
    temp,
    rh,
    month,
    occupancy,
    comfort_threshold=0.5,
    comfort_weight=1.5,
    violation_weight=3.0,
    energy_weight=0.10,
    is_tropical=True,
):
    pmv = calculate_pmv(temp, rh, month, is_tropical=is_tropical)
    if occupancy > 0:
        comfort_penalty = abs(pmv)
        violation_penalty = max(0.0, abs(pmv) - comfort_threshold) ** 2
        overcool_penalty = max(0.0, -pmv - comfort_threshold) ** 2
    else:
        comfort_penalty = 0.0
        violation_penalty = 0.0
        overcool_penalty = 0.0
    energy_penalty = power / POWER_SCALE_W
    shaped_reward = raw_reward - comfort_weight * comfort_penalty - energy_weight * energy_penalty
    shaped_reward -= violation_weight * violation_penalty
    shaped_reward -= 0.5 * violation_weight * overcool_penalty
    return shaped_reward, pmv, comfort_penalty + violation_penalty + overcool_penalty, energy_penalty


def make_runperiod(start_month, end_month, start_day=1, end_day=30):
    return {
        "runperiod": (
            int(start_day),
            int(start_month),
            1991,
            int(end_day),
            int(end_month),
            1991,
        )
    }


def random_window_runperiod(rng, start_month, end_month, start_day, end_day, window_days):
    start = date(1991, int(start_month), int(start_day))
    end = date(1991, int(end_month), int(end_day))
    latest = end - timedelta(days=int(window_days) - 1)
    if latest < start:
        return make_runperiod(start_month, end_month, start_day, end_day)
    offset = int(rng.integers(0, (latest - start).days + 1))
    window_start = start + timedelta(days=offset)
    window_end = window_start + timedelta(days=int(window_days) - 1)
    return make_runperiod(
        window_start.month,
        window_end.month,
        window_start.day,
        window_end.day,
    )


def plot_training_history(history, plot_out_path):
    df = pd.DataFrame(history)
    if df.empty:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Total Reward & Raw Reward
    axes[0, 0].plot(df["episode"], df["total_reward"], label="Total Reward", color="blue")
    axes[0, 0].plot(df["episode"], df["raw_total_reward"], label="Raw Reward", color="orange", linestyle="--")
    axes[0, 0].set_title("Training Reward over Episodes")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Avg Loss
    axes[0, 1].plot(df["episode"], df["avg_loss"], color="red")
    axes[0, 1].set_title("Average DQN Loss over Episodes")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)
    
    # 3. Avg Power (HVAC electricity)
    axes[1, 0].plot(df["episode"], df["avg_power"], color="green")
    axes[1, 0].set_title("Average HVAC Power Demand over Episodes")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Power (W)")
    axes[1, 0].grid(True)
    
    # 4. Avg PMV abs (Comfort)
    axes[1, 1].plot(df["episode"], df["avg_pmv_abs"], color="purple")
    axes[1, 1].set_title("Average PMV absolute over Episodes")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("abs(PMV)")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_out_path = Path(plot_out_path)
    plot_out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_out_path)
    plt.close()
    print(f"Training history plots saved to {plot_out_path}")


def train(
    episodes,
    max_steps,
    weather_file,
    model_out,
    history_out,
    batch_size=32,
    learning_rate=0.001,
    seed=42,
    comfort_threshold=0.5,
    comfort_weight=1.5,
    violation_weight=3.0,
    energy_weight=0.10,
    epsilon_decay=0.9999,
    random_window_days=None,
    start_month=None,
    end_month=None,
    start_day=1,
    end_day=30,
    is_tropical=True,
    plot_out=None,
):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env_name = "Eplus-5zone-mixed-continuous-v1"
    weather_file_abs = os.path.abspath(weather_file)
    base_config_params = None
    if start_month is not None and end_month is not None:
        base_config_params = make_runperiod(start_month, end_month, start_day, end_day)

    env = None
    obs_names = None
    indices = None

    state_dim = STATE_DIM
    num_actions = len(ACTION_GRID)
    agent = DQNAgent(
        state_dim,
        num_actions,
        learning_rate=learning_rate,
        epsilon_decay=epsilon_decay,
    )
    replay_buffer = ReplayBuffer(10000)
    history = []
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        config_params = base_config_params
        if random_window_days and start_month is not None and end_month is not None:
            config_params = random_window_runperiod(
                rng,
                start_month,
                end_month,
                start_day,
                end_day,
                random_window_days,
            )
            if env is not None:
                env.close()
            env = None

        if env is None:
            env = gym.make(env_name, weather_files=[weather_file_abs], config_params=config_params)
            obs_names = list(env.unwrapped.observation_variables)
            indices = build_indices(obs_names)

        obs, info = env.reset(seed=seed + ep)
        state = state_from_obs(obs, indices)
        ep_reward = 0.0
        ep_raw_reward = 0.0
        ep_losses = []
        ep_power = []
        ep_indoor_temp = []
        ep_pmv_abs = []
        ep_comfort_penalty = []
        ep_energy_penalty = []

        for step in range(max_steps):
            action_idx = agent.act(state)
            action = np.array(map_action(action_idx, is_tropical=is_tropical), dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_obs, raw_reward, terminated, truncated, info = env.step(action)
            next_state = state_from_obs(next_obs, indices)
            done = terminated or truncated
            power = float(next_obs[indices["HVAC_electricity_demand_rate"]])
            indoor_temp = float(next_obs[indices["air_temperature"]])
            indoor_rh = float(next_obs[indices["air_humidity"]])
            month = float(next_obs[indices["month"]])
            occupancy = float(next_obs[indices["people_occupant"]])
            reward, pmv, comfort_penalty, energy_penalty = shape_reward(
                raw_reward=float(raw_reward),
                power=power,
                temp=indoor_temp,
                rh=indoor_rh,
                month=month,
                occupancy=occupancy,
                comfort_threshold=comfort_threshold,
                comfort_weight=comfort_weight,
                violation_weight=violation_weight,
                energy_weight=energy_weight,
                is_tropical=is_tropical,
            )

            replay_buffer.push(state, action_idx, reward, next_state, done)
            loss = agent.train(replay_buffer, batch_size)
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            ep_reward += float(reward)
            ep_raw_reward += float(raw_reward)
            ep_power.append(power)
            ep_indoor_temp.append(indoor_temp)
            ep_pmv_abs.append(abs(pmv))
            ep_comfort_penalty.append(comfort_penalty)
            ep_energy_penalty.append(energy_penalty)

            if done:
                break

        agent.update_target_network()
        row = {
            "episode": ep + 1,
            "steps": step + 1,
            "runperiod": str(config_params["runperiod"]) if config_params else "default",
            "total_reward": ep_reward,
            "raw_total_reward": ep_raw_reward,
            "epsilon": agent.epsilon,
            "avg_loss": float(np.mean(ep_losses)) if ep_losses else np.nan,
            "avg_power": float(np.mean(ep_power)) if ep_power else np.nan,
            "avg_indoor_temp": float(np.mean(ep_indoor_temp)) if ep_indoor_temp else np.nan,
            "avg_pmv_abs": float(np.mean(ep_pmv_abs)) if ep_pmv_abs else np.nan,
            "avg_comfort_penalty": float(np.mean(ep_comfort_penalty)) if ep_comfort_penalty else np.nan,
            "avg_energy_penalty": float(np.mean(ep_energy_penalty)) if ep_energy_penalty else np.nan,
        }
        history.append(row)
        print(
            f"Episode {ep + 1}/{episodes}, "
            f"Reward: {ep_reward:.2f}, "
            f"Raw Reward: {ep_raw_reward:.2f}, "
            f"Epsilon: {agent.epsilon:.3f}, "
            f"Avg Power: {row['avg_power']:.2f} W, "
            f"Avg PMV abs: {row['avg_pmv_abs']:.2f}"
        )

    if env is not None:
        env.close()

    model_out = Path(model_out)
    history_out = Path(history_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    history_out.parent.mkdir(parents=True, exist_ok=True)
    agent.model.save_weights(model_out)
    pd.DataFrame(history).to_csv(history_out, index=False)
    print(f"Model saved to {model_out}")
    print(f"Training history saved to {history_out}")

    if plot_out:
        plot_training_history(history, plot_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--weather", type=str, required=True)
    parser.add_argument("--model-out", type=str, default="models/dqn_hvac_vietnam.weights.h5")
    parser.add_argument("--history-out", type=str, default="outputs/training_history.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comfort-threshold", type=float, default=0.5)
    parser.add_argument("--comfort-weight", type=float, default=1.5)
    parser.add_argument("--violation-weight", type=float, default=3.0)
    parser.add_argument("--energy-weight", type=float, default=0.10)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999)
    parser.add_argument("--random-window-days", type=int, default=None)
    parser.add_argument("--start-month", type=int, default=None)
    parser.add_argument("--end-month", type=int, default=None)
    parser.add_argument("--start-day", type=int, default=1)
    parser.add_argument("--end-day", type=int, default=30)
    parser.add_argument("--tropical", action="store_true", default=True, help="Use Clo=0.5 and Heating=15.0 for HCMC climate")
    parser.add_argument("--no-tropical", action="store_false", dest="tropical", help="Disable tropical mode")
    parser.add_argument("--plot-out", type=str, default=None, help="Path to save training history plots")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        max_steps=args.steps,
        weather_file=args.weather,
        model_out=args.model_out,
        history_out=args.history_out,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        comfort_threshold=args.comfort_threshold,
        comfort_weight=args.comfort_weight,
        violation_weight=args.violation_weight,
        energy_weight=args.energy_weight,
        epsilon_decay=args.epsilon_decay,
        random_window_days=args.random_window_days,
        start_month=args.start_month,
        end_month=args.end_month,
        start_day=args.start_day,
        end_day=args.end_day,
        is_tropical=args.tropical,
        plot_out=args.plot_out,
    )
