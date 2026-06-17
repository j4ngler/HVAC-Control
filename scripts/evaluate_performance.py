import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import gymnasium as gym
import sinergym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from pythermalcomfort.models import pmv_ppd_iso

# Ensure EnergyPlus environment
if 'EPLUS_PATH' not in os.environ:
    os.environ['EPLUS_PATH'] = '/usr/local/EnergyPlus-25-2-0'
if os.environ['EPLUS_PATH'] not in sys.path:
    sys.path.insert(0, os.environ['EPLUS_PATH'])

# Re-import training components
from train_rl import ACTION_GRID, DQN, DQNAgent, STATE_DIM, map_action, normalize_state_values

def clothing_by_month(month, is_tropical=True):
    if is_tropical:
        return 0.5
    month = int(month)
    return 0.8 if month in {11, 12, 1, 2} else 0.5


def calculate_pmv(temp, rh, month, air_velocity=0.1, met_rate=1.2, is_tropical=True):
    """PMV Fanger calculation using pythermalcomfort ISO 7730 implementation.

    Mean radiant temperature is approximated as air temperature and air velocity
    is fixed until those variables are exposed from EnergyPlus observations.
    """
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


def state_from_obs(obs, indices):
    return normalize_state_values([obs[indices[k]] for k in indices.keys()])


def rule_based_action(obs, indices, comfort_threshold, is_tropical=True):
    month = obs[indices["month"]]
    hour = obs[indices["hour"]]
    temp = obs[indices["i_temp"]]
    rh = obs[indices["i_rh"]]
    occ = obs[indices["occ"]]
    pmv = calculate_pmv(temp, rh, month, is_tropical=is_tropical)

    if occ <= 0 or hour < 8 or hour >= 18:
        cooling = 30.0
    elif pmv > comfort_threshold:
        cooling = 24.0
    elif pmv > 0.2:
        cooling = 25.0
    elif pmv < -comfort_threshold:
        cooling = 27.0
    else:
        cooling = 26.0
    heating = 15.0 if is_tropical else 21.0
    return [heating, cooling]


def schedule_based_action(obs, indices, is_tropical=True):
    hour = obs[indices["hour"]]
    occ = obs[indices["occ"]]
    if 8 <= hour < 18 and occ > 0:
        cooling = 25.0
    else:
        cooling = 30.0
    heating = 15.0 if is_tropical else 21.0
    return [heating, cooling]

def evaluate(
    episodes,
    weather_file,
    model_path,
    max_steps=1000,
    summary_out="summary_metrics.csv",
    comfort_threshold=0.5,
    start_month=None,
    end_month=None,
    start_day=1,
    end_day=30,
    plot_out="performance_comparison.png",
    is_tropical=True,
    time_series_out="artifacts/outputs/hcm_summer/evaluation_time_series.csv",
):
    env_name = 'Eplus-5zone-mixed-continuous-v1'
    weather_file_abs = os.path.abspath(weather_file)
    config_params = None
    if start_month is not None and end_month is not None:
        config_params = {
            "runperiod": (
                int(start_day),
                int(start_month),
                1991,
                int(end_day),
                int(end_month),
                1991,
            )
        }
    env = gym.make(env_name, weather_files=[weather_file_abs], config_params=config_params)
    
    obs_names = list(env.unwrapped.observation_variables)
    indices = {
        'month': obs_names.index('month'),
        'hour': obs_names.index('hour'),
        'o_temp': obs_names.index('outdoor_temperature'),
        'o_rh': obs_names.index('outdoor_humidity'),
        'i_temp': obs_names.index('air_temperature'),
        'i_rh': obs_names.index('air_humidity'),
        'occ': obs_names.index('people_occupant'),
        'power': obs_names.index('HVAC_electricity_demand_rate')
    }
    
    state_dim = STATE_DIM
    num_actions = len(ACTION_GRID)
    agent = DQNAgent(state_dim, num_actions)
    if os.path.exists(model_path):
        agent.model(tf.zeros((1, state_dim), dtype=tf.float32))
        agent.model.load_weights(model_path)
        print(f"Loaded model from {model_path}")
    
    results = {}
    heating_val = 15.0 if is_tropical else 21.0
    ref_baseline = "Baseline_Fix_25"
    controllers = {
        "RL": None,
        "Baseline_Fix_25": [heating_val, 25.0],
        "Baseline_Fix_26": [heating_val, 26.0],
        "Baseline_Fix_27": [heating_val, 27.0],
        "Schedule_Based": "schedule",
        "Rule_Based": "rule",
    }

    for name, controller in controllers.items():
        print(f"\nRunning Evaluation: {name}...")
        results[name] = {
            'energy': [],
            'indoor_temp': [],
            'outdoor_temp': [],
            'pmv': [],
            'reward': [],
            'occ': [],
            'month': [],
            'rh': [],
            'hour': [],
        }
        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            step = 0
            while not done and step < max_steps:
                if name == "RL":
                    state = state_from_obs(obs, indices)
                    old_eps = agent.epsilon
                    agent.epsilon = 0
                    action_idx = agent.act(state)
                    agent.epsilon = old_eps
                    action = map_action(action_idx, is_tropical=is_tropical)
                elif controller == "rule":
                    action = rule_based_action(obs, indices, comfort_threshold, is_tropical=is_tropical)
                elif controller == "schedule":
                    action = schedule_based_action(obs, indices, is_tropical=is_tropical)
                else:
                    action = controller

                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))

                results[name]['energy'].append(obs[indices['power']])
                results[name]['indoor_temp'].append(obs[indices['i_temp']])
                results[name]['outdoor_temp'].append(obs[indices['o_temp']])
                results[name]['occ'].append(obs[indices['occ']])
                results[name]['month'].append(obs[indices['month']])
                results[name]['rh'].append(obs[indices['i_rh']])
                results[name]['hour'].append(obs[indices['hour']])
                
                pmv_val = calculate_pmv(
                    obs[indices['i_temp']], 
                    obs[indices['i_rh']], 
                    obs[indices['month']], 
                    is_tropical=is_tropical
                )
                results[name]['pmv'].append(pmv_val)
                results[name]['reward'].append(reward)
                done = terminated or truncated
                step += 1

    env.close()

    metrics = {}
    for name, values in results.items():
        power_arr = np.array(values["energy"])
        pmv_arr = np.array(values["pmv"])
        occ_arr = np.array(values["occ"])

        # EUI / Total Energy in kWh: sum(W) * 0.25h / 1000
        total_energy_kwh = float(np.sum(power_arr) * 0.25 / 1000.0)

        # Filter comfort statistics during occupied hours (occ > 0)
        occupied_indices = occ_arr > 0
        if np.any(occupied_indices):
            occ_pmv = pmv_arr[occupied_indices]
            avg_pmv_abs = float(np.mean(np.abs(occ_pmv)))
            violation_rate = float(np.mean(np.abs(occ_pmv) > comfort_threshold) * 100)
            discomfort_hours = float(np.sum(np.abs(occ_pmv) > comfort_threshold) * 0.25)
            pmv_hours = float(np.sum(np.maximum(0.0, np.abs(occ_pmv) - comfort_threshold)) * 0.25)
        else:
            avg_pmv_abs = 0.0
            violation_rate = 0.0
            discomfort_hours = 0.0
            pmv_hours = 0.0

        metrics[name] = {
            "total_energy_kwh": total_energy_kwh,
            "avg_power_w": float(np.mean(power_arr)),
            "occupied_avg_pmv_abs": avg_pmv_abs,
            "occupied_pmv_violation_rate_pct": violation_rate,
            "occupied_discomfort_hours": discomfort_hours,
            "occupied_pmv_hours": pmv_hours,
        }

    ref_energy = metrics[ref_baseline]["total_energy_kwh"]
    rl_energy = metrics["RL"]["total_energy_kwh"]
    rl_pmv_abs = metrics["RL"]["occupied_avg_pmv_abs"]
    rl_pmv_violation_rate = metrics["RL"]["occupied_pmv_violation_rate_pct"]

    print("\n" + "="*80)
    print("EVALUATION RESULTS (ASHRAE Standards - occupied hours comfort)")
    print("="*80)
    for name, values in metrics.items():
        saving = (ref_energy - values["total_energy_kwh"]) / ref_energy * 100 if ref_energy > 0 else 0.0
        print(
            f"{name:16s} Energy: {values['total_energy_kwh']:7.2f} kWh | "
            f"Saving vs {ref_baseline}: {saving:6.2f} % | "
            f"Occupied PMV abs: {values['occupied_avg_pmv_abs']:.2f} | "
            f"Violation: {values['occupied_pmv_violation_rate_pct']:5.2f} % | "
            f"Discomfort: {values['occupied_discomfort_hours']:5.1f} hrs | "
            f"PMV-hours: {values['occupied_pmv_hours']:5.1f}"
        )
    print("="*80)

    summary_row = {
        "episodes": episodes,
        "steps": max_steps,
        "comfort_threshold": comfort_threshold,
        "rl_total_energy_kwh": rl_energy,
        "baseline_total_energy_kwh": ref_energy,
        "energy_saving_pct": (ref_energy - rl_energy) / ref_energy * 100 if ref_energy > 0 else 0.0,
        "rl_occupied_avg_pmv_abs": rl_pmv_abs,
        "baseline_occupied_avg_pmv_abs": metrics[ref_baseline]["occupied_avg_pmv_abs"],
        "rl_occupied_pmv_violation_rate_pct": rl_pmv_violation_rate,
        "baseline_occupied_pmv_violation_rate_pct": metrics[ref_baseline]["occupied_pmv_violation_rate_pct"],
    }
    for name, values in metrics.items():
        prefix = name.lower()
        summary_row[f"{prefix}_total_energy_kwh"] = values["total_energy_kwh"]
        summary_row[f"{prefix}_energy_saving_vs_{ref_baseline.lower()}_pct"] = (
            ref_energy - values["total_energy_kwh"]
        ) / ref_energy * 100 if ref_energy > 0 else 0.0
        summary_row[f"{prefix}_occupied_avg_pmv_abs"] = values["occupied_avg_pmv_abs"]
        summary_row[f"{prefix}_occupied_pmv_violation_rate_pct"] = values["occupied_pmv_violation_rate_pct"]
        summary_row[f"{prefix}_occupied_discomfort_hours"] = values["occupied_discomfort_hours"]
        summary_row[f"{prefix}_occupied_pmv_hours"] = values["occupied_pmv_hours"]

    summary = pd.DataFrame([summary_row])
    summary_out = Path(summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    print(f"Summary metrics saved to '{summary_out}'")

    # Save detailed time series
    time_series_rows = []
    run_len = len(results["RL"]["energy"])
    for step_idx in range(run_len):
        row = {
            "step": step_idx,
            "month": results["RL"]["month"][step_idx],
            "hour": results["RL"]["hour"][step_idx],
            "outdoor_temp": results["RL"]["outdoor_temp"][step_idx],
            "occupancy": results["RL"]["occ"][step_idx],
        }
        for name in results.keys():
            row[f"{name}_indoor_temp"] = results[name]["indoor_temp"][step_idx]
            row[f"{name}_energy"] = results[name]["energy"][step_idx]
            row[f"{name}_pmv"] = results[name]["pmv"][step_idx]
            row[f"{name}_rh"] = results[name]["rh"][step_idx]
        time_series_rows.append(row)
        
    ts_df = pd.DataFrame(time_series_rows)
    time_series_out = Path(time_series_out)
    time_series_out.parent.mkdir(parents=True, exist_ok=True)
    ts_df.to_csv(time_series_out, index=False)
    print(f"Detailed evaluation time series saved to '{time_series_out}'")

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(results['RL']['indoor_temp'][:200], label='RL Indoor Temp', color='blue')
    plt.plot(results[ref_baseline]['indoor_temp'][:200], label=f'{ref_baseline} Indoor Temp', color='red', linestyle='--')
    plt.plot(results['Schedule_Based']['indoor_temp'][:200], label='Schedule-Based Indoor Temp', color='green', linestyle=':')
    plt.plot(results['Rule_Based']['indoor_temp'][:200], label='Rule-Based Indoor Temp', color='purple', linestyle='-.')
    plt.plot(results['RL']['outdoor_temp'][:200], label='Outdoor Temp', color='gray', alpha=0.5)
    plt.axhline(22, color='green', linestyle=':', alpha=0.5, label='Comfort Range')
    plt.axhline(26, color='green', linestyle=':', alpha=0.5)
    plt.title('Indoor Temperature Comparison (First 200 steps)')
    plt.legend()
    plt.ylabel('Temp (C)')

    plt.subplot(3, 1, 2)
    for name, style in [
        ('RL', '-'),
        (ref_baseline, '--'),
        ('Baseline_Fix_26', ':'),
        ('Baseline_Fix_27', '-.'),
        ('Schedule_Based', '-'),
        ('Rule_Based', '-'),
    ]:
        plt.plot(results[name]['energy'][:200], label=f'{name} Power', linestyle=style, alpha=0.85)
    plt.title('Power Demand Comparison')
    plt.legend()
    plt.ylabel('Power (W)')

    plt.subplot(3, 1, 3)
    plt.plot(results['RL']['pmv'][:200], label='RL PMV', color='blue')
    plt.plot(results[ref_baseline]['pmv'][:200], label=f'{ref_baseline} PMV', color='red', linestyle='--')
    plt.plot(results['Schedule_Based']['pmv'][:200], label='Schedule-Based PMV', color='green', linestyle=':')
    plt.plot(results['Rule_Based']['pmv'][:200], label='Rule-Based PMV', color='purple', linestyle='-.')
    plt.title('Comfort Index (PMV) Comparison')
    plt.axhline(0, color='black', alpha=0.3)
    plt.axhline(comfort_threshold, color='green', linestyle=':', alpha=0.6, label='Comfort Threshold')
    plt.axhline(-comfort_threshold, color='green', linestyle=':', alpha=0.6)
    plt.legend()
    plt.ylabel('PMV')

    plt.tight_layout()
    plot_out = Path(plot_out)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_out)
    print(f"\nComparison plots saved to '{plot_out}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weather', type=str, required=True)
    parser.add_argument('--model', type=str, default='models/dqn_hvac_vietnam.weights.h5')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--summary-out', type=str, default='summary_metrics.csv')
    parser.add_argument('--comfort-threshold', type=float, default=0.5)
    parser.add_argument('--start-month', type=int, default=None)
    parser.add_argument('--end-month', type=int, default=None)
    parser.add_argument('--start-day', type=int, default=1)
    parser.add_argument('--end-day', type=int, default=30)
    parser.add_argument('--plot-out', type=str, default='performance_comparison.png')
    parser.add_argument("--tropical", action="store_true", default=True, help="Use Clo=0.5 and Heating=15.0 for HCMC climate")
    parser.add_argument("--no-tropical", action="store_false", dest="tropical", help="Disable tropical mode")
    parser.add_argument('--time-series-out', type=str, default='artifacts/outputs/hcm_summer/evaluation_time_series.csv')
    args = parser.parse_args()
    
    evaluate(
        args.episodes,
        args.weather,
        args.model,
        args.steps,
        args.summary_out,
        args.comfort_threshold,
        args.start_month,
        args.end_month,
        args.start_day,
        args.end_day,
        args.plot_out,
        is_tropical=args.tropical,
        time_series_out=args.time_series_out,
    )
