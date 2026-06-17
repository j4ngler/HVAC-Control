import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import sinergym
from pythermalcomfort.models import pmv_ppd_iso


if "EPLUS_PATH" not in os.environ:
    os.environ["EPLUS_PATH"] = "/usr/local/EnergyPlus-25-2-0"
if os.environ["EPLUS_PATH"] not in sys.path:
    sys.path.insert(0, os.environ["EPLUS_PATH"])


def clothing_by_month(month):
    month = int(month)
    return 0.8 if month in {11, 12, 1, 2} else 0.5


def calculate_pmv_ppd(temp, rh, month, air_velocity=0.1, met_rate=1.2):
    clo = clothing_by_month(month)
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
        ppd = float(result.ppd)
    except Exception:
        pmv = (float(temp) - 23.0) * 0.5 + (float(rh) - 50.0) * 0.01
        ppd = np.nan
    return float(np.clip(pmv, -3.0, 3.0)), ppd, clo


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


def add_comfort_fields(prefix, row, values, obs_names, comfort_threshold, air_velocity, met_rate):
    obs = dict(zip(obs_names, values))
    temp = obs["air_temperature"]
    rh = obs["air_humidity"]
    month = obs["month"]
    pmv, ppd, clo = calculate_pmv_ppd(temp, rh, month, air_velocity, met_rate)
    row[f"{prefix}_pmv"] = pmv
    row[f"{prefix}_ppd"] = ppd
    row[f"{prefix}_comfort_violation"] = abs(pmv) > comfort_threshold
    row[f"{prefix}_mean_radiant_temperature_assumed"] = temp
    row[f"{prefix}_air_velocity_assumed"] = air_velocity
    row[f"{prefix}_met_rate"] = met_rate
    row[f"{prefix}_clo_value"] = clo


def generate_data(args):
    weather_file_abs = os.path.abspath(args.weather)
    config_params = make_runperiod(args.start_month, args.end_month, args.start_day, args.end_day)
    print(f"Initializing Environment: {args.env}")
    print(f"Using weather file: {weather_file_abs}")
    print(f"Using runperiod: {config_params['runperiod']}")

    env = gym.make(args.env, weather_files=[weather_file_abs], config_params=config_params)
    obs_names = list(env.unwrapped.observation_variables)
    transition_data = []

    for ep in range(args.episodes):
        print(f"Starting Episode {ep + 1}/{args.episodes}")
        obs, info = env.reset()
        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated) and step < args.max_steps:
            state_dict = {f"State_{name}": val for name, val in zip(obs_names, obs)}
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_dict = {f"NextState_{name}": val for name, val in zip(obs_names, next_obs)}

            row = {
                "City": args.city,
                "Season": args.season,
                "Weather_File": Path(weather_file_abs).name,
                "Start_Month": args.start_month,
                "End_Month": args.end_month,
                "Episode": ep + 1,
                "Step": step,
                "Action_Heating_Setpoint": action[0],
                "Action_Cooling_Setpoint": action[1],
                "Reward": reward,
                "Terminated": terminated,
            }
            row.update(state_dict)
            row.update(next_state_dict)
            add_comfort_fields(
                "State",
                row,
                obs,
                obs_names,
                args.comfort_threshold,
                args.air_velocity,
                args.met_rate,
            )
            add_comfort_fields(
                "NextState",
                row,
                next_obs,
                obs_names,
                args.comfort_threshold,
                args.air_velocity,
                args.met_rate,
            )

            transition_data.append(row)
            obs = next_obs
            step += 1

            if step % 200 == 0:
                print(f"  ... Running Step {step}/{args.max_steps}")

    env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(transition_data)
    df.to_csv(output, index=False)
    print(f"\nData generation complete! Saved {len(df)} transition rows to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate city-season HVAC transition data.")
    parser.add_argument("--env", type=str, default="Eplus-5zone-mixed-continuous-v1")
    parser.add_argument("--city", type=str, required=True)
    parser.add_argument("--season", type=str, required=True)
    parser.add_argument("--weather", type=str, required=True)
    parser.add_argument("--start-month", type=int, required=True)
    parser.add_argument("--end-month", type=int, required=True)
    parser.add_argument("--start-day", type=int, default=1)
    parser.add_argument("--end-day", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--comfort-threshold", type=float, default=0.5)
    parser.add_argument("--air-velocity", type=float, default=0.1)
    parser.add_argument("--met-rate", type=float, default=1.2)
    generate_data(parser.parse_args())
