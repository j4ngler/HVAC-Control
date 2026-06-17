import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def generate_plots(csv_path, output_dir, comfort_threshold=0.5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} steps from {csv_path}")
    
    controllers = ["RL", "Baseline_Fix_25", "Baseline_Fix_26", "Baseline_Fix_27", "Schedule_Based", "Rule_Based"]
    colors = {
        "RL": "blue",
        "Baseline_Fix_25": "red",
        "Baseline_Fix_26": "orange",
        "Baseline_Fix_27": "magenta",
        "Schedule_Based": "green",
        "Rule_Based": "purple"
    }
    styles = {
        "RL": "-",
        "Baseline_Fix_25": "--",
        "Baseline_Fix_26": ":",
        "Baseline_Fix_27": "-.",
        "Schedule_Based": "-",
        "Rule_Based": "-"
    }
    
    # Filter occupied steps for comfort metrics
    occ_df = df[df["occupancy"] > 0]
    
    # 1. Cumulative Energy Plot
    plt.figure(figsize=(10, 6))
    for name in controllers:
        cum_energy = (df[f"{name}_energy"] * 0.25 / 1000.0).cumsum()
        plt.plot(df["step"], cum_energy, label=name, color=colors[name], linestyle=styles[name], linewidth=2)
    plt.title("Tích lũy Điện năng Tiêu thụ (Cumulative Energy Consumption)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Số bước mô phỏng (Simulation Step)", fontsize=12)
    plt.ylabel("Điện năng tiêu thụ (kWh)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_energy.png", dpi=300)
    plt.close()
    print(f"Saved cumulative energy plot to {output_dir / 'cumulative_energy.png'}")
    
    # 2. PMV Box Plot during Occupied Hours
    plt.figure(figsize=(10, 6))
    box_data = [occ_df[f"{name}_pmv"] for name in controllers]
    bp = plt.boxplot(box_data, labels=controllers, patch_artist=True, showfliers=False)
    
    for patch, name in zip(bp['boxes'], controllers):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)
        patch.set_edgecolor(colors[name])
    
    # Add comfort range shaded area
    plt.axhspan(-comfort_threshold, comfort_threshold, color='green', alpha=0.1, label='Vùng tiện nghi (ASHRAE 55)')
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    
    plt.title("Phân bố chỉ số PMV trong giờ hoạt động (PMV Distribution - Occupied Hours)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Chỉ số tiện nghi nhiệt PMV", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5, axis='y')
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir / "pmv_boxplot.png", dpi=300)
    plt.close()
    print(f"Saved PMV boxplot to {output_dir / 'pmv_boxplot.png'}")
    
    # 3. Diurnal Profiles (Daily cycle averaged)
    # Group by exact hour (0.0, 0.25, ..., 23.75)
    daily_avg = df.groupby("hour").mean().reset_index()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Subplot 3a: Temp
    for name in controllers:
        axes[0].plot(daily_avg["hour"], daily_avg[f"{name}_indoor_temp"], label=name, color=colors[name], linestyle=styles[name], linewidth=2)
    axes[0].plot(daily_avg["hour"], daily_avg["outdoor_temp"], label="Outdoor Temp", color="gray", alpha=0.5, linestyle=":")
    axes[0].set_ylabel("Nhiệt độ phòng (°C)", fontsize=11)
    axes[0].set_title("Biến thiên Nhiệt độ Trung bình Ngày", fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", ncol=2)
    
    # Subplot 3b: Power
    for name in controllers:
        axes[1].plot(daily_avg["hour"], daily_avg[f"{name}_energy"], label=name, color=colors[name], linestyle=styles[name], linewidth=2)
    axes[1].set_ylabel("Công suất HVAC (W)", fontsize=11)
    axes[1].set_title("Biến thiên Công suất Tiêu thụ Trung bình Ngày", fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 3c: PMV
    for name in controllers:
        axes[2].plot(daily_avg["hour"], daily_avg[f"{name}_pmv"], label=name, color=colors[name], linestyle=styles[name], linewidth=2)
    axes[2].axhspan(-comfort_threshold, comfort_threshold, color='green', alpha=0.1)
    axes[2].axhline(0, color='black', alpha=0.3, linestyle='--')
    axes[2].set_ylabel("Chỉ số PMV", fontsize=11)
    axes[2].set_xlabel("Giờ trong ngày (Hour of Day)", fontsize=12)
    axes[2].set_title("Biến thiên Chỉ số PMV Trung bình Ngày", fontsize=12, fontweight='bold')
    axes[2].grid(True, linestyle="--", alpha=0.5)
    
    plt.xticks(range(0, 24, 2))
    plt.suptitle("Đặc tính Vận hành Trung bình Ngày (Daily Diurnal Profiles)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "diurnal_profiles.png", dpi=300)
    plt.close()
    print(f"Saved diurnal profiles to {output_dir / 'diurnal_profiles.png'}")
    
    # 4. Energy vs Comfort Tradeoff Scatter Plot
    plt.figure(figsize=(10, 6))
    
    fix_27_val = df["Baseline_Fix_27_energy"].sum() * 0.25 / 1000.0
    fix_25_val = df["Baseline_Fix_25_energy"].sum() * 0.25 / 1000.0
    x_min = fix_27_val - 50.0
    x_max = fix_25_val + 100.0
    
    for name in controllers:
        total_kwh = (df[f"{name}_energy"].sum() * 0.25 / 1000.0)
        avg_pmv_abs = occ_df[f"{name}_pmv"].abs().mean()
        plt.scatter(total_kwh, avg_pmv_abs, color=colors[name], s=150, marker='o', label=name, edgecolors='black', zorder=5)
        # Shift text slightly to avoid overlapping
        offset = 5.0
        plt.text(total_kwh + offset, avg_pmv_abs, name, fontsize=10, fontweight='bold')
        
    plt.title("Đánh đổi giữa Năng lượng và Tiện nghi nhiệt (Energy-Comfort Tradeoff)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Tổng Điện năng Tiêu thụ (kWh) - Càng thấp càng tốt", fontsize=12)
    plt.ylabel("Độ lệch PMV trung bình khi có người (|PMV| trung bình) - Càng thấp càng tốt", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    # Highlight optimal region (bottom-left)
    plt.axhspan(0, comfort_threshold, color='green', alpha=0.05, label='Khu vực tiện nghi (Tiện nghi đạt chuẩn)')
    plt.xlim(x_min, x_max)
    plt.ylim(0.0, 0.8)
    plt.legend(fontsize=11, loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "energy_comfort_tradeoff.png", dpi=300)
    plt.close()
    print(f"Saved energy comfort tradeoff to {output_dir / 'energy_comfort_tradeoff.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="artifacts/outputs/hcm_summer/evaluation_time_series.csv")
    parser.add_argument("--out-dir", type=str, default="artifacts/outputs/hcm_summer")
    parser.add_argument("--comfort-threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    generate_plots(args.csv, args.out_dir, args.comfort_threshold)
