import json
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_css_gantt(t_ads, t_bd, t_des, t_rep, save_path):
    bed_cycle = t_ads + t_bd + t_des + t_rep
    offset = t_ads

    seq_A = [
        ('Adsorption', t_ads, '#1f77b4'),
        ('Blowdown',   t_bd,  '#d62728'),
        ('Purge',      t_des, '#2ca02c'),
        ('Repress',    t_rep, '#ff7f0e'),
    ]

    cumulative = 0.0
    seq_B_offset = 0.0
    seq_B_start_idx = 0
    for i, (_, dur, _) in enumerate(seq_A):
        if cumulative + dur > offset:
            seq_B_start_idx = i
            seq_B_offset = offset - cumulative
            break
        cumulative += dur

    seq_B = []
    remaining = bed_cycle
    name, dur, color = seq_A[seq_B_start_idx]
    first_dur = dur - seq_B_offset
    seq_B.append((name, first_dur, color))
    remaining -= first_dur
    idx = (seq_B_start_idx + 1) % 4
    while remaining > 1e-9:
        name, dur, color = seq_A[idx]
        take = min(dur, remaining)
        seq_B.append((name, take, color))
        remaining -= take
        idx = (idx + 1) % 4

    fig, ax = plt.subplots(figsize=(13, 4.5), tight_layout=True)
    bed_y = {'Bed A': 11, 'Bed B': 1}
    bar_h = 8

    seen = set()

    def draw_sequence(ax, seq, y):
        t = 0.0
        for name, dur, color in seq:
            label = name if name not in seen else None
            seen.add(name)
            ax.broken_barh([(t, dur)], (y, bar_h), facecolors=color,
                           edgecolor='black', linewidth=0.7, label=label)
            if dur > bed_cycle * 0.025:
                txt_color = 'white' if name in ('Adsorption', 'Blowdown') else 'black'
                ax.text(t + dur / 2, y + bar_h / 2, f'{name}\n{dur:.0f} s',
                        ha='center', va='center',
                        fontsize=8.5, color=txt_color, fontweight='bold')
            t += dur

    draw_sequence(ax, seq_A, bed_y['Bed A'])
    draw_sequence(ax, seq_B, bed_y['Bed B'])

    ax.axvline(offset, linestyle='--', linewidth=1.0, color='grey', alpha=0.7)
    ax.text(offset, bed_y['Bed A'] + bar_h + 0.5,
            f'Bed B ads start\nt = {offset:.0f} s', ha='center', va='bottom', fontsize=8,
            color='grey')

    ax.set_yticks([bed_y['Bed B'] + bar_h / 2, bed_y['Bed A'] + bar_h / 2])
    ax.set_yticklabels(['Bed B', 'Bed A'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Time within one full cycle (s)', fontsize=11)
    ax.set_xlim(0, bed_cycle)
    ax.set_ylim(-1, bed_y['Bed A'] + bar_h + 3.5)
    ax.set_title(
        f'PSA Cyclic Steady-State Schedule — Bed Pair (one full cycle = {bed_cycle:.0f} s)',
        fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', linestyle=':', alpha=0.5)

    legend_phases = [
        Patch(facecolor='#1f77b4', edgecolor='black', label=f'Adsorption ({t_ads:.0f} s)'),
        Patch(facecolor='#d62728', edgecolor='black', label=f'Blowdown ({t_bd:.0f} s)'),
        Patch(facecolor='#2ca02c', edgecolor='black', label=f'Purge ({t_des:.0f} s)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label=f'Repress ({t_rep:.0f} s)'),
    ]
    ax.legend(handles=legend_phases, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=10)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

script_dir = os.path.dirname(os.path.abspath(__file__))

ads_script = "PSA(ads,New).py" 
des_script = "PSA(Depressurization).py"
rep_script = "PSA(Repressurization).py"

ads_path = os.path.join(script_dir, ads_script)
des_path = os.path.join(script_dir, des_script)
rep_path = os.path.join(script_dir, rep_script)

def wipe_states():
    print("\n--- Cleaning old bed states ---")
    files_to_delete = [
        "adsorption_end_state.npz", 
        "desorption_end_state.npz", 
        "previous_cycle_state.npz",
        "repressurization_end_state.npz"
    ]
    for file in files_to_delete:
        file_path = os.path.join(script_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ Deleted: {file}")
    print("✨ Bed is clean.")

Phigh = 15 * 101325
Plow = 1 * 101325
PFratio = 0.07
L = 13
d = 3
master_params = {
    "L": L,
    "T": 30 + 273.15,
    "R": 8.314,
    "P_high": Phigh,
    "P_low": Plow,
    "d": d,
    "Nsets": 3,
    "ratio_layer1": 0.8,
    "purge_fraction": PFratio,
    "N": 100,
    "dp": 0.002,
    "mu": 1.135086e-05,
    "P_atm_Pa": 101325.0,
    "eps_1": 0.35,
    "eps_2": 0.35,
    "rho_s_1": 850,
    "rho_s_2": 1160,
    "Adsorption_Ratio": 0.50,
    "Blowdown_Ratio": 0.15,
    "Purge_Ratio": 0.25,
    'Repress_Ratio': 0.1,
    "t_ads_start": 0,
    "t_ads_end": 2000,
    "t_op_ads": 400,
    "t_ads_safety_ratio": 0.9,
    "tau_bd": 30.0
}

config_path = os.path.join(script_dir, "master_config.json")
with open(config_path, "w") as f:
    json.dump(master_params, f, indent=4)

print("\n" + "="*50)
print(" PHASE A: SCOUT RUN (FINDING BREAKTHROUGH)")
print("="*50)
wipe_states()

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"
env["PSA_CYCLE"] = "0"
env["RUN_TYPE"] = "SCOUT"

subprocess.run(["python3", ads_path], env=env, check=True)

with open(config_path, "r") as f:
    optimized_config = json.load(f)

print(f"\n---> Scout Complete. Optimized Adsorption Time: {optimized_config['t_op_ads']:.1f} s <---")

wipe_states()

max_cycles = 50
convergence_tolerance = 8e-3
env["RUN_TYPE"] = "CSS"

for cycle in range(1, max_cycles + 1):
    print(f"\n" + "="*50)
    print(f" STARTING CYCLE {cycle} (CSS LOOP)")
    print("="*50)

    env["PSA_CYCLE"] = str(cycle)
    
    subprocess.run(["python3", ads_path], env=env, check=True)
    subprocess.run(["python3", des_path], env=env, check=True)
    subprocess.run(["python3", rep_path], env=env, check=True)

    state_file = os.path.join(script_dir, "repressurization_end_state.npz")
    prev_state_file = os.path.join(script_dir, "previous_cycle_state.npz")

    if os.path.exists(prev_state_file) and os.path.exists(state_file):
        current_q = np.load(state_file)['q_end']
        previous_q = np.load(prev_state_file)['q_end']
        
        residual = np.linalg.norm(current_q - previous_q) / np.linalg.norm(previous_q)
        print(f"\n---> Cycle {cycle} Convergence Residual: {residual:.4e} <---")
        
        if residual < convergence_tolerance:
            print("\n✅ CYCLIC STEADY STATE REACHED!")

            with open(config_path, "r") as f:
                final_config = json.load(f)
                
            final_t_ads = final_config.get("t_op_ads", 0)
            final_t_bd = final_config.get("t_blowdown_end", 0)
            final_t_des = final_config.get("tf_des", 0)
            final_t_rep = final_config.get("t_rep", 60.0)
            
            total_cycle_time = final_t_ads + final_t_bd + final_t_des + final_t_rep
            
            print("\n" + "="*50)
            print(" FINAL CSS TIMING METRICS (For CapEx/OpEx)")
            print("="*50)
            print(f"Adsorption Time:         {final_t_ads:>6.1f} s")
            print(f"Depressurization (BD):   {final_t_bd:>6.1f} s")
            print(f"Desorption (Purge):      {final_t_des:>6.1f} s")
            print(f"Repressurization:        {final_t_rep:>6.1f} s")
            print("-" * 50)
            print(f"TOTAL CYCLE TIME:        {total_cycle_time:>6.1f} s")
            print(f"Total Cycles Simulated:  {cycle}")
            print("="*50 + "\n")

            with open(os.path.join(script_dir, "results", "CSS_Timing_Report.txt"), "w") as f:
                f.write("FINAL CSS TIMING METRICS\n")
                f.write(f"Adsorption:     {final_t_ads:.1f} s\n")
                f.write(f"Blowdown:       {final_t_bd:.1f} s\n")
                f.write(f"Purge:          {final_t_des:.1f} s\n")
                f.write(f"Repress:        {final_t_rep:.1f} s\n")
                f.write(f"Total Cycle:    {total_cycle_time:.1f} s\n")
                f.write(f"Cycles Simulated: {cycle}\n")
                f.write('' + "="*30 + "\n")
                f.write('COLUMN SPECS:\n')
                f.write(f"Bed Length (L): {final_config['L']} m\n")
                f.write(f"Bed Diameter (d): {final_config['d']} m\n")
                f.write(f'L/D Ratio: {final_config["L"] / final_config["d"]:.3f}\n')
                f.write(f'Number of Pairs: {final_config["Nsets"]}\n')

            gantt_path = os.path.join(script_dir, "results", "CSS_Gantt_Schedule.png")
            plot_css_gantt(t_ads=final_t_ads,
                           t_bd=final_t_bd,
                           t_des=final_t_des,
                           t_rep=final_t_rep,
                           save_path=gantt_path)
            print(f"📊 CSS Gantt chart saved to {gantt_path}")

            break
    else:
        print("\n---> Cycle 1 complete. Establishing baseline. <---")
    
    if os.path.exists(state_file):
        data = np.load(state_file)
        np.savez(prev_state_file, q_end=data['q_end'])

print("\n" + "="*50)
print(" PHASE C: FINAL DIAGNOSTIC RUN (CSS BREAKTHROUGH)")
print("="*50)

env["PSA_CYCLE"] = "FINAL"
env["RUN_TYPE"] = "FINAL"

subprocess.run(["python3", ads_path], env=env, check=True)
print("\n🎉 AUTOMATED WORKFLOW COMPLETE. Check the 'results' folder.")
