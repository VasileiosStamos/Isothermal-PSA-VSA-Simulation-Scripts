import os
from math import pi
import numpy as np
import matplotlib
import json

matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- SETUP ---
run_type = os.environ.get("RUN_TYPE", "CSS")
current_cycle = int(os.environ.get("PSA_CYCLE", 1))

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "results")
cycle_folder = os.path.join(base_dir, f"{run_type}_cycle_{current_cycle}")
os.makedirs(cycle_folder, exist_ok=True)

print(f"\n--- Starting Repressurization for [{run_type}] Cycle {current_cycle} ---")

# --- LOAD PARAMETERS & BED STATE ---
config_path = os.path.join(script_dir, "master_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

tf_rep = float(config.get("t_rep", 60.0))
t_eval_rep = np.linspace(0, tf_rep, 150)

L = float(config["L"])
T = float(config["T"])
R = float(config["R"])
d = float(config["d"])
A = pi * (d / 2)**2
N = int(config["N"])
P_high = float(config["P_high"])
P_low = float(config["P_low"])

state_file = os.path.join(script_dir, 'desorption_end_state.npz')
if not os.path.exists(state_file):
    raise FileNotFoundError(f"Cannot find {state_file}.")

state_data = np.load(state_file)
y0_rep = np.concatenate([state_data['C_end'], state_data['q_end']])

labels = ['N2', 'CO2', 'O2']
colors = ['gray', 'blue', 'red']
MW = np.array([0.028014, 0.044009, 0.031998])
dz = L / N
z_nodes = np.linspace(dz/2, L - dz/2, N)

eps = float(config.get("eps", 0.35))
rho_s = float(config.get("rho_s", 2000))
mass_transfer_coeff = ((1 - eps) / eps) * rho_s

# Toth isotherm constants
k_ldf = np.array([[0.0021], [0.0143], [0.002]]).repeat(N, axis=1) 

k_qs_1 = np.array([[1.89],[2.82],[1e-8]]).repeat(N, axis=1)
k_qs_2 = np.array([[-2.25e-4],[-3.50e-4],[1e-8]]).repeat(N, axis=1)
q_s = k_qs_1 + k_qs_2 * T    

b_0 = np.array([[1.16e-9], [2.83e-9], [1e-8]]).repeat(N, axis=1)  
B_val = np.array([[1944.61], [2598.2], [1e-8]]).repeat(N, axis=1)                 
b_toth = b_0 * np.exp(B_val / T)

# feed composition at open top
y_rep = np.array(config.get("y_rep", [1.0, 0.0, 0.0]), dtype=float)

# --- REPRESSURIZATION PHASE ---

print(f"\n--- Solving Repressurization phase (Target: {P_high/1e5:.1f} bar in {tf_rep:.1f}s) ---")
pbar_rep = tqdm(total=P_high, desc="Pressurizing", unit="Pa", bar_format="{l_bar}{bar}| {n:.2e}/{total_fmt} [{elapsed}<{remaining}]")
last_P_mean = [P_low] 

def pde_repressurization(t, y):
    C = y[:3*N].reshape((3, N))
    q = y[3*N:].reshape((3, N))

    # prevent negative mass
    C_safe = np.maximum(C, 1e-12)
    q_safe = np.maximum(q, 1e-12)

    dPdt_target = (P_high - P_low) / tf_rep
    P_t = P_low + dPdt_target * t

    if P_t > last_P_mean[0]:
        pbar_rep.update(P_t - last_P_mean[0])
        last_P_mean[0] = P_t

    C_total_theo = P_t / (R * T)
    dCtot_dt_compression = dPdt_target / (R * T)

    y_frac = C_safe / np.sum(C_safe, axis=0)
    P_partial_Pa = np.clip((y_frac * P_t), 1e-10, P_high * 1.5)

    denom = 1.0 + (b_toth * P_partial_Pa)
    q_star = (b_toth * P_partial_Pa * q_s) / denom

    dqdt = k_ldf * (q_star - q_safe)
    sum_dqdt = np.sum(dqdt, axis=0)

    # velocity anchored at closed bottom
    dv_dz = -(dCtot_dt_compression + mass_transfer_coeff * sum_dqdt) / C_total_theo

    v_local = np.zeros(N)
    v_local[0] = 0.0

    for j in range(0, N-1):
        v_local[j+1] = v_local[j] + dv_dz[j] * dz

    v_inlet = v_local[-1] + dv_dz[-1] * dz

    # dynamic upwinding
    F_L = np.zeros((3, N))
    mask_fwd = v_local > 0.0
    mask_bwd = v_local <= 0.0

    F_L[:, 1:] = (v_local[1:] * C_safe[:, :-1]) * mask_fwd[1:] + (v_local[1:] * C_safe[:, 1:]) * mask_bwd[1:]
    F_L[:, 0] = 0.0

    F_R = np.zeros((3, N))
    F_R[:, :-1] = F_L[:, 1:]

    # inlet: feed in or leak out
    C_feed = y_rep * (P_t / (R * T))
    F_R[:, -1] = (v_inlet * C_safe[:, -1]) * (v_inlet > 0) + (v_inlet * C_feed) * (v_inlet <= 0)
    
    d_vC_dz = (F_R - F_L) / dz
    dCdt = -d_vC_dz - mass_transfer_coeff * dqdt
    
    return np.concatenate([dCdt.flatten(), dqdt.flatten()])

sol_rep = solve_ivp(pde_repressurization, [t_eval_rep[0], t_eval_rep[-1]], y0_rep, method='BDF',
                   t_eval=t_eval_rep,
                   rtol=1e-3, atol=1e-4, first_step=0.1) 

pbar_rep.close() 

actual_steps_rep = sol_rep.y.shape[1]
C_history_rep = sol_rep.y[:3*N, :].T.reshape((actual_steps_rep, 3, N))
t_actual_rep = sol_rep.t
P_history = np.sum(np.maximum(C_history_rep, 1e-10), axis=1) * R * T

# --- VISUALIZATION ---
fig_p, ax_p = plt.subplots(figsize=(10, 6), tight_layout=True)

sample_indices = np.linspace(0, actual_steps_rep - 1, 5, dtype=int)
time_cmap = plt.cm.viridis(np.linspace(0, 0.9, len(sample_indices)))

for i, idx in enumerate(sample_indices):
    t_val = t_actual_rep[idx]
    ax_p.plot(z_nodes, P_history[idx] / 1e5, 
              label=f't={t_val:.1f}s', color=time_cmap[i], linewidth=2.5)

ax_p.set_title(f'Repressurization: Imposed Pressure Build-up ({tf_rep}s)', fontsize=14, fontweight='bold')
ax_p.set_xlabel('Column Length z (m)', fontsize=12)
ax_p.set_ylabel('Pressure (bar)', fontsize=12)
ax_p.axvline(x=L, color='red', linestyle='--', label='Product End Inlet', alpha=0.7)
ax_p.axvline(x=0, color='black', linestyle='-', label='Feed End (Closed)', alpha=0.7)
ax_p.set_ylim(0, P_high/1e5 * 1.1) 
ax_p.legend()
ax_p.grid(True, linestyle=':', alpha=0.6)

# --- SAVE RESULTS ---
pressure_path = os.path.join(cycle_folder, "rep_pressure_wave.png")
fig_p.savefig(pressure_path)

rep_state_file = os.path.join(script_dir, 'repressurization_end_state.npz')
np.savez(rep_state_file, 
         C_end=sol_rep.y[:3*N, -1], 
         q_end=sol_rep.y[3*N:, -1])

print(f"✅ State saved to {rep_state_file}")

plt.close('all')
os._exit(0)