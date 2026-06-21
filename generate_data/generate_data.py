# generate_data.py
#
# Coupled Spring-Mass System (2 mass)
#
#   m1*x1'' = -k1*x1 + k2*(x2 - x1)
#   m2*x2'' = -k2*(x2 - x1)
#
# State vector: [x1, v1, x2, v2]
#
# Usage:
#   python generate_data.py
#
# Output (vào thư mục data/):
#   train_data.csv      ← train + validate
#   test_data_1.csv     ← test run 1, different initial condition
#   test_data_2.csv     ← test run 2, 

import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp


M1 = 1.0    # mass 1 (kg)
M2 = 1.0    # mass 2 (kg)
K1 = 1.0    # spring 1 stiffness (N/m)  
K2 = 0.5    # spring 2 stiffness (N/m)  

NOISE_STD = 0.02    
RANDOM_SEED = 42

# true model
def coupled_spring_mass(t, state):
    x1, v1, x2, v2 = state
    a1 = (-K1 * x1 + K2 * (x2 - x1)) / M1
    a2 = (-K2 * (x2 - x1))            / M2
    return [v1, a1, v2, a2]

# generate csv file
def generate_csv(filename, x0, t_end=25.0, n_points=1000, noise_std=NOISE_STD, seed=None):
    """
    Integrate hệ từ x0, add Gaussian noise, lưu CSV.

    Args:
        filename  : đường dẫn output
        x0        : [x1_0, v1_0, x2_0, v2_0]
        t_end     : thời gian kết thúc (giây)
        n_points  : số điểm thời gian
        noise_std : std của Gaussian noise
        seed      : random seed cho noise
    """
    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        coupled_spring_mass,
        t_span, x0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-9, atol=1e-9,   # tolerance for clean data
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    X_clean = sol.y.T   # (n_points, 4)

    # Add Gaussian noise
    rng   = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, X_clean.shape)
    X_noisy = X_clean + noise

    df = pd.DataFrame(
        np.column_stack([t_eval, X_noisy]),
        columns=['t', 'x1', 'v1', 'x2', 'v2']
    )
    df.to_csv(filename, index=False)
    print(f"✅ Saved: {filename}  |  x0={x0}  |  {n_points} points  |  noise_std={noise_std}")

# -----------------------------------------------------------------------
# 4. Generate các file
# -----------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    # --- Train data ---
    # IC: x1=1.0, v1=0.0, x2=0.5, v2=0.0
    generate_csv(
        filename  = 'data/train_data.csv',
        x0        = [1.0, 0.0, 0.5, 0.0],
        t_end     = 25.0,
        n_points  = 1000,
        seed      = 42,
    )

    # --- Test Run 1 ---
    # IC : x1=2.0, v1=0.5, x2=-1.0, v2=0.0
    generate_csv(
        filename  = 'data/test_data_1.csv',
        x0        = [2.0, 0.5, -1.0, 0.0],
        t_end     = 25.0,
        n_points  = 1000,
        seed      = 123,
    )

    # --- Test Run 2 ---
    # IC : x1=-1.5, v1=0.0, x2=2.0, v2=-0.5
    generate_csv(
        filename  = 'data/test_data_2.csv',
        x0        = [-1.5, 0.0, 2.0, -0.5],
        t_end     = 25.0,
        n_points  = 1000,
        seed      = 999,
    )

    print("\n📁 Files generated:")
    for f in ['train_data.csv', 'test_data_1.csv', 'test_data_2.csv']:
        path = f'data/{f}'
        df   = pd.read_csv(path)
        print(f"   {path}  →  {len(df)} rows, columns: {list(df.columns)}")