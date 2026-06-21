# generate_vanderpol.py
#
# Van der Pol Oscillator
#
#   dx/dt = v
#   dv/dt = μ(1 - x²)v - x
#
# State vector: [x, v]
# Terms cần để SINDy recover: x, v, x²v → degree=2 (hoặc degree=3 nếu dùng x²v)
#
# Usage:
#   python generate_vanderpol.py
#
# Output (vào thư mục data/):
#   vanderpol_train.csv
#   vanderpol_test_1.csv
#   vanderpol_test_2.csv

import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------
# 1. Tham số hệ thống
# -----------------------------------------------------------------------
MU = 1.5    # nonlinearity parameter
            # μ=0   → linear (simple harmonic)
            # μ=1   → mildly nonlinear
            # μ=1.5 → clearly nonlinear, SINDy cần degree≥2
            # μ>3   → stiff, khó integrate

NOISE_STD = 0.02

# -----------------------------------------------------------------------
# 2. Phương trình vi phân
# -----------------------------------------------------------------------
def van_der_pol(t, state):
    x, v = state
    dxdt = v
    dvdt = MU * (1 - x**2) * v - x
    return [dxdt, dvdt]

# -----------------------------------------------------------------------
# 3. Hàm generate CSV
# -----------------------------------------------------------------------
def generate_csv(filename, x0, t_end=20.0, n_points=1000,
                 noise_std=NOISE_STD, seed=None):
    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, n_points)

    sol = solve_ivp(
        van_der_pol,
        t_span, x0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-9, atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    X_clean = sol.y.T   # (n_points, 2)

    rng    = np.random.default_rng(seed)
    noise  = rng.normal(0.0, noise_std, X_clean.shape)
    X_noisy = X_clean + noise

    df = pd.DataFrame(
        np.column_stack([t_eval, X_noisy]),
        columns=['t', 'x', 'v']
    )
    df.to_csv(filename, index=False)
    print(f"✅ Saved: {filename}  |  x0={x0}  |  {n_points} points  |  noise_std={noise_std}")

# -----------------------------------------------------------------------
# 4. Generate files
# -----------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    # Train — IC gần limit cycle
    generate_csv(
        filename = 'data/vanderpol_train.csv',
        x0       = [1.0, 0.0],
        t_end    = 20.0,
        n_points = 1000,
        seed     = 42,
    )

    # Test 1 — IC khác, bên trong limit cycle
    generate_csv(
        filename = 'data/vanderpol_test_1.csv',
        x0       = [0.5, 1.5],
        t_end    = 20.0,
        n_points = 1000,
        seed     = 123,
    )

    # Test 2 — IC khác, bên ngoài limit cycle
    generate_csv(
        filename = 'data/vanderpol_test_2.csv',
        x0       = [3.0, 0.0],
        t_end    = 20.0,
        n_points = 1000,
        seed     = 999,
    )

    print(f"\n📐 Van der Pol system (μ={MU}):")
    print(f"   dx/dt = v")
    print(f"   dv/dt = {MU}·(1 - x²)·v - x")
    print(f"\n💡 SINDy settings gợi ý:")
    print(f"   Library  : Polynomial, degree=3")
    print(f"   Threshold: 0.05 ~ 0.15")
    print(f"\n📁 Files generated:")
    for f in ['vanderpol_train.csv', 'vanderpol_test_1.csv', 'vanderpol_test_2.csv']:
        path = f'data/{f}'
        df   = pd.read_csv(path)
        print(f"   {path}  →  {len(df)} rows, columns: {list(df.columns)}")