# generate_fourier_systems.py
#
# Generate data for Fourier-type systems to test SINDy Fourier library
#
# System 1: Simple forced oscillator (linear + Fourier)
#   dx/dt = v
#   dv/dt = -x + 2.5*sin(t)          ← forcing term sin(t)
#
# System 2: Harder — double frequency forcing
#   dx/dt = v
#   dv/dt = -x - 0.5*v + 2.5*sin(t) - 1.2*cos(2t)
#
# System 3: Pure Fourier ODE (state = [x, t] trick)
#   Treat t as extra state variable so SINDy can see it
#   dx/dt = 2.5*sin(x2) - 1.2*cos(2*x2)
#   dx2/dt = 1   ← x2 = t
#
# NOTE: pySINDy Fourier library uses sin(kx), cos(kx) — functions of STATE,
# not time. To handle f(t), we use the x2=t trick in System 3.

import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp

os.makedirs('data', exist_ok=True)

NOISE = 0.02
SEED  = 42

def add_noise(X, std=NOISE, seed=SEED):
    rng = np.random.default_rng(seed)
    return X + rng.normal(0, std, X.shape)

# ─────────────────────────────────────────────────────────────
# System 1: Forced Harmonic Oscillator
# dx/dt  = v
# dv/dt  = -x + 2.5*sin(t)
#
# Expected SINDy result (Polynomial + Fourier Combined, degree=1):
#   dx/dt  = v
#   dv/dt  = -1.0*x + 2.5*sin(t)
# ─────────────────────────────────────────────────────────────
def forced_oscillator(t, s):
    x, v = s
    return [v, -x + 2.5 * np.sin(t)]

t1    = np.linspace(0, 20, 1000)
sol1  = solve_ivp(forced_oscillator, (0, 20), [0.0, 0.0],
                  t_eval=t1, rtol=1e-9, atol=1e-9)
X1    = add_noise(sol1.y.T)
df1   = pd.DataFrame(np.column_stack([t1, X1]), columns=['t', 'x', 'v'])
df1.to_csv('data/fourier_forced_oscillator.csv', index=False)
print("✅ System 1 — Forced Harmonic Oscillator")
print("   dx/dt = v")
print("   dv/dt = -x + 2.5·sin(t)")
print("   → Suggested: Combined library, degree=1, n_frequencies=1")
print()

# ─────────────────────────────────────────────────────────────
# System 2: Damped + Double Frequency Forcing (harder)
# dx/dt  = v
# dv/dt  = -x - 0.5*v + 2.5*sin(t) - 1.2*cos(2t)
#
# Expected SINDy result (Combined, degree=1, n_frequencies=2):
#   dx/dt  = v
#   dv/dt  = -1.0*x - 0.5*v + 2.5*sin(t) - 1.2*cos(2t)
# ─────────────────────────────────────────────────────────────
def damped_double_forced(t, s):
    x, v = s
    return [v, -x - 0.5*v + 2.5*np.sin(t) - 1.2*np.cos(2*t)]

t2   = np.linspace(0, 25, 1000)
sol2 = solve_ivp(damped_double_forced, (0, 25), [1.0, 0.0],
                 t_eval=t2, rtol=1e-9, atol=1e-9)
X2   = add_noise(sol2.y.T)
df2  = pd.DataFrame(np.column_stack([t2, X2]), columns=['t', 'x', 'v'])
df2.to_csv('data/fourier_damped_double.csv', index=False)
print("✅ System 2 — Damped Double Frequency Forcing")
print("   dx/dt = v")
print("   dv/dt = -x - 0.5·v + 2.5·sin(t) - 1.2·cos(2t)")
print("   → Suggested: Combined library, degree=1, n_frequencies=2")
print()

# ─────────────────────────────────────────────────────────────
# System 3: Pure Fourier ODE — x2 = t trick
# dx1/dt = 2.5*sin(x2) - 1.2*cos(2*x2)   ← đây là phương trình gốc
# dx2/dt = 1                               ← x2 tracks time
#
# SINDy sẽ tìm dx1/dt = f(x1, x2) với x2 là biến state đại diện cho t
# Expected: dx1/dt = 2.5*sin(x2) - 1.2*cos(2*x2)
#           dx2/dt = 1.0
# ─────────────────────────────────────────────────────────────
def pure_fourier_ode(t, s):
    x1, x2 = s
    return [2.5*np.sin(x2) - 1.2*np.cos(2*x2), 1.0]

t3   = np.linspace(0, 20, 1000)
sol3 = solve_ivp(pure_fourier_ode, (0, 20), [0.0, 0.0],
                 t_eval=t3, rtol=1e-9, atol=1e-9)
X3   = add_noise(sol3.y.T, std=0.01)  # noise nhỏ hơn vì hệ này nhạy hơn
df3  = pd.DataFrame(np.column_stack([t3, X3]), columns=['t', 'x1', 'x2'])
df3.to_csv('data/fourier_pure_ode.csv', index=False)
print("✅ System 3 — Pure Fourier ODE (x2 = t trick)")
print("   dx1/dt = 2.5·sin(x2) - 1.2·cos(2·x2)   ← phương trình gốc")
print("   dx2/dt = 1.0                              ← x2 tracks time t")
print("   → Suggested: Fourier library, n_frequencies=2, threshold=0.05")
print()

print("─" * 55)
print("📁 Files saved in data/:")
for f in ['fourier_forced_oscillator.csv',
          'fourier_damped_double.csv',
          'fourier_pure_ode.csv']:
    df  = pd.read_csv(f'data/{f}')
    print(f"   {f}  →  {len(df)} rows, cols: {list(df.columns)}")

print()
print("💡 Tips khi test với SINDy:")
print("   System 1 & 2: dùng Combined library (Polynomial + Fourier)")
print("   System 3    : dùng Fourier library thuần, n_frequencies=2")
print("   Threshold   : bắt đầu từ 0.05, tăng nếu quá nhiều terms")