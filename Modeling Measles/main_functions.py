import pandas as pd
import numpy as np

def convert_cumulative_to_SIR(df, date_col='date', cumulative_col='cumulative_cases',
                              population=None, infectious_period=2, recovered_col=None,
                              new_case_col='new_cases', I_col='I_est', R_col='R_est', S_col='S_est'):
    """
    Convert cumulative reported cases into S, I, R estimates for SIR modeling.
    - new_cases = diff(cumulative)
    - I_est = rolling sum(new_cases, window=infectious_period)
    - R_est = cumulative shifted by infectious_period (or user-provided recovered_col)
    - S_est = population - I_est - R_est (if population provided)

    Returns a copy of the dataframe with the added columns.
    """
    df = df.copy()
    # Ensure date column sorted if present
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

    if cumulative_col not in df.columns:
        raise ValueError(f"Column '{cumulative_col}' not found in dataframe.")

    # Compute new cases (incident)
    df[new_case_col] = df[cumulative_col].diff().fillna(
        df[cumulative_col].iloc[0])
    df[new_case_col] = df[new_case_col].clip(lower=0)

    # Estimate I(t) as rolling sum over infectious_period
    if infectious_period <= 0:
        raise ValueError("infectious_period must be positive integer.")
    df[I_col] = df[new_case_col].rolling(
        window=infectious_period, min_periods=1).sum()

    # Estimate R(t)
    if recovered_col and recovered_col in df.columns:
        df[R_col] = df[recovered_col].fillna(0)
    else:
        df[R_col] = df[cumulative_col].shift(infectious_period).fillna(0)

    # Compute S(t) if population provided
    if population is not None:
        df[S_col] = population - df[I_col] - df[R_col]
        df[S_col] = df[S_col].clip(lower=0)
    else:
        df[S_col] = np.nan

    # Ensure numeric and non-negative
    for col in [new_case_col, I_col, R_col]:
        df[col] = df[col].astype(float).clip(lower=0)
    if population is not None:
        df[S_col] = df[S_col].astype(float)

    return df
#-----------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from main_functions import convert_cumulative_to_SIR
# 1. Read your weekly measles data
df = pd.read_csv("measles_nigeria_data_2020-2021_new_cases.csv")
df["date"] = pd.to_datetime(df["date"])
# The CSV has weekly *new* confirmed cases.
# Build a cumulative column for the helper function:
df["cumulative_cases"] = df["confirmed_cases"].cumsum()

# 2. Convert cumulative cases → S, I, R using a WEEK-based infectious period
population_nigeria = 206_000_000  # approx 2020–2021
infectious_period_weeks = 2       # 12 days ≈ 1.7 weeks → ~2 weekly time steps
sir_df = convert_cumulative_to_SIR(
    df,
    date_col="date",
    cumulative_col="cumulative_cases",
    population=population_nigeria,
    infectious_period=infectious_period_weeks)
print(sir_df.head())

# Plot I(t) alone
plt.figure(figsize=(10, 6))
plt.plot(sir_df["date"], sir_df["I_est"], label="Estimated Infectious I(t)")

plt.xlabel("Date")
plt.ylabel("Estimated number infectious")
plt.title("Estimated Infectious Population I(t) for Measles in Nigeria (2020-2021)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#------------------------------------------------------------------------
from main_functions import convert_cumulative_to_SIR

def euler_sir(beta, gamma, S0, I0, R0, t, N):
    """
    Solve the SIR model using Euler's method.
    Parameters:
    - beta: Infection rate
    - gamma: Recovery rate
    - S0: Initial susceptible population
    - I0: Initial infected population
    - R0: Initial recovered population
    - t: Array of time points (days or weeks)
    - N: Total population
    Returns:
    - S: Array of susceptible population over time
    - I: Array of infected population over time
    - R: Array of recovered population over time
    """
    S = np.empty(len(t), float)
    I = np.empty(len(t), float)
    R = np.empty(len(t), float)
    S[0], I[0], R[0] = S0, I0, R0
    for n in range(len(t) - 1):
        dt = t[n + 1] - t[n]  # step size
        # SIR ODEs:
        # dS/dt = -beta * S * I / N
        # dI/dt =  beta * S * I / N - gamma * I
        # dR/dt =  gamma * I
        dS = -beta * S[n] * I[n] / N
        dI = beta * S[n] * I[n] / N - gamma * I[n]
        dR = gamma * I[n]
        # Euler update: next = current + derivative * dt
        S[n + 1] = S[n] + dS * dt
        I[n + 1] = I[n] + dI * dt
        R[n + 1] = R[n] + dR * dt
    return S, I, R

sir_df = convert_cumulative_to_SIR(
    df,
    date_col="date",
    cumulative_col="cumulative_cases",
    population=population_nigeria,
    infectious_period=infectious_period_weeks)

# Time array: one step per row (assume equal time spacing: weekly)
t = np.arange(len(sir_df), dtype=float)  # 0, 1, 2, ..., n-1
N = population_nigeria

# Initial conditions from the first row of your data-based SIR estimates
S0 = sir_df["S_est"].iloc[0]
I0 = sir_df["I_est"].iloc[0]
R0 = sir_df["R_est"].iloc[0]

I_true = sir_df["I_est"].values

beta_guess = 0.14
gamma_guess = 0.1

# Step 3: Get model S, I, R from Euler SIR
S_model, I_model, R_model = euler_sir(
    beta=beta_guess,
    gamma=gamma_guess,
    S0=S0,
    I0=I0,
    R0=R0,
    t=t,
    N=N)
print("I_true min/max:", I_true.min(), I_true.max())
print("I_model min/max:", I_model.min(), I_model.max())

# -------- Plot: TRUE vs MODEL  --------
plt.figure(figsize=(10, 5))
plt.plot(sir_df["date"], I_true, label="True I(t)")
plt.plot(sir_df["date"], I_model, label="Model I(t)", linestyle="--")
 
plt.xlabel("Date")
plt.ylabel("Infected people")
plt.title(f"I(t): data vs model, beta={beta_guess}, gamma={gamma_guess}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: SSE in raw counts
sse_I = np.sum((I_model - I_true) ** 2)
print("SSE for I(t):", sse_I)

# ----------------------------

from scipy.optimize import minimize
I_true = sir_df["I_est"].values
t = np.arange(len(sir_df), dtype=float)
N = population_nigeria

S0 = sir_df["S_est"].iloc[0]
I0 = sir_df["I_est"].iloc[0]
R0 = sir_df["R_est"].iloc[0]

def sse_objective(params):
    """
    Objective function for optimization.
    params[0] = beta, params[1] = gamma
    Returns SSE between model I(t) and data I_true.
    """
    beta, gamma = params
    # Enforce positivity (if optimizer ever tries negative values)
    if beta <= 0 or gamma <= 0:
        return 1e30  # huge penalty
    S_model, I_model, R_model = euler_sir(
        beta=beta,
        gamma=gamma,
        S0=S0,
        I0=I0,
        R0=R0,
        t=t,
        N=N)
    return np.sum((I_model - I_true) ** 2)
# Initial guess for (beta, gamma)
initial_guess = np.array([0.2, 0.5])  # you can change these
# Bounds to keep beta, gamma in reasonable ranges
bounds = [(1e-6, 2.0),  # beta between ~0 and 2 per week
          (1e-6, 2.0)]  # gamma between ~0 and 2 per week
result = minimize(
    sse_objective,
    x0=initial_guess,
    bounds=bounds,
    method="L-BFGS-B")
beta_opt, gamma_opt = result.x


print("Optimal beta:", beta_opt)
print("Optimal gamma:", gamma_opt)
print("Optimal SSE:", result.fun)
# Recompute model with optimal parameters
S_opt, I_opt, R_opt = euler_sir(
    beta=beta_opt,
    gamma=gamma_opt,
    S0=S0,
    I0=I0,
    R0=R0,
    t=t,
    N=N)
# Plot data vs optimized model
plt.figure(figsize=(10, 5))
plt.plot(sir_df["date"], I_true, label="True I(t)")
plt.plot(sir_df["date"], I_opt, "--", label=f"Model I(t), β={beta_opt:.3f}, γ={gamma_opt:.3f}")
plt.xlabel("Date")
plt.ylabel("Infected people")
plt.title("I(t): data vs optimized SIR model (SciPy minimize)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------

I_true_all = sir_df["I_est"].values
dates_all = sir_df["date"].values
n = len(sir_df)
t_all = np.arange(n, dtype=float)  # 0,1,2,... (weekly steps)

# Split index for first vs second half
mid = n // 2  # integer division

# First-half data (for fitting)
I_true_fit = I_true_all[:mid]
t_fit = t_all[:mid]

# Second-half data (for error evaluation)
I_true_second = I_true_all[mid:]
t_second = t_all[mid:]

N = population_nigeria

# Initial conditions from the first row (same as before)
S0 = sir_df["S_est"].iloc[0]
I0 = sir_df["I_est"].iloc[0]
R0 = sir_df["R_est"].iloc[0]


# ---------- SSE function that uses ONLY FIRST HALF ----------
def sir_sse_first_half(beta, gamma):
    """
    SSE between model and data for the FIRST HALF of the time series.
    """
    # Run model over the FULL time (simplest), then compare only first half
    S_model, I_model, R_model = euler_sir(
        beta=beta,
        gamma=gamma,
        S0=S0,
        I0=I0,
        R0=R0,
        t=t_all,
        N=N
    )
    # Use only first-half points for SSE
    return np.sum((I_model[:mid] - I_true_fit) ** 2)


# ---------- GRID SEARCH on beta, gamma using FIRST HALF ----------
beta_values = np.linspace(0.01, 1.0, 50)   # adjust range/resolution if needed
gamma_values = np.linspace(0.05, 1.0, 50)

best_sse = np.inf
best_beta = None
best_gamma = None

for beta in beta_values:
    for gamma in gamma_values:
        sse = sir_sse_first_half(beta, gamma)
        if sse < best_sse:
            best_sse = sse
            best_beta = beta
            best_gamma = gamma

print("Best beta (fit on first half):", best_beta)
print("Best gamma (fit on first half):", best_gamma)
print("Best SSE on FIRST HALF:", best_sse)


# ---------- RUN MODEL WITH BEST PARAMETERS ON FULL TIME RANGE ----------
S_best, I_best, R_best = euler_sir(
    beta=best_beta,
    gamma=best_gamma,
    S0=S0,
    I0=I0,
    R0=R0,
    t=t_all,
    N=N
)

# ---------- STEP 4: COMPUTE ERROR ON SECOND HALF ----------
I_model_second = I_best[mid:]
sse_second_half = np.sum((I_model_second - I_true_second) ** 2)

print("SSE on SECOND HALF (prediction error for Euler model):", sse_second_half)

plt.figure(figsize=(10, 5))

# First half: where parameters were fit
plt.plot(dates_all[:mid], I_true_fit, label="True I(t) – first half")
plt.plot(dates_all[:mid], I_best[:mid], "--", label="Model I(t) – fitted")

# Second half: prediction vs data
plt.plot(dates_all[mid:], I_true_second, label="True I(t) – second half")
plt.plot(dates_all[mid:], I_model_second, "--", label="Model I(t) – prediction")

plt.xlabel("Date")
plt.ylabel("Infected people")
plt.title("I(t): Fit on first half, predict second half")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------
#RK5
from scipy.integrate import solve_ivp
def sir_ode(t, y, beta, gamma, N):
    """
    RHS of the SIR ODE system for solve_ivp.
    y = [S, I, R]
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def rk45_sir(beta, gamma, S0, I0, R0, t_eval, N):
    """
    Integrate the SIR model using solve_ivp with RK45.
    - t_eval should be your array of time points (e.g. t_all = np.arange(n))
    Returns S, I, R arrays aligned with t_eval.
    """
    sol = solve_ivp(
        fun=lambda t, y: sir_ode(t, y, beta, gamma, N),
        t_span=(t_eval[0], t_eval[-1]),
        y0=[S0, I0, R0],
        t_eval=t_eval,
        method="RK45")
    if not sol.success:
        # If the solver fails, return huge SSE later by handling in SSE function
        raise RuntimeError("solve_ivp failed: " + sol.message)
    S, I, R = sol.y  # shape (3, len(t_eval))
    return S, I, R

I_true_all = sir_df["I_est"].values
dates_all = sir_df["date"].values
n = len(sir_df)
t_all = np.arange(n, dtype=float)  # 0,1,2,... weekly steps

mid = n // 2  # split index

I_true_fit = I_true_all[:mid]      # first half
t_fit = t_all[:mid]

I_true_second = I_true_all[mid:]   # second half
t_second = t_all[mid:]

N = population_nigeria

S0 = sir_df["S_est"].iloc[0]
I0 = sir_df["I_est"].iloc[0]
R0 = sir_df["R_est"].iloc[0]

import numpy as np
def sir_sse_first_half_rk45(beta, gamma):
    """
    SSE between RK45 model and data for the FIRST HALF of the time series.
    """
    # Enforce positivity; if not, give huge penalty
    if beta <= 0 or gamma <= 0:
        return 1e30
    try:
        S_model, I_model, R_model = rk45_sir(
            beta=beta,
            gamma=gamma,
            S0=S0,
            I0=I0,
            R0=R0,
            t_eval=t_all,  # integrate over the whole time range
            N=N)
    except RuntimeError:
        return 1e30  # solver failed; treat as bad parameters
    # Compare only on first half
    return np.sum((I_model[:mid] - I_true_fit) ** 2)
beta_values = np.linspace(0.01, 1.0, 50)   # you can narrow this if needed
gamma_values = np.linspace(0.05, 1.0, 50)

best_sse_rk45 = np.inf
best_beta_rk45 = None
best_gamma_rk45 = None

for beta in beta_values:
    for gamma in gamma_values:
        sse = sir_sse_first_half_rk45(beta, gamma)
        if sse < best_sse_rk45:
            best_sse_rk45 = sse
            best_beta_rk45 = beta
            best_gamma_rk45 = gamma

print("RK45 best beta (fit on first half):", best_beta_rk45)
print("RK45 best gamma (fit on first half):", best_gamma_rk45)
print("RK45 best SSE on FIRST HALF:", best_sse_rk45)

S_best_rk45, I_best_rk45, R_best_rk45 = rk45_sir(
    beta=best_beta_rk45,
    gamma=best_gamma_rk45,
    S0=S0,
    I0=I0,
    R0=R0,
    t_eval=t_all,
    N=N)
# Second-half SSE for RK45
I_model_second_rk45 = I_best_rk45[mid:]
sse_second_half_rk45 = np.sum((I_model_second_rk45 - I_true_second) ** 2)
print("Euler SSE on second half:", sse_second_half)   # from Step 4
print("RK45 SSE on second half:", sse_second_half_rk45)

plt.figure(figsize=(10, 5))
plt.plot(dates_all, I_true_all, label="True I(t)", linewidth=2)

# If you kept the Euler best I(t) as I_best_euler:
plt.plot(dates_all, I_best, "--", label="Euler I(t) (best fit)")

plt.plot(dates_all, I_best_rk45, ":", label="RK45 I(t) (best fit)")

plt.axvline(dates_all[mid], color="k", linestyle=":", label="Train/test split")

plt.xlabel("Date")
plt.ylabel("Infected people")
plt.title("I(t): Euler vs RK45, fit on first half")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()