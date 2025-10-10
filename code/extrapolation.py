# %%
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
_HERE = Path(__file__).resolve().parent
zt0  = pd.read_csv(_HERE / "ZT0.csv").sort_values("distance")
zt12 = pd.read_csv(_HERE / "ZT12.csv").sort_values("distance")

#locate the emperical peak
pk0_idx  = zt0["probability"].idxmax()
pk12_idx = zt12["probability"].idxmax()
x_pk0,  y_pk0  = zt0.loc[pk0_idx,  ["distance","probability"]]
x_pk12, y_pk12 = zt12.loc[pk12_idx, ["distance","probability"]]

def gamma_kernel(x, A, alpha, beta):
    return A * np.power(x, alpha-1) * np.exp(-beta*x)

def make_guess(df):
    x_peak = df.loc[df["probability"].idxmax(), "distance"]
    y_peak = df["probability"].max()
    alpha0 = 3.0                                     # modestly > 1 so curve rises
    beta0  = (alpha0 - 1) / max(x_peak, 1e-3)        # puts the peak near x_peak
    A0     = (y_peak) / (x_peak**(alpha0-1) * np.exp(-beta0*x_peak))
    return [A0, alpha0, beta0]

p0_zt0  = make_guess(zt0)
p0_zt12 = make_guess(zt12)
bounds = (0, np.inf)      # keep A, alpha, beta ≥ 0
lower0 = [0, 0, 0.045]          #  β  must be ≥ 0.045 µm⁻¹   (≈ half-life 15 µm)
upper0 = [np.inf, np.inf, np.inf]

pars0,  _ = curve_fit(
    gamma_kernel,
    zt0["distance"],  zt0["probability"],
    p0=p0_zt0, bounds=(lower0, upper0), maxfev=10_000
)

pars12, _ = curve_fit(
    gamma_kernel,
    zt12["distance"], zt12["probability"],
    p0=p0_zt12, bounds=(lower0, upper0), maxfev=10_000
)
print("ZT0 parms:", pars0)      # [A, α, β]
print("ZT12 parms:", pars12)

#extrapolating
x_fut = np.linspace(50, 120, 800)    # future distances (µm)
pred0  = gamma_kernel(x_fut, *pars0)
pred12 = gamma_kernel(x_fut, *pars12)
#for longer graphical representation
x_fut_long = np.linspace(50, 250, 1000)          # 0–250 µm, smoother curve
y0_long  = gamma_kernel(x_fut_long, *pars0)
y12_long = gamma_kernel(x_fut_long, *pars12)
# Optional: set the default font/size once for the whole script
mpl.rcParams.update({
    "font.size":     10,
    "axes.labelsize":12,
    "axes.titlesize":14,
    "xtick.major.size":0,   # turn off major ticks if you want the clean look
    "ytick.major.size":0,
})


# 2.  build an x-grid that includes BOTH data range and extrapolation
x_full = np.concatenate([zt0["distance"], x_fut])  # ensures the fit is drawn
x_full = np.sort(np.unique(x_full))                # no duplicates, sorted
y0_full  = gamma_kernel(x_full, *pars0)
y12_full = gamma_kernel(x_full, *pars12)
# 2b.  for longer extrapolation, use the same x-grid
x_full_long = np.concatenate([zt0["distance"], x_fut_long])
x_full_long = np.sort(np.unique(x_full_long))
y0_full_long  = gamma_kernel(x_full_long, *pars0)
y12_full_long = gamma_kernel(x_full_long, *pars12)

# 𝟛.  start the figure   (6×3.5 inches roughly matches the proportion you sent)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.set_facecolor("#7f7f7f20")               # light grey panel  (20 % opacity)

#  (a) raw data points
ax.errorbar(
    zt0["distance"],  zt0["probability"],
    yerr=zt0.get("sem"),            # <-- only if you saved SE/SEM in the CSV
    fmt="o",  markersize=6,         markerfacecolor="#8E3FE8",  #   purple
    markeredgecolor="black",   ecolor="#8E3FE8",  elinewidth=1,
    label="ZT0 data",
)
ax.errorbar(
    zt12["distance"], zt12["probability"],
    yerr=zt12.get("sem"),
    fmt="o",  markersize=6,         markerfacecolor="#FFC32B",  #   gold
    markeredgecolor="black",   ecolor="#FFC32B", elinewidth=1,
    label="ZT12 data"
)

#  (b) fitted curves (solid vs dashed)
ax.plot(x_full, y0_full,  lw=2.2, color="#8E3FE8",  label="ZT0 fit")
ax.plot(x_full, y12_full, lw=2.2, color="#FFC32B", ls="--", label="ZT12 fit")

#  (c) cosmetic tweaks to echo the published figure
ax.set_xlim(0, 50)                 # match the same x-span
ax.set_ylim(0, 1.05)               # a tiny head-room avoids clipping markers
ax.set_xlabel("Pyr-PV distance (µm)")
ax.set_ylabel("Conn. Prob.")
ax.spines[['right', 'top']].set_visible(False)  # minimalist frame
ax.legend(frameon=False, loc="upper right")

plt.tight_layout()
plt.show()
fig.savefig("Pyr-PV Distance VS Conn.Prob.png", dpi=300)

# Panel B – identical cosmetics, but showing the full 0–250 µm span
fig2, ax2 = plt.subplots(figsize=(6, 3.5))
ax2.set_facecolor("#7f7f7f20")

# (a) raw data
ax2.errorbar(
    zt0["distance"],  zt0["probability"],
    yerr=zt0.get("sem"),
    fmt="o",  markersize=6,
    markerfacecolor="#8E3FE8", markeredgecolor="black",
    ecolor="#8E3FE8", elinewidth=1, label="ZT0 data",
)
ax2.errorbar(
    zt12["distance"], zt12["probability"],
    yerr=zt12.get("sem"),
    fmt="o",  markersize=6,
    markerfacecolor="#FFC32B", markeredgecolor="black",
    ecolor="#FFC32B", elinewidth=1, label="ZT12 data",
)

# (b) fitted curves
ax2.plot(x_full_long, y0_full_long,  lw=2.2,
         color="#8E3FE8",  label="ZT0 fit")
ax2.plot(x_full_long, y12_full_long, lw=2.2,
         color="#FFC32B", ls="--", label="ZT12 fit")

# (c) cosmetics – only the x-limit changes
ax2.set_xlim(0, 250)           # ← shows the extrapolation
ax2.set_ylim(0, 1.05)
ax2.set_xlabel("Pyr-PV distance (µm)")
ax2.set_ylabel("Conn. Prob.")
ax2.spines[['right', 'top']].set_visible(False)
ax2.legend(frameon=False, loc="upper right")

plt.tight_layout()
plt.show()
fig2.savefig("Pyr-PV Distance VS Conn.Prob., extrapolated.png", dpi=300)
# %%
# to obtain the predicted values:
#ZT0 prediction at a specific distance
x_value_0 = 100  #edit this line for different distances
x_prediction_0 = gamma_kernel(x_value_0, *pars0)
print(f"Predicted connection probability at {x_value_0} µm (ZT0): {x_prediction_0:.4f}")
#ZT12 prediction at a specific distance
x_value_12 = 100  #edit this line for different distances
x_prediction_12 = gamma_kernel(x_value_12, *pars12)
print(f"Predicted connection probability at {x_value_12} µm (ZT12): {x_prediction_12:.4f}")

#for a range of values:
x_vals = np.linspace(10, 250, 240)  # edit this line for different ranges
predictions_0 = gamma_kernel(x_vals, *pars0)
predictions_12 = gamma_kernel(x_vals, *pars12)

df_extrapolated = pd.DataFrame({
    "distance_um": x_vals,
    "ZT0_prob": predictions_0,
    "ZT12_prob": predictions_12
})

# 4. Save to CSV
df_extrapolated.to_csv("extrapolated_probabilities.csv", index=False)

print("Extrapolated values saved to 'extrapolated_probabilities.csv'")
# %%
