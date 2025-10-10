# %%
# 1.1  Imports
from pathlib import Path
import pickle, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %%
# 1.2  Constants
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "work_dir" / "base_a0.07_pf0.0"  # adjust if you renamed the directory
SIM_DUR_MS = 5_000                # simulated time (5 s)
N_E, N_PV = 8_000, 2_000          # fixed cell counts
# 2 Data Loading
files = sorted(ROOT.rglob("*.pickle"))
print(f"Found {len(files)} pickles")

# 3.1  Pull numeric info from filenames
pat = re.compile(
    r"PV_dist(?P<laser>[0-9.]+)_p(?P<prob>[0-9.]+)_"
    r"(?P<kind>spikes|positions)\.pickle"
)

def parse_fname(f):
    m = pat.search(f.name)
    laser   = float(m["laser"])
    prob    = float(m["prob"])
    kind    = m["kind"]           # 'spikes' or 'positions'
    cond    = "ZT0" if "results_1" in f.parts else "ZT12"
    return laser, prob, kind, cond

# 3.2  Load a single pair (spikes & positions) into one DataFrame
def load_pair(spike_f, pos_f, laser, prob, cond):
    with open(spike_f, "rb") as fh: spikes = pickle.load(fh)
    with open(pos_f,   "rb") as fh: pos    = pickle.load(fh)

    ids   = np.arange(len(spikes))           # 0 … 9 999
    types = np.where(ids < N_E, "E", "PV")   # first 8 000 are E

    rate  = [len(s)/(SIM_DUR_MS/1000) for s in spikes]
    x, y  = pos[:,0], pos[:,1]

    offset_um = int(round((0.5 - laser) * np.sqrt(2) * 1000))  # physical offset

    return pd.DataFrame({
        "cond": cond,          # 'ZT0' or 'ZT12'
        "laserPos": laser,
        "offset_um": offset_um,
        "Prob_pe": prob,
        "id": ids,
        "type": types,
        "rate_Hz": rate,
        "x": x,
        "y": y,
    })

rows = []
for f in files:
    laser, prob, kind, cond = parse_fname(f)
    # pick the matching position file
    mate = f.with_name(f.name.replace("spikes", "positions")
                              if "spikes" in f.name
                              else f.name.replace("positions", "spikes"))
    # only act once per pair (skip when we're on 'positions')
    if kind == "spikes":
        df = load_pair(f, mate, laser, prob, cond)
        rows.append(df)

df = pd.concat(rows, ignore_index=True)
print(df.shape)
df.head()
# There should be 10 000 rows for every (cond, offset, prob) combo
per_pair = (df.groupby(["cond", "offset_um", "Prob_pe"])
              .size()
              .rename("rows"))
bad = per_pair[per_pair != (N_E + N_PV)]
if not bad.empty:
    raise ValueError(
        "Some pickle pairs are incomplete:\n" + bad.to_string()
    )
print("All pickle pairs contain exactly", N_E+N_PV, "neurons.")

#uncomment the next line to save the DataFrame
#df.to_csv("all_rates.csv", index=False) 

assert set(df["offset_um"]) == {20., 40., 65., 100., 150.}
sizes = df.groupby("cond").size()
expected_per_cond = (N_E + N_PV) * len(df.groupby(["offset_um", "Prob_pe"]))
assert (sizes % (N_E + N_PV) == 0).all(), (
    f"Each condition should be a whole-number multiple of {N_E+N_PV}; "
    f"got {sizes.to_dict()}"
)


pop = (df.groupby(["cond","offset_um","Prob_pe","type"])
         .rate_Hz.agg(["mean","sem"])
         .reset_index())

# %%
def hill(x, base, max_amp, x50, n):
    """baseline + amp * x^n / (x^n + x50^n)"""
    return base + max_amp * (x**n) / (x**n + x50**n)

# ---- colour map & basic settings ---------------------------------
CELL   = "E"
COLORS = {"ZT0": "purple", "ZT12": "gold"}
FIG, AXES = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

for ax, cond in zip(AXES, ["ZT0", "ZT12"]):
    sub = (pop[(pop.type == CELL) & (pop.cond == cond)]
             .sort_values("offset_um"))

    x = sub.offset_um.values         # [20,40,65,100,150]
    y = sub["mean"].values
    yerr = sub["sem"].values
    col = COLORS[cond]

    # 1) scatter the means ± SEM
    ax.errorbar(x, y, yerr=yerr, fmt="o", color=col, zorder=3, label=cond)

    # 2) fit Hill curve – start-point guesses matter with only 5 pts
    p0 = [y.min(), y.max()-y.min(), 80, 2]      # baseline, amp, x50, n
    try:
        popt, _ = curve_fit(hill, x, y, p0=p0, maxfev=10_000)
        xsmooth = np.linspace(x.min(), x.max(), 400)
        ax.plot(xsmooth, hill(xsmooth, *popt), color=col, lw=2, zorder=2)
    except RuntimeError:
        # fallback: straight line if optimiser fails
        coef = np.polyfit(x, y, 1)
        xsmooth = np.linspace(x.min(), x.max(), 400)
        ax.plot(xsmooth, np.polyval(coef, xsmooth), color=col, lw=2, zorder=2)

    # 3) cosmetics
    ax.set_title(cond)
    ax.set_xlabel("Laser–soma offset (µm)")
    ax.grid(alpha=.3)
    if cond == "ZT0":
        ax.set_ylabel("E mean firing rate (Hz)")

# ---- unified legend outside the panels ---------------------------
handles, labels = [], []
for cond in ["ZT0", "ZT12"]:
    h = AXES[0].plot([], [], color=COLORS[cond], marker="o", ls="-")[0]
    handles.append(h); labels.append(cond)
FIG.legend(handles, labels, title="Condition", loc="center left",
           bbox_to_anchor=(0.87, 0.5), frameon=False)

FIG.suptitle("Offset vs firing rate – Excitatory population")
FIG.tight_layout(rect=[0, 0, 0.9, 1])   # leave space at right for legend
FIG.savefig("Offset Distance VS Pyr Firing.png", dpi=300)
plt.show()

# %%
def hill(x, base, amp, x50, n):
    return base + amp * (x**n) / (x**n + x50**n)

CELL   = "E"
COLMAP = {"ZT0": "purple", "ZT12": "gold"}

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

for ax, cond in zip(axes, ["ZT0", "ZT12"]):
    # ── pick rows for this circadian condition and E cells
    sub = (pop[(pop.type == CELL) & (pop.cond == cond)]
             .sort_values("Prob_pe"))          # 5 rows, one per prob
    
    x     = sub.Prob_pe.values                 # e.g. [0.026 … 0.595]
    y     = sub["mean"].values                 # mean E firing
    yerr  = sub["sem"].values
    color = COLMAP[cond]

    # 1) plot the raw means ± SEM
    ax.errorbar(x, y, yerr=yerr, fmt="o", ms=6,
                color=color, label=cond, zorder=3)

    # 2) Hill fit through the five points
    p0 = [y.min(), y.max()-y.min(), 0.2, 2]    # sensible initial guess
    try:
        popt, _ = curve_fit(hill, x, y, p0=p0, maxfev=8000)
        xfine = np.linspace(x.min(), x.max(), 400)
        ax.plot(xfine, hill(xfine, *popt), color=color, lw=2, zorder=2)
    except RuntimeError:                       # optimiser failed → straight line
        coef = np.polyfit(x, y, 1)
        ax.plot(xfine, np.polyval(coef, xfine), color=color, lw=2, zorder=2)

    # 3) panel cosmetics
    ax.set_title(cond)
    ax.set_xlabel("PV→E connection probability")
    ax.grid(alpha=.3)
    if cond == "ZT0":
        ax.set_ylabel("E mean firing rate (Hz)")

# ---- legend outside the axes --------------------------------
handles = [axes[0].lines[0], axes[1].lines[0]]   # first plotted marker in each panel
labels  = ["ZT0", "ZT12"]
fig.legend(handles, labels, title="Condition",
           loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

fig.suptitle("E-cell firing vs PV→E connection probability")
fig.tight_layout(rect=[0, 0, 0.9, 1])            # leave room for legend
#fig.savefig("Prob_vs_Firing_E.png", dpi=300)
plt.show()
# %%
