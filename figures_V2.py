#%%
# Contrast sensitivity curves (No-Opto) for ZT0 vs ZT12
# Loads pickled spikes/positions produced by run_distal_stim/test.py
# and plots mean E-cell firing rate vs contrast for both circadian conditions.
#
# Expected filenames inside `RESULTS_DIR` (examples):
#   NoOpto_ZT0_0.02_spikes.pickle
#   NoOpto_ZT0_0.02_positions.pickle
#   NoOpto_ZT12_0.02_spikes.pickle
#   NoOpto_ZT12_0.02_positions.pickle
#
# Notes:
# - Spikes pickle: list of length 10000 (E[0:8000], PV[8000:9000], SST[9000:10000]),
#   each element is a 1D np.array of spike times in ms (within a single 800 ms trial).
# - Positions pickle: (10000, 2) array, same order as spikes, positions in mm (0..1).
# - We compute firing rates as nspikes / (simtime_ms / 1000.0)  (i.e., Hz).
# - ROI: neurons within a radius (st_lat) around the visual center (0.5, 0.5).
#        This mirrors the area where the visual input is strongest in the model.
#
# Save outputs:
#   ./contrast_curves_noopto_ZT0_ZT12.png
#   ./contrast_curves_noopto_ZT0_ZT12.pdf
#   ./contrast_curve_values.csv           (table with means/SEMs per contrast)
#%%
import os
import re
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, 'code')
from help_funcs import *
ZT0color = "#8E3FE8"
ZT12color = "#FFC32B"

# --------- Paths (edit these if your layout differs) ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # this file's folder
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'work_dir', 'base_a0.07_pf0.8', 'results_2')
simname   = 'base_a0.07_pf0.8'
workdir   = 'work_dir'
widths_mm = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
CONTRAST_FIXED = 0.33
binlen    = 200  # ms time window used by fig4.py
seedlist  = np.array([1])  # we’re reading results_1; extend if you run more seeds

# --------- Analysis constants (match simulator defaults) ------
N_E = 8000
N_PV = 1000
N_SST = 1000
CIRCADIANS = ['ZT0', 'ZT12']

#%%
# Helpers
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def resolve_paths_by_sigma(base_dir, circadian, sigma_mm, contrast=CONTRAST_FIXED):
    """Find one spikes and one positions file for (circadian, contrast=0.33, sigma_mm).
    Matches both ..._dist_... and ..._fixed_... by using a wildcard.
    """
    sig_str = f"{sigma_mm:.3f}"  # test.py writes 3 decimals, e.g. 0.200
    base = f"NoOpto_{circadian}_{contrast}_*_sig{sig_str}mm"
    sp_glob = os.path.join(base_dir, base + "_spikes.pickle")
    xy_glob = os.path.join(base_dir, base + "_positions.pickle")
    sp = glob.glob(sp_glob)
    xy = glob.glob(xy_glob)
    if len(sp) != 1 or len(xy) != 1:
        raise FileNotFoundError(f"Ambiguous or missing files for pattern:\n  {sp_glob}\n  {xy_glob}\nFound: spikes={sp}, positions={xy}")
    return sp[0], xy[0]


# Containers for final curves (means ± sem over ROI E cells)

curve_mean = {'ZT0': [], 'ZT12': []}
curve_sem  = {'ZT0': [], 'ZT12': []}

for circ in ['ZT0', 'ZT12']:
    for sig in widths_mm:
        sp_path, xy_path = resolve_paths_by_sigma(RESULTS_DIR, circ, sig, contrast=CONTRAST_FIXED)
        all_spikes    = load_pickle(sp_path)
        all_positions = load_pickle(xy_path)

        ctrl_E_mean, _, ctrl_E_err, _ = getStimRateMeans_v2(
            all_spikes[:8000], all_positions[:8000], binlen=binlen
        )
        curve_mean[circ].append(ctrl_E_mean)
        curve_sem[circ].append(ctrl_E_err)

curve_mean['ZT0']  = np.array(curve_mean['ZT0'])
curve_sem['ZT0']   = np.array(curve_sem['ZT0'])
curve_mean['ZT12'] = np.array(curve_mean['ZT12'])
curve_sem['ZT12']  = np.array(curve_sem['ZT12'])




#%%
# Plot
plt.figure(figsize=(4.8,4.2), dpi=150)

plt.errorbar(widths_mm, curve_mean['ZT12'], yerr=curve_sem['ZT12'],
             fmt='o', ms=5, mec=(0.4,0.4,0.4), mfc='white', ecolor=ZT12color,
             elinewidth=1.5, capsize=3, label='ZT12 Pyr')

plt.errorbar(widths_mm, curve_mean['ZT0'],  yerr=curve_sem['ZT0'],
             fmt='o', ms=5, mec='black', mfc='white', ecolor=ZT0color,
             elinewidth=1.5, capsize=3, label='ZT0 Pyr')

# --- Naka–Rushton fit on width→rate curves (uses help_funcs) ---
# --- DoNR fit (center NR - surround NR) ---

wgrid = np.linspace(widths_mm.min(), widths_mm.max(), 400)

p0_zt0, _  = fit_naka_rushton_diff(widths_mm, curve_mean['ZT0'],  sigma=curve_sem['ZT0'])
p0_zt12, _ = fit_naka_rushton_diff(widths_mm, curve_mean['ZT12'], sigma=curve_sem['ZT12'])

fit_zt0  = naka_rushton_diff(wgrid,  *p0_zt0)
fit_zt12 = naka_rushton_diff(wgrid,  *p0_zt12)

plt.plot(wgrid, fit_zt0,  color=ZT0color, linewidth=2.0, label='ZT0 DoNR fit')
plt.plot(wgrid, fit_zt12, color=ZT12color, linewidth=2.0, linestyle='--', label='ZT12 DoNR fit')

# (Optional) report peak width from the fit
w_opt_zt0  = wgrid[np.argmax(fit_zt0)]
w_opt_zt12 = wgrid[np.argmax(fit_zt12)]
print(f"Peak σ (ZT0) ≈ {w_opt_zt0*1000:.0f} µm; Peak σ (ZT12) ≈ {w_opt_zt12*1000:.0f} µm")



plt.ylabel('E response (spks/s)')
plt.xlabel('Gaussian width σ (mm)')
plt.xticks(widths_mm)
plt.grid(True, alpha=0.25)
plt.legend(frameon=False)
plt.tight_layout()

PNG_OUT = os.path.join(PROJECT_ROOT, 'width_curves_noopto_ZT0_ZT12.png')
plt.savefig(PNG_OUT, bbox_inches='tight')
print(f"Saved figures:\n  {PNG_OUT}")
plt.show()


#%%
# ==== Heatmap config & helpers (50 µm grid) ====
GRID_UM   = 100
GRID_MM   = GRID_UM / 1000.0
BINS      = np.arange(0.0, 1.0 + 1e-12, GRID_MM)  # 0..1 mm in 50 µm steps

# Match Fig.4-style windows
SIMTIME_MS = 800.0   # per repeat
DELAY_MS   = 200.0   # visual onset delay
BINLEN_MS  = 200.0   # window length used for control vs stim

def load_data_sigma(circadian, sigma_mm, contrast=CONTRAST_FIXED):
    sp_path, xy_path = resolve_paths_by_sigma(RESULTS_DIR, circadian, sigma_mm, contrast=contrast)
    return load_pickle(sp_path), load_pickle(xy_path)

def infer_nrepeats(spike_list, simtime_ms=SIMTIME_MS):
    # Robustly infer how many repeats are in this file from the max spike time
    tmax = 0.0
    for s in spike_list:
        if len(s):
            tmax = max(tmax, float(s.max()))
    # A little slack so we don't miss the last window
    return int(np.floor((tmax + 1e-6) / simtime_ms))

def control_window_starts(spike_list):
    """Control windows start at 200 ms every repeat, length 200 ms."""
    nrepeats = infer_nrepeats(spike_list, SIMTIME_MS)
    return [DELAY_MS + k*SIMTIME_MS for k in range(nrepeats)]

def rates_over_windows(spike_list, idx_slice, starts_ms, binlen_ms=BINLEN_MS):
    """Average Hz per neuron over all control windows."""
    dur_s = (len(starts_ms) * binlen_ms) / 1000.0
    n = idx_slice.stop - idx_slice.start
    r = np.empty(n, dtype=float)
    for k, i in enumerate(range(idx_slice.start, idx_slice.stop)):
        s = spike_list[i]
        if len(s) == 0:
            r[k] = 0.0
            continue
        cnt = 0
        for t0 in starts_ms:
            cnt += np.count_nonzero((s >= t0) & (s < t0 + binlen_ms))
        r[k] = cnt / dur_s
    return r

def grid_average(positions, idx_slice, rates):
    """Mean rate per 50 µm bin. Returns array shaped (20, 20) with NaN for empty bins."""
    xy = positions[idx_slice, :]
    Hsum, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=[BINS, BINS], weights=rates)
    Hcnt, _, _   = np.histogram2d(xy[:, 0], xy[:, 1], bins=[BINS, BINS])
    with np.errstate(invalid='ignore', divide='ignore'):
        Havg = Hsum / np.maximum(Hcnt, 1)
        Havg[Hcnt == 0] = np.nan
    return Havg.T  # transpose so rows map to y for imshow(origin='lower')



#%%
# ==== Compute heatmaps for a chosen contrast ====
SELECT_SIGMA_MM = 0.15  # change as needed

maps = {}  # key: (circadian, pop) -> 2D grid of mean Hz
for circ in ['ZT0', 'ZT12']:
    spikes, pos = load_data_sigma(circ, SELECT_SIGMA_MM)
    starts = control_window_starts(spikes)  # [200, 1000, 1800, ...] ms

    # E (Pyr)
    rates_E  = rates_over_windows(spikes, slice(0, N_E), starts)
    maps[(circ, 'E')]  = grid_average(pos, slice(0, N_E), rates_E)

    # PV
    rates_PV = rates_over_windows(spikes, slice(N_E, N_E + N_PV), starts)
    maps[(circ, 'PV')] = grid_average(pos, slice(N_E, N_E + N_PV), rates_PV)

# Shared color scale across panels (helps visual comparison)
vmin = np.nanmin([maps[(c, p)] for c in ['ZT0', 'ZT12'] for p in ['E', 'PV']])
vmax = np.nanmax([maps[(c, p)] for c in ['ZT0', 'ZT12'] for p in ['E', 'PV']])

print("Control windows used (ms):", starts[:5], "... total:", len(starts))
print("Global mean Hz:",
      "ZT0 E:", np.nanmean(maps[('ZT0','E')]),
      "ZT0 PV:", np.nanmean(maps[('ZT0','PV')]),
      "ZT12 E:", np.nanmean(maps[('ZT12','E')]),
      "ZT12 PV:", np.nanmean(maps[('ZT12','PV')]))

#%%
# ==== Plot heatmaps ====
# ---- E-only heatmaps (ZT0, ZT12, and Δ = ZT12 − ZT0) ----
sp0, xy0  = load_data_sigma('ZT0',  SELECT_SIGMA_MM)
sp12,xy12 = load_data_sigma('ZT12', SELECT_SIGMA_MM)
starts0   = control_window_starts(sp0)
starts12  = control_window_starts(sp12)

ratesE0   = rates_over_windows(sp0,  slice(0, N_E), starts0)
ratesE12  = rates_over_windows(sp12, slice(0, N_E), starts12)
mapE0     = grid_average(xy0,  slice(0, N_E), ratesE0)
mapE12    = grid_average(xy12, slice(0, N_E), ratesE12)

# Absolute scale shared between ZT0 and ZT12
vminE = np.nanmin([mapE0, mapE12])
vmaxE = np.nanmax([mapE0, mapE12])

# Difference map (ZT12 − ZT0) with symmetric diverging scale
diffE = mapE12 - mapE0
mE    = np.nanmax(np.abs(diffE))

figE, axE = plt.subplots(1, 3, figsize=(12, 3.5), dpi=150, constrained_layout=True)

# ZT0
im0  = axE[0].imshow(mapE0,  origin='lower', extent=[0,1000,0,1000],
                     vmin=vminE, vmax=vmaxE, cmap='magma', aspect='equal')
axE[0].set_title(f"ZT0 Pyr — σ={int(SELECT_SIGMA_MM*1000)}µm")

# ZT12
im12 = axE[1].imshow(mapE12, origin='lower', extent=[0,1000,0,1000],
                     vmin=vminE, vmax=vmaxE, cmap='magma', aspect='equal')
axE[1].set_title(f"ZT12 Pyr — σ={int(SELECT_SIGMA_MM*1000)}µm")

# Δ (ZT12 − ZT0)
imD  = axE[2].imshow(diffE, origin='lower', extent=[0,1000,0,1000],
                     vmin=-mE, vmax= mE, cmap='coolwarm', aspect='equal')
axE[2].set_title("Pyr Δ (ZT12 − ZT0)")

for a in axE:
    a.set_xlabel('x (µm)'); a.set_ylabel('y (µm)')
    a.set_xticks([0,250,500,750,1000]); a.set_yticks([0,250,500,750,1000])

# One colorbar for absolute, one for difference
cbar_abs  = figE.colorbar(im12, ax=axE[:2].ravel().tolist(), shrink=0.9, label='Firing rate (Hz)')
cbar_diff = figE.colorbar(imD,  ax=[axE[2]],                 shrink=0.9, label='Δ Hz (ZT12−ZT0)')

# (Optional) save a dedicated filename for the 1×3 E-only figure
figE.savefig(os.path.join(PROJECT_ROOT, f'heatmaps_E_with_diff_sig={int(SELECT_SIGMA_MM*1000)}µm.png'), dpi=300)

#%%
# ---- PV maps + difference ----
ratesPV0  = rates_over_windows(sp0,  slice(N_E, N_E+N_PV), starts0)
ratesPV12 = rates_over_windows(sp12, slice(N_E, N_E+N_PV), starts12)
mapPV0    = grid_average(xy0,  slice(N_E, N_E+N_PV), ratesPV0)
mapPV12   = grid_average(xy12, slice(N_E, N_E+N_PV), ratesPV12)

# Shared scale for absolute PV maps
vminPV = np.nanmin([mapPV0, mapPV12]); vmaxPV = np.nanmax([mapPV0, mapPV12])

figPV, axPV = plt.subplots(1, 3, figsize=(12, 3.5), dpi=150, constrained_layout=True)
axPV[0].set_title("ZT0 PV");  im_pv0  = axPV[0].imshow(mapPV0,  origin='lower', extent=[0,1000,0,1000], vmin=vminPV, vmax=vmaxPV, cmap='magma')
axPV[1].set_title("ZT12 PV"); im_pv12 = axPV[1].imshow(mapPV12, origin='lower', extent=[0,1000,0,1000], vmin=vminPV, vmax=vmaxPV, cmap='magma')

# Difference map (ZT12 - ZT0) with diverging colormap centered at 0
diffPV = mapPV12 - mapPV0
m = np.nanmax(np.abs(diffPV))
axPV[2].set_title("PV Δ (ZT12 − ZT0)")
im_diff = axPV[2].imshow(diffPV, origin='lower', extent=[0,1000,0,1000], vmin=-m, vmax=m, cmap='coolwarm')

for a in axPV:
    a.set_xlabel('x (µm)'); a.set_ylabel('y (µm)')

figPV.colorbar(im_pv12, ax=axPV[:2].ravel().tolist(), shrink=0.9, label='Firing rate (Hz)')
figPV.colorbar(im_diff,  ax=[axPV[2]],                shrink=0.9, label='Δ Hz (ZT12−ZT0)')
figPV.savefig(os.path.join(PROJECT_ROOT, f'heatmaps_PV_with_diff_sig{int(SELECT_SIGMA_MM*1000)}µm.png'), dpi=300)


#%% ------------- Multi-sigma heatmaps: helpers -------------
def compute_maps_for_sigma(circadian, sigma_mm):
    """Return {'E': 20x20 Hz map, 'PV': 20x20 Hz map} for a given (ZT, σ)."""
    sp, xy  = load_data_sigma(circadian, sigma_mm)
    starts  = control_window_starts(sp)

    rates_E  = rates_over_windows(sp, slice(0, N_E), starts)
    rates_PV = rates_over_windows(sp, slice(N_E, N_E + N_PV), starts)

    mapE  = grid_average(xy, slice(0, N_E),              rates_E)
    mapPV = grid_average(xy, slice(N_E, N_E + N_PV),     rates_PV)
    return {'E': mapE, 'PV': mapPV}

def sanity_check_files(widths=widths_mm, contrast=CONTRAST_FIXED):
    """Early fail with a readable message if any (ZT, σ) file is missing."""
    missing = []
    for s in widths:
        for circ in CIRCADIANS:
            try:
                resolve_paths_by_sigma(RESULTS_DIR, circ, s, contrast=contrast)
            except FileNotFoundError as e:
                missing.append(str(e))
    if missing:
        print("\n[Missing results for some (ZT, σ):]")
        for m in missing: print(" -", m)
        raise FileNotFoundError("One or more required spike/position files were not found.")

#%% ------------- Multi-sigma heatmaps: main plotting -------------
def plot_heatmaps_grid_all_sigmas(pop='E', widths=widths_mm, save=True, dpi=300):
    """
    Build a grid of size len(widths) x 3 for a single population:
      columns: ZT0 | ZT12 | (ZT12 − ZT0)
      rows:    increasing σ in mm (e.g., 0.10 … 0.30)
    """
    assert pop in ('E', 'PV')
    sanity_check_files(widths)

    # 1) Collect maps for all (ZT, σ)
    maps = {}             # (sigma, circadian) -> 20x20 map
    diffs = {}            # sigma -> 20x20 (ZT12 − ZT0)
    for s in widths:
        m0  = compute_maps_for_sigma('ZT0',  s)[pop]
        m12 = compute_maps_for_sigma('ZT12', s)[pop]
        maps[(s, 'ZT0')]  = m0
        maps[(s, 'ZT12')] = m12
        diffs[s]          = m12 - m0

    # 2) Global color scales (fair comparison across σ)
    abs_vmin = np.nanmin([maps[(s,'ZT0')]  for s in widths] + [maps[(s,'ZT12')] for s in widths])
    abs_vmax = np.nanmax([maps[(s,'ZT0')]  for s in widths] + [maps[(s,'ZT12')] for s in widths])
    diff_max = np.nanmax([np.nanmax(np.abs(diffs[s])) for s in widths])

    # 3) Plot grid
    nrows, ncols = len(widths), 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.4*nrows), dpi=150, constrained_layout=True)
    axes = np.atleast_2d(axes)

    cmap_abs = 'magma' if pop == 'E' else 'viridis'
    for i, s in enumerate(widths):
        # ZT0
        im0 = axes[i,0].imshow(maps[(s,'ZT0')], origin='lower', extent=[0,1000,0,1000],
                               vmin=abs_vmin, vmax=abs_vmax, cmap=cmap_abs, aspect='equal')
        # ZT12
        im1 = axes[i,1].imshow(maps[(s,'ZT12')], origin='lower', extent=[0,1000,0,1000],
                               vmin=abs_vmin, vmax=abs_vmax, cmap=cmap_abs, aspect='equal')
        # Difference
        im2 = axes[i,2].imshow(diffs[s],         origin='lower', extent=[0,1000,0,1000],
                               vmin=-diff_max, vmax=diff_max,  cmap='coolwarm', aspect='equal')

        # Row label with σ in µm for quick reading
        axes[i,0].set_ylabel(f'σ = {int(s*1000)} µm')

        for j in range(ncols):
            a = axes[i,j]
            a.set_xticks([0,250,500,750,1000])
            a.set_yticks([0,250,500,750,1000])
            a.set_xlabel('x (µm)')
            a.set_ylabel('y (µm)')

    # Column titles once, at top row
    axes[0,0].set_title('ZT0')
    axes[0,1].set_title('ZT12')
    axes[0,2].set_title('ZT12 − ZT0')

    # Colorbars: one for absolute (cols 0–1), one for difference (col 2)
    cbar_abs  = fig.colorbar(im1, ax=axes[:, :2].ravel().tolist(), shrink=0.92, label='Firing rate (Hz)')
    cbar_diff = fig.colorbar(im2, ax=axes[:,  2].ravel().tolist(), shrink=0.92, label='Δ Hz (ZT12−ZT0)')

    if save:
        out = os.path.join(PROJECT_ROOT, f'heatmaps_grid_{pop}_{len(widths)}x3.png')
        fig.savefig(out, dpi=dpi)
        print(f"[saved] {out}")

    return fig

#%% ------------- Run: produce two 5×3 figures (E and PV) -------------
_ = plot_heatmaps_grid_all_sigmas(pop='E',  widths=widths_mm)
_ = plot_heatmaps_grid_all_sigmas(pop='PV', widths=widths_mm)

# %%
