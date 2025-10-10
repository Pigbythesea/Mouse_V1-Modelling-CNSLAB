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
import pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, 'code')
from help_funcs import *
ZT0color = "#8E3FE8"
ZT12color = "#FFC32B"

# --------- Paths (edit these if your layout differs) ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # this file's folder
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'work_dir', 'base_a0.07_pf0.8', 'results_1')
simname   = 'base_a0.07_pf0.8'
workdir   = 'work_dir'
contrasts = np.array([0.02, 0.05, 0.1, 0.18, 0.33])
binlen    = 200  # ms time window used by fig4.py
seedlist  = np.array([1])  # we’re reading results_1; extend if you run more seeds

# Output paths
PNG_OUT = os.path.join(PROJECT_ROOT, 'contrast_curves_noopto_ZT0_ZT12.png')
PDF_OUT = os.path.join(PROJECT_ROOT, 'contrast_curves_noopto_ZT0_ZT12.pdf')
CSV_OUT = os.path.join(PROJECT_ROOT, 'contrast_curve_values.csv')

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

# Containers for final curves (means ± sem over ROI E cells)
curve_mean = {'ZT0': [], 'ZT12': []}
curve_sem  = {'ZT0': [], 'ZT12': []}

for circ in ['ZT0', 'ZT12']:
    # folder holding the files for this seed (results_1)
    folder = f'{workdir}/{simname}/results_{seedlist[0]}/'
    for cont in contrasts:
        # Load spikes/positions for this circadian + contrast
        with open(os.path.join(folder, f'NoOpto_{circ}_{cont}_spikes.pickle'), 'rb') as f:
            all_spikes = pickle.load(f)
        with open(os.path.join(folder, f'NoOpto_{circ}_{cont}_positions.pickle'), 'rb') as f:
            all_positions = pickle.load(f)

        # E cells are first 8000; pass E spikes/positions to helper
        # Helper returns: ctrl_mean, stim_mean, ctrl_err, stim_err
        ctrl_E_mean, _, ctrl_E_err, _ = getStimRateMeans_v2(
            all_spikes[:8000], all_positions[:8000], binlen=binlen
        )

        curve_mean[circ].append(ctrl_E_mean)
        curve_sem[circ].append(ctrl_E_err)

# convert to arrays for fitting/plotting
curve_mean['ZT0'] = np.array(curve_mean['ZT0'])
curve_sem['ZT0']  = np.array(curve_sem['ZT0'])
curve_mean['ZT12'] = np.array(curve_mean['ZT12'])
curve_sem['ZT12']  = np.array(curve_sem['ZT12'])
#optional fixed value as control inherited from original code project
curve_mean['FIXED'] = []
curve_sem['FIXED']  = []

folder = f'{workdir}/{simname}/results_{seedlist[0]}/'
for cont in contrasts:
    with open(os.path.join(folder, f'NoOpto_ZT0_{cont}_fixed_spikes.pickle'), 'rb') as f:
        all_spikes = pickle.load(f)
    with open(os.path.join(folder, f'NoOpto_ZT0_{cont}_fixed_positions.pickle'), 'rb') as f:
        all_positions = pickle.load(f)

    m, _, se, _ = getStimRateMeans_v2(all_spikes[:8000], all_positions[:8000], binlen=binlen)
    curve_mean['FIXED'].append(m)
    curve_sem['FIXED'].append(se)

curve_mean['FIXED'] = np.array(curve_mean['FIXED'])
curve_sem['FIXED']  = np.array(curve_sem['FIXED'])



#%%
# Plot
# Fit Naka–Rushton to each circadian curve
def fit_curve(x, y, yerr):
    params, cov = fit_naka_rushton(x, y, sigma=yerr)
    return params  # (m, C, n, k)

params_ZT0  = fit_curve(contrasts[:], curve_mean['ZT0'][:],  curve_sem['ZT0'][:])
params_ZT12 = fit_curve(contrasts[:], curve_mean['ZT12'][:], curve_sem['ZT12'][:])

cgrid = np.arange(0.02, 0.331, 0.001)
fit_ZT0  = naka_rushton(cgrid, *params_ZT0)
fit_ZT12 = naka_rushton(cgrid, *params_ZT12)

#optional fixed value as control inherited from original code project
params_FIXED = fit_curve(contrasts[:], curve_mean['FIXED'][:], curve_sem['FIXED'][:])
fit_FIXED    = naka_rushton(cgrid, *params_FIXED)

plt.figure(figsize=(4.5,4), dpi=150)

# scatter means with error bars (like fig4)
plt.errorbar(contrasts*100, curve_mean['ZT12'], yerr=curve_sem['ZT12'],
             fmt='o', ms=5, mec=(0.4,0.4,0.4), mfc='white', ecolor=ZT12color,
             elinewidth=1.5, capsize=3, label='ZT12 Pyr')
plt.errorbar(contrasts*100, curve_mean['ZT0'],  yerr=curve_sem['ZT0'],
             fmt='o', ms=5, mec='black', mfc='white', ecolor=ZT0color,
             elinewidth=1.5, capsize=3, label='ZT0 Pyr')
plt.errorbar(contrasts*100, curve_mean['FIXED'], yerr=curve_sem['FIXED'],
             fmt='s', ms=5, mec='black', mfc='white', ecolor='black',
             elinewidth=1.5, capsize=3, label='Control')



# fitted curves
plt.plot(cgrid*100, fit_ZT0,  color=ZT0color, linewidth=2.5)
plt.plot(cgrid*100, fit_ZT12, color=ZT12color, linewidth=2.5, linestyle='--')
plt.plot(cgrid*100, fit_FIXED, linewidth=2.5, linestyle=':', color='black')


plt.ylabel('E response (spks/s)')
plt.xlabel('Contrast (%)')
plt.xlim(0, 33)
plt.xticks([0,10,20,30])
plt.grid(True, alpha=0.25)
plt.legend(frameon=False)
plt.tight_layout()

# same filenames as before (or change if you prefer)
plt.savefig(PNG_OUT, bbox_inches='tight')
#plt.savefig(PDF_OUT, bbox_inches='tight')
print(f"Saved figures:\n  {PNG_OUT}")
plt.show()

#%%
# ==== Heatmap config & helpers (50 µm grid) ====
GRID_UM   = 50
GRID_MM   = GRID_UM / 1000.0
BINS      = np.arange(0.0, 1.0 + 1e-12, GRID_MM)  # 0..1 mm in 50 µm steps

# Match Fig.4-style windows
SIMTIME_MS = 800.0   # per repeat
DELAY_MS   = 200.0   # visual onset delay
BINLEN_MS  = 200.0   # window length used for control vs stim

def load_data(circadian, contrast):
    sp = load_pickle(os.path.join(RESULTS_DIR, f'NoOpto_{circadian}_{contrast}_spikes.pickle'))
    xy = load_pickle(os.path.join(RESULTS_DIR, f'NoOpto_{circadian}_{contrast}_positions.pickle'))
    return sp, xy

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
SELECT_CONTRAST = 0.33  # change as needed

maps = {}  # key: (circadian, pop) -> 2D grid of mean Hz
for circ in ['ZT0', 'ZT12']:
    spikes, pos = load_data(circ, SELECT_CONTRAST)
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
sp0, xy0  = load_data('ZT0',  SELECT_CONTRAST)
sp12,xy12 = load_data('ZT12', SELECT_CONTRAST)
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
axE[0].set_title(f"ZT0 Pyr — {int(SELECT_CONTRAST*100)}%")

# ZT12
im12 = axE[1].imshow(mapE12, origin='lower', extent=[0,1000,0,1000],
                     vmin=vminE, vmax=vmaxE, cmap='magma', aspect='equal')
axE[1].set_title(f"ZT12 Pyr — {int(SELECT_CONTRAST*100)}%")

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
figE.savefig(os.path.join(PROJECT_ROOT, f'heatmaps_E_with_diff_{int(SELECT_CONTRAST*100)}pct.png'), dpi=300)

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


#%%
# ==== Batch all contrasts (saves one figure per contrast) ====
for cont in contrasts:
    # Build E-only maps for both circadians
    mapsE = {}
    for circ in ['ZT0', 'ZT12']:
        spikes, pos = load_data(circ, cont)
        starts   = control_window_starts(spikes)
        rates_E  = rates_over_windows(spikes, slice(0, N_E), starts)
        mapsE[circ] = grid_average(pos, slice(0, N_E), rates_E)

    # Absolute scale shared across ZT0/ZT12
    vminE = np.nanmin([mapsE['ZT0'], mapsE['ZT12']])
    vmaxE = np.nanmax([mapsE['ZT0'], mapsE['ZT12']])

    # Difference map (ZT12 − ZT0)
    diffE = mapsE['ZT12'] - mapsE['ZT0']
    mE    = np.nanmax(np.abs(diffE))

    # Plot 1×3: ZT0, ZT12, Δ
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=150, constrained_layout=True)

    im0   = axes[0].imshow(mapsE['ZT0'],  origin='lower', extent=[0,1000,0,1000],
                           vmin=vminE, vmax=vmaxE, cmap='magma', aspect='equal')
    axes[0].set_title(f"ZT0 Pyr — {int(cont*100)}%")

    im12  = axes[1].imshow(mapsE['ZT12'], origin='lower', extent=[0,1000,0,1000],
                           vmin=vminE, vmax=vmaxE, cmap='magma', aspect='equal')
    axes[1].set_title(f"ZT12 Pyr — {int(cont*100)}%")

    imD   = axes[2].imshow(diffE, origin='lower', extent=[0,1000,0,1000],
                           vmin=-mE, vmax= mE, cmap='coolwarm', aspect='equal')
    axes[2].set_title("Pyr Δ (ZT12 − ZT0)")

    for ax in axes:
        ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
        ax.set_xticks([0,250,500,750,1000]); ax.set_yticks([0,250,500,750,1000])

    # Colorbars: absolute (shared for first two), and diff
    fig.colorbar(im12, ax=axes[:2].ravel().tolist(), shrink=0.85, label='Firing rate (Hz)')
    fig.colorbar(imD,  ax=[axes[2]],                 shrink=0.85, label='Δ Hz (ZT12−ZT0)')

    # Save
    fig.savefig(os.path.join(PROJECT_ROOT, f'heatmaps_E_with_diff_{int(cont*100)}pct.png'), dpi=300)
    fig.savefig(os.path.join(PROJECT_ROOT, f'heatmaps_E_with_diff_{int(cont*100)}pct.pdf'))
    plt.close(fig)
