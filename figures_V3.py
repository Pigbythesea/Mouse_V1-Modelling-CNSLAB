#%%
# Width-response curves (No-Opto) for ZT0 vs ZT12
# Loads pickled spikes/positions produced by testv2.py
# and plots mean E-cell firing rate vs RECTANGLE WIDTH for both circadian conditions.
#
# File naming assumed (examples):
#   NoOpto_ZT0_0.33_dist_w0.200mm_spikes.pickle
#   NoOpto_ZT0_0.33_dist_w0.200mm_positions.pickle
# (Backward compatibility: if *_w*.mm not found, we try *_sig*.mm)
#
# Notes:
# - Spikes pickle: list of length 10000 (E[0:8000], PV[8000:9000], SST[9000:10000]),
#   each element is a 1D np.array of spike times in ms (within a single 800 ms trial).
# - Positions pickle: (10000, 2) array, same order as spikes, positions in mm (0..1).
# - We compute firing rates as nspikes / (simtime_ms / 1000.0)  (i.e., Hz).
#
# Save outputs:
#   ./width_curves_noopto_ZT0_ZT12.png
#   ./heatmaps_E_with_diff_w=XXXum.png
#   ./heatmaps_PV_with_diff_w=XXXum.png
#   ./heatmaps_grid_{E|PV}_{Nx3}.png
#%%
import os
import re
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# If this script lives at repo root and help_funcs.py is in ./code/
sys.path.insert(1, 'code')
from help_funcs import *  # uses your existing helpers (incl. NR fit funcs)

# --------- Colors ----------
ZT0color = "#8E3FE8"
ZT12color = "#FFC32B"

# --------- Paths (edit these if your layout differs) ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # this file's folder
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'work_dir', 'base_a0.07_pf0.8', 'results_5')
simname     = 'base_a0.07_pf0.8'
workdir     = 'work_dir'

# Sweep of rectangle widths (mm)
widths_mm   = np.array([0.10, 0.15, 0.20, 0.25, 0.30])

# Use the same contrast you used when generating the files
CONTRAST_FIXED = 0.33

# Match your earlier figure code
binlen    = 200  # ms time window used by fig4.py
seedlist  = np.array([1])  # extend if you ran more seeds / result folders

# --------- Population sizes (match simulator defaults) ----------
N_E   = 8000
N_PV  = 1000
N_SST = 1000
CIRCADIANS = ['ZT0', 'ZT12']

#%%
# --------- IO helpers ----------
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _resolve_pair(base_dir, base_pattern):
    """Return exactly one spikes + one positions file for a glob base_pattern."""
    sp_glob = os.path.join(base_dir, base_pattern + "_spikes.pickle")
    xy_glob = os.path.join(base_dir, base_pattern + "_positions.pickle")
    sp = glob.glob(sp_glob)
    xy = glob.glob(xy_glob)
    if len(sp) != 1 or len(xy) != 1:
        return None, None
    return sp[0], xy[0]

def resolve_paths_by_width(base_dir, circadian, width_mm, contrast=CONTRAST_FIXED):
    """
    Prefer rectangle-tagged files (*_w{:.3f}mm_*). If not found, fall back to old
    Gaussian-tagged files (*_sig{:.3f}mm_*), for backward compatibility.
    """
    tag = f"{width_mm:.3f}"
    # prefer rectangle files
    base_w = f"NoOpto_{circadian}_{contrast}_*_w{tag}mm"
    sp, xy = _resolve_pair(base_dir, base_w)
    if sp and xy:
        return sp, xy
    # fallback to legacy Gaussian tags
    base_sig = f"NoOpto_{circadian}_{contrast}_*_sig{tag}mm"
    sp, xy = _resolve_pair(base_dir, base_sig)
    if sp and xy:
        return sp, xy
    raise FileNotFoundError(
        f"Missing files for both patterns:\n  {os.path.join(base_dir, base_w)}_[spikes|positions].pickle\n"
        f"  {os.path.join(base_dir, base_sig)}_[spikes|positions].pickle"
    )

# Containers for final curves (means ± sem over ROI E cells)
curve_mean = {'ZT0': [], 'ZT12': []}
curve_sem  = {'ZT0': [], 'ZT12': []}

#%%
# --------- Build width→response curves (E population) ----------
for circ in CIRCADIANS:
    for w in widths_mm:
        sp_path, xy_path = resolve_paths_by_width(RESULTS_DIR, circ, w, contrast=CONTRAST_FIXED)
        all_spikes    = load_pickle(sp_path)
        all_positions = load_pickle(xy_path)

        # existing helper: computes ROI Pyr rate mean/SEM over a window (from your codebase)
        ctrl_E_mean, _, ctrl_E_err, _ = getStimRateMeans_v2(
            all_spikes[:N_E], all_positions[:N_E], binlen=binlen
        )
        curve_mean[circ].append(ctrl_E_mean)
        curve_sem[circ].append(ctrl_E_err)

curve_mean = {k: np.array(v) for k, v in curve_mean.items()}
curve_sem  = {k: np.array(v) for k, v in curve_sem.items()}

#%%
# --------- Plot: E response vs rectangle width ----------
plt.figure(figsize=(4.8,4.2), dpi=150)

plt.errorbar(widths_mm, curve_mean['ZT12'], yerr=curve_sem['ZT12'],
             fmt='o', ms=5, mec=(0.4,0.4,0.4), mfc='white', ecolor=ZT12color,
             elinewidth=1.5, capsize=3, label='ZT12 Pyr')

plt.errorbar(widths_mm, curve_mean['ZT0'],  yerr=curve_sem['ZT0'],
             fmt='o', ms=5, mec='black', mfc='white', ecolor=ZT0color,
             elinewidth=1.5, capsize=3, label='ZT0 Pyr')

# Naka–Rushton (difference-of-two NR) fit over widths
wgrid = np.linspace(widths_mm.min(), widths_mm.max(), 400)
p0_zt0,  _ = fit_naka_rushton_diff(widths_mm, curve_mean['ZT0'],  sigma=curve_sem['ZT0'])
p0_zt12, _ = fit_naka_rushton_diff(widths_mm, curve_mean['ZT12'], sigma=curve_sem['ZT12'])
fit_zt0  = naka_rushton_diff(wgrid,  *p0_zt0)
fit_zt12 = naka_rushton_diff(wgrid,  *p0_zt12)

plt.plot(wgrid, fit_zt0,  color=ZT0color,  linewidth=2.0, label='ZT0 DoNR fit')
plt.plot(wgrid, fit_zt12, color=ZT12color, linewidth=2.0, linestyle='--', label='ZT12 DoNR fit')

# Report peak width from fits
w_opt_zt0  = wgrid[np.argmax(fit_zt0)]
w_opt_zt12 = wgrid[np.argmax(fit_zt12)]
print(f"Peak width (ZT0) ≈ {w_opt_zt0*1000:.0f} µm; Peak width (ZT12) ≈ {w_opt_zt12*1000:.0f} µm")

plt.ylabel('E response (spks/s)')
plt.xlabel('Rectangle width w (mm)')
plt.xticks(widths_mm)
plt.grid(True, alpha=0.25)
plt.legend(frameon=False)
plt.tight_layout()

PNG_OUT = os.path.join(PROJECT_ROOT, 'width_curves_noopto_ZT0_ZT12.png')
#plt.savefig(PNG_OUT, bbox_inches='tight')
#print(f"Saved figures:\n  {PNG_OUT}")
plt.show()

#%%
# ==== Heatmap config & helpers (50 µm grid) ====
GRID_UM   = 100
GRID_MM   = GRID_UM / 1000.0
BINS      = np.arange(0.0, 1.0 + 1e-12, GRID_MM)  # 0..1 mm in 50 µm steps

# Match prior window logic used in your figs (800 ms trial, 200 ms window starting at 200 ms)
SIMTIME_MS = 800.0
DELAY_MS   = 200.0
BINLEN_MS  = 200.0

def load_data_width(circadian, width_mm, contrast=CONTRAST_FIXED):
    sp_path, xy_path = resolve_paths_by_width(RESULTS_DIR, circadian, width_mm, contrast=contrast)
    return load_pickle(sp_path), load_pickle(xy_path)

def infer_nrepeats(spike_list, simtime_ms=SIMTIME_MS):
    tmax = 0.0
    for s in spike_list:
        if len(s):
            tmax = max(tmax, float(s.max()))
    return int(np.floor((tmax + 1e-6) / simtime_ms))

def control_window_starts(spike_list):
    """Window starts at 200 ms of each repeat; length 200 ms."""
    nrepeats = infer_nrepeats(spike_list, SIMTIME_MS)
    return [DELAY_MS + k*SIMTIME_MS for k in range(nrepeats)]

def rates_over_windows(spike_list, idx_slice, starts_ms, binlen_ms=BINLEN_MS):
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
    return Havg.T  # rows map to y for imshow(origin='lower')

#%%
# ==== Compute heatmaps for a chosen width ====
SELECT_WIDTH_MM = 0.30  # change as needed

maps = {}  # key: (circadian, pop) -> 2D grid of mean Hz
for circ in CIRCADIANS:
    spikes, pos = load_data_width(circ, SELECT_WIDTH_MM)
    starts = control_window_starts(spikes)  # [200, 1000, 1800, ...] ms

    # E (Pyr)
    rates_E  = rates_over_windows(spikes, slice(0, N_E), starts)
    maps[(circ, 'E')]  = grid_average(pos, slice(0, N_E), rates_E)

    # PV
    rates_PV = rates_over_windows(spikes, slice(N_E, N_E + N_PV), starts)
    maps[(circ, 'PV')] = grid_average(pos, slice(N_E, N_E + N_PV), rates_PV)

# Shared color scale across panels (helps visual comparison)
vmin = np.nanmin([maps[(c, p)] for c in CIRCADIANS for p in ['E', 'PV']])
vmax = np.nanmax([maps[(c, p)] for c in CIRCADIANS for p in ['E', 'PV']])

print("Control windows used (ms):", starts[:5], "... total:", len(starts))
print("Global mean Hz:",
      "ZT0 E:", np.nanmean(maps[('ZT0','E')]),
      "ZT0 PV:", np.nanmean(maps[('ZT0','PV')]),
      "ZT12 E:", np.nanmean(maps[('ZT12','E')]),
      "ZT12 PV:", np.nanmean(maps[('ZT12','PV')]))

#%%
# ==== Plot heatmaps (E) for one width ====
sp0, xy0  = load_data_width('ZT0',  SELECT_WIDTH_MM)
sp12,xy12 = load_data_width('ZT12', SELECT_WIDTH_MM)
starts0   = control_window_starts(sp0)
starts12  = control_window_starts(sp12)

ratesE0   = rates_over_windows(sp0,  slice(0, N_E), starts0)
ratesE12  = rates_over_windows(sp12, slice(0, N_E), starts12)
mapE0     = grid_average(xy0,  slice(0, N_E), ratesE0)
mapE12    = grid_average(xy12, slice(0, N_E), ratesE12)

vminE = np.nanmin([mapE0, mapE12])
vmaxE = np.nanmax([mapE0, mapE12])

diffE = mapE12 - mapE0
mE    = np.nanmax(np.abs(diffE))

figE, axE = plt.subplots(1, 3, figsize=(12, 3.5), dpi=150, constrained_layout=True)

# ZT0
im0  = axE[0].imshow(mapE0,  origin='lower', extent=[0,1000,0,1000],
                     vmin=vminE, vmax=vmaxE, cmap='magma', aspect='equal')
axE[0].set_title(f"ZT0 Pyr — w={int(SELECT_WIDTH_MM*1000)}µm")

# ZT12
im12 = axE[1].imshow(mapE12, origin='lower', extent=[0,1000,0,1000],
                     vmin=vminE, vmax=vmaxE, cmap='magma', aspect='equal')
axE[1].set_title(f"ZT12 Pyr — w={int(SELECT_WIDTH_MM*1000)}µm")

# Δ (ZT12 − ZT0)
imD  = axE[2].imshow(diffE, origin='lower', extent=[0,1000,0,1000],
                     vmin=-mE, vmax= mE, cmap='coolwarm', aspect='equal')
axE[2].set_title("Pyr Δ (ZT12 − ZT0)")

for a in axE:
    a.set_xlabel('x (µm)'); a.set_ylabel('y (µm)')
    a.set_xticks([0,250,500,750,1000]); a.set_yticks([0,250,500,750,1000])

# Colorbars
figE.colorbar(im12, ax=axE[:2].ravel().tolist(), shrink=0.9, label='Firing rate (Hz)')
figE.colorbar(imD,  ax=[axE[2]],                 shrink=0.9, label='Δ Hz (ZT12−ZT0)')

#figE.savefig(os.path.join(PROJECT_ROOT, f'heatmaps_E_with_diff_w={int(SELECT_WIDTH_MM*1000)}um.png'), dpi=300)

#%%
# ==== PV heatmaps (same width) ====
ratesPV0  = rates_over_windows(sp0,  slice(N_E, N_E+N_PV), starts0)
ratesPV12 = rates_over_windows(sp12, slice(N_E, N_E+N_PV), starts12)
mapPV0    = grid_average(xy0,  slice(N_E, N_E+N_PV), ratesPV0)
mapPV12   = grid_average(xy12, slice(N_E, N_E+N_PV), ratesPV12)

vminPV = np.nanmin([mapPV0, mapPV12]); vmaxPV = np.nanmax([mapPV0, mapPV12])

figPV, axPV = plt.subplots(1, 3, figsize=(12, 3.5), dpi=150, constrained_layout=True)
axPV[0].set_title("ZT0 PV");  im_pv0  = axPV[0].imshow(mapPV0,  origin='lower', extent=[0,1000,0,1000], vmin=vminPV, vmax=vmaxPV, cmap='magma')
axPV[1].set_title("ZT12 PV"); im_pv12 = axPV[1].imshow(mapPV12, origin='lower', extent=[0,1000,0,1000], vmin=vminPV, vmax=vmaxPV, cmap='magma')

diffPV = mapPV12 - mapPV0
m = np.nanmax(np.abs(diffPV))
axPV[2].set_title("PV Δ (ZT12 − ZT0)")
im_diff = axPV[2].imshow(diffPV, origin='lower', extent=[0,1000,0,1000], vmin=-m, vmax=m, cmap='coolwarm')

for a in axPV:
    a.set_xlabel('x (µm)'); a.set_ylabel('y (µm)')

figPV.colorbar(im_pv12, ax=axPV[:2].ravel().tolist(), shrink=0.9, label='Firing rate (Hz)')
figPV.colorbar(im_diff,  ax=[axPV[2]],                shrink=0.9, label='Δ Hz (ZT12−ZT0)')
#figPV.savefig(os.path.join(PROJECT_ROOT, f'heatmaps_PV_with_diff_w={int(SELECT_WIDTH_MM*1000)}um.png'), dpi=300)

#%% ------------- Multi-width heatmaps: helpers -------------
def compute_maps_for_width(circadian, width_mm):
    """Return {'E': 20x20 Hz map, 'PV': 20x20 Hz map} for a given (ZT, width)."""
    sp, xy  = load_data_width(circadian, width_mm)
    starts  = control_window_starts(sp)
    rates_E  = rates_over_windows(sp, slice(0, N_E), starts)
    rates_PV = rates_over_windows(sp, slice(N_E, N_E + N_PV), starts)
    mapE  = grid_average(xy, slice(0, N_E),              rates_E)
    mapPV = grid_average(xy, slice(N_E, N_E + N_PV),     rates_PV)
    return {'E': mapE, 'PV': mapPV}

def sanity_check_files(widths=widths_mm, contrast=CONTRAST_FIXED):
    """Readable early fail if any (ZT, width) file is missing."""
    missing = []
    for w in widths:
        for circ in CIRCADIANS:
            try:
                resolve_paths_by_width(RESULTS_DIR, circ, w, contrast=contrast)
            except FileNotFoundError as e:
                missing.append(str(e))
    if missing:
        print("\n[Missing results for some (ZT, width):]")
        for m in missing: print(" -", m)
        raise FileNotFoundError("One or more required spike/position files were not found.")

#%% ------------- Multi-width heatmaps: main plotting -------------
def plot_heatmaps_grid_all_widths(pop='E', widths=widths_mm, save=False, dpi=300):
    """
    Build a grid of size len(widths) x 3 for a single population:
      columns: ZT0 | ZT12 | (ZT12 − ZT0)
      rows:    increasing width in mm (e.g., 0.10 … 0.30)
    """
    assert pop in ('E', 'PV')
    sanity_check_files(widths)

    # 1) Collect maps for all (ZT, width)
    maps = {}             # (width, circadian) -> 20x20 map
    diffs = {}            # width -> 20x20 (ZT12 − ZT0)
    for w in widths:
        m0  = compute_maps_for_width('ZT0',  w)[pop]
        m12 = compute_maps_for_width('ZT12', w)[pop]
        maps[(w, 'ZT0')]  = m0
        maps[(w, 'ZT12')] = m12
        diffs[w]          = m12 - m0

    # 2) Global color scales (fair comparison across widths)
    abs_vmin = np.nanmin([maps[(w,'ZT0')]  for w in widths] + [maps[(w,'ZT12')] for w in widths])
    abs_vmax = np.nanmax([maps[(w,'ZT0')]  for w in widths] + [maps[(w,'ZT12')] for w in widths])
    diff_max = np.nanmax([np.nanmax(np.abs(diffs[w])) for w in widths])

    # 3) Plot grid
    nrows, ncols = len(widths), 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.4*nrows), dpi=150, constrained_layout=True)
    axes = np.atleast_2d(axes)

    cmap_abs = 'magma' if pop == 'E' else 'viridis'
    for i, w in enumerate(widths):
        # ZT0
        im0 = axes[i,0].imshow(maps[(w,'ZT0')], origin='lower', extent=[0,1000,0,1000],
                               vmin=abs_vmin, vmax=abs_vmax, cmap=cmap_abs, aspect='equal')
        # ZT12
        im1 = axes[i,1].imshow(maps[(w,'ZT12')], origin='lower', extent=[0,1000,0,1000],
                               vmin=abs_vmin, vmax=abs_vmax, cmap=cmap_abs, aspect='equal')
        # Difference
        im2 = axes[i,2].imshow(diffs[w],        origin='lower', extent=[0,1000,0,1000],
                               vmin=-diff_max, vmax=diff_max,  cmap='coolwarm', aspect='equal')

        # Row label: width in µm
        axes[i,0].set_ylabel(f'w = {int(w*1000)} µm')

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

#%% ------------- Run: produce two N×3 figures (E and PV) -------------
_ = plot_heatmaps_grid_all_widths(pop='E',  widths=widths_mm)
_ = plot_heatmaps_grid_all_widths(pop='PV', widths=widths_mm)


#%% X-profile: average across y, binned in x
'''
extra stuff
'''

def x_profile(rate_map):
    # mean across y (rows) → 1D profile over x
    return np.nanmean(rate_map, axis=0)

for w in widths_mm:
    m0E  = compute_maps_for_width('ZT0',  w)['E']
    m12E = compute_maps_for_width('ZT12', w)['E']
    m0PV = compute_maps_for_width('ZT0',  w)['PV']
    m12PV= compute_maps_for_width('ZT12', w)['PV']

    plt.figure(figsize=(5.2,3.6), dpi=150)
    plt.plot(np.linspace(0,1000,m0E.shape[1]), x_profile(m0E),  label=f'E ZT0 w={int(w*1000)}µm', color=ZT0color)
    plt.plot(np.linspace(0,1000,m12E.shape[1]), x_profile(m12E), label=f'E ZT12 w={int(w*1000)}µm', color=ZT12color)
    plt.plot(np.linspace(0,1000,m0PV.shape[1]), x_profile(m0PV), label=f'PV ZT0', linestyle='--', color=ZT0color)
    plt.plot(np.linspace(0,1000,m12PV.shape[1]), x_profile(m12PV),label=f'PV ZT12', linestyle='--', color=ZT12color)
    plt.xlabel('x (µm)'); plt.ylabel('Mean rate across y (Hz)')
    plt.title(f'X-profile at width {int(w*1000)} µm')
    plt.grid(True, alpha=0.25); plt.legend(frameon=False); plt.tight_layout()

#%% Fraction of active neurons vs width (simple threshold)
THRESH_HZ = 5.0
def active_fraction(sp, pos, idx_slice, starts):
    rates = rates_over_windows(sp, idx_slice, starts)
    return np.mean(rates > THRESH_HZ)

fracs = {'ZT0_E':[], 'ZT12_E':[], 'ZT0_PV':[], 'ZT12_PV':[]}
for w in widths_mm:
    sp0, xy0  = load_data_width('ZT0',  w); st0  = control_window_starts(sp0)
    sp12,xy12 = load_data_width('ZT12', w); st12 = control_window_starts(sp12)
    fracs['ZT0_E'].append( active_fraction(sp0,  xy0,  slice(0,N_E),             st0))
    fracs['ZT12_E'].append(active_fraction(sp12, xy12, slice(0,N_E),             st12))
    fracs['ZT0_PV'].append(active_fraction(sp0,  xy0,  slice(N_E,N_E+N_PV),     st0))
    fracs['ZT12_PV'].append(active_fraction(sp12, xy12, slice(N_E,N_E+N_PV),    st12))

plt.figure(figsize=(5.2,3.6), dpi=150)
plt.plot(widths_mm, fracs['ZT0_E'],  '-o', label='E ZT0',  color=ZT0color)
plt.plot(widths_mm, fracs['ZT12_E'], '-o', label='E ZT12', color=ZT12color)
plt.plot(widths_mm, fracs['ZT0_PV'],  '--o', label='PV ZT0',  color=ZT0color)
plt.plot(widths_mm, fracs['ZT12_PV'], '--o', label='PV ZT12', color=ZT12color)
plt.xlabel('Rectangle width w (mm)'); plt.ylabel(f'Active fraction (> {THRESH_HZ} Hz)')
plt.grid(True, alpha=0.25); plt.legend(frameon=False); plt.tight_layout()

#%% Rate vs distance to nearest bar edge (wrap-aware)
def dist_to_edges_x(x, center=0.5, width=0.2, L=1.0):
    # periodic bar edges at x = center ± width/2
    edges = np.array([center - width/2, center + width/2])
    # wrap all to [0,1)
    x = x % L; edges = edges % L
    # periodic distance function
    def pd(a,b): 
        d = np.abs((a - b + 0.5) % 1.0 - 0.5)
        return d
    d0 = pd(x, edges[0]); d1 = pd(x, edges[1])
    return np.minimum(d0, d1)  # nearest edge distance (mm)

def binned_rate_vs_edge(sp, xy, idx_slice, starts, center=0.5, width=0.2, nbins=20):
    rates = rates_over_windows(sp, idx_slice, starts)
    x = xy[idx_slice, 0]
    d = dist_to_edges_x(x, center=center, width=width, L=1.0)
    bins = np.linspace(0, 0.5, nbins+1)  # up to half the sheet
    digit = np.digitize(d, bins) - 1
    prof = np.array([np.nanmean(rates[digit==k]) if np.any(digit==k) else np.nan for k in range(nbins)])
    centers = 0.5*(bins[:-1]+bins[1:])
    return centers*1000.0, prof  # µm

W_SHOW = 0.10
sp0, xy0  = load_data_width('ZT0',  W_SHOW);  st0  = control_window_starts(sp0)
sp12, xy12= load_data_width('ZT12', W_SHOW);  st12 = control_window_starts(sp12)

x_um, e0  = binned_rate_vs_edge(sp0,  xy0,  slice(0,N_E),             st0,  width=W_SHOW)
_,    e12 = binned_rate_vs_edge(sp12, xy12, slice(0,N_E),             st12, width=W_SHOW)
_,    p0  = binned_rate_vs_edge(sp0,  xy0,  slice(N_E,N_E+N_PV),      st0,  width=W_SHOW)
_,    p12 = binned_rate_vs_edge(sp12, xy12, slice(N_E,N_E+N_PV),      st12, width=W_SHOW)

plt.figure(figsize=(5.2,3.6), dpi=150)
plt.plot(x_um, e0,  label='E ZT0',  color=ZT0color)
plt.plot(x_um, e12, label='E ZT12', color=ZT12color)
plt.plot(x_um, p0,  '--', label='PV ZT0',  color=ZT0color)
plt.plot(x_um, p12, '--', label='PV ZT12', color=ZT12color)
plt.xlabel('Distance to nearest bar edge (µm)'); plt.ylabel('Mean rate (Hz)')
plt.title(f'Edge-anchored profiles @ w={int(W_SHOW*1000)} µm')
plt.grid(True, alpha=0.25); plt.legend(frameon=False); plt.tight_layout()

# %%
#%% ------------- NEW: Multi-width heatmaps for ALL neurons (E+PV+SST) -------------
def compute_map_all_neurons(circadian, width_mm):
    """Return a 20x20 Hz map averaging rates across ALL neurons (E+PV+SST)."""
    sp, xy  = load_data_width(circadian, width_mm)
    starts  = control_window_starts(sp)
    TOT_N   = N_E + N_PV + N_SST
    rates_all = rates_over_windows(sp, slice(0, TOT_N), starts)
    return grid_average(xy, slice(0, TOT_N), rates_all)

def plot_heatmaps_grid_all_neurons(widths=widths_mm, save=True, dpi=300):
    """
    Build a grid of size len(widths) x 3 for ALL neurons pooled:
      columns: ZT0 | ZT12 | (ZT12 − ZT0)
      rows:    increasing width in mm (e.g., 0.10 … 0.30)
    """
    sanity_check_files(widths)

    # 1) Collect maps for all (ZT, width)
    maps_all = {}     # (width, circadian) -> 20x20 map (Hz)
    diffs    = {}     # width -> 20x20 (ZT12 − ZT0)
    for w in widths:
        m0  = compute_map_all_neurons('ZT0',  w)
        m12 = compute_map_all_neurons('ZT12', w)
        maps_all[(w, 'ZT0')]  = m0
        maps_all[(w, 'ZT12')] = m12
        diffs[w]              = m12 - m0

    # 2) Global color scales across all rows (fair comparison)
    abs_vmin = np.nanmin([maps_all[(w, 'ZT0')]  for w in widths] +
                         [maps_all[(w, 'ZT12')] for w in widths])
    abs_vmax = np.nanmax([maps_all[(w, 'ZT0')]  for w in widths] +
                         [maps_all[(w, 'ZT12')] for w in widths])
    diff_max = np.nanmax([np.nanmax(np.abs(diffs[w])) for w in widths])

    # 3) Plot grid
    nrows, ncols = len(widths), 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.4*nrows), dpi=150, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for i, w in enumerate(widths):
        im0 = axes[i,0].imshow(maps_all[(w,'ZT0')],  origin='lower', extent=[0,1000,0,1000],
                               vmin=abs_vmin, vmax=abs_vmax, cmap='magma', aspect='equal')
        im1 = axes[i,1].imshow(maps_all[(w,'ZT12')], origin='lower', extent=[0,1000,0,1000],
                               vmin=abs_vmin, vmax=abs_vmax, cmap='magma', aspect='equal')
        im2 = axes[i,2].imshow(diffs[w],             origin='lower', extent=[0,1000,0,1000],
                               vmin=-diff_max, vmax=diff_max, cmap='coolwarm', aspect='equal')

        axes[i,0].set_ylabel(f'w = {int(w*1000)} µm')
        for j in range(ncols):
            a = axes[i,j]
            a.set_xticks([0,250,500,750,1000])
            a.set_yticks([0,250,500,750,1000])
            a.set_xlabel('x (µm)')
            a.set_ylabel('y (µm)')

    axes[0,0].set_title('ZT0 (ALL)')
    axes[0,1].set_title('ZT12 (ALL)')
    axes[0,2].set_title('ZT12 − ZT0 (ALL)')

    # Colorbars (abs for cols 0–1; diff for col 2)
    fig.colorbar(im1, ax=axes[:, :2].ravel().tolist(), shrink=0.92, label='Firing rate (Hz)')
    fig.colorbar(im2, ax=axes[:,  2].ravel().tolist(), shrink=0.92, label='Δ Hz (ZT12−ZT0)')

    if save:
        out = os.path.join(PROJECT_ROOT, f'heatmaps_grid_ALL_{len(widths)}x3.png')
        fig.savefig(out, dpi=dpi)
        print(f"[saved] {out}")

    return fig

#%% ------------- Run the ALL-neuron grid -------------
_ = plot_heatmaps_grid_all_neurons(widths=widths_mm, save=True, dpi=300)

# %%
