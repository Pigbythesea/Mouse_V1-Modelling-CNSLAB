#%%
'''
# Opto Postprocessing & Visualization

This notebook scans `results_*` folders for files named like  
`PV_dist{...}_condZT{0/12}_{spikes,positions}.pickle`, loads them, and produces:

1. **Distance → Center FR plot:** distance from the laser to the visual center (0.5, 0.5) vs. the **mean firing rate in the central 50×50 µm square**, with ZT0 and ZT12 overlaid.
2. **Spatial heatmap:** the 1×1 mm sheet sliced into 50×50 µm tiles; each tile’s color is the mean firing rate of neurons inside it.

**Notes / assumptions**
- Positions are in **mm** on a 1×1 mm sheet (like your sim). 50 µm = **0.05 mm**.
- We compute firing rates as (#spikes) / (simulation duration). If the duration is not known, we estimate it from the greatest spike time seen (and add a 1 ms cushion). You can override this.
- By default, we plot **E (pyramidal) cells only**. You can switch to PV or both.
'''

#%%

from pathlib import Path
import re, pickle, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------ User-configurable parameters ------------
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "work_dir" / "base_a0.07_pf0.0"   # adjust if you renamed it

SEARCH_ROOTS = [ROOT]                  # we'll search under ROOT instead

# Choose which cell type to analyze: 'E' (default), 'PV', or 'both'
CELLTYPE = 'E'

# Tile size for center square and heatmaps (in µm); 50 gives a 20x20 grid on a 1x1 mm sheet
TILE_SIZE_UM = 50.0

# If you know the exact total simulation duration in ms, put it here.
# If None, we'll infer from the max spike time in each run.
SIM_DURATION_MS_OVERRIDE = None

# Figure output directory
FIG_DIR = Path('figs'); FIG_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------
SHEET_MM = 1.0
CENTER = np.array([0.5, 0.5])  # mm
TILE_MM = TILE_SIZE_UM / 1000.0

def parse_meta(spike_path: Path):
    """Parse distance and condition from filename like PV_dist0.323_condZT12_spikes.pickle"""
    m = re.search(r'PV_dist([0-9.]+)_cond(ZT\d+)_spikes\.pickle', spike_path.name)
    if not m:
        raise ValueError(f'Unexpected filename: {spike_path.name}')
    dist_mm = float(m.group(1))
    cond = m.group(2)  # 'ZT0' or 'ZT12'
    laser_xy = np.array([dist_mm, dist_mm])  # along the diagonal in your runs
    # Euclidean distance from visual center (0.5, 0.5) in µm
    center_dist_um = 1000.0 * float(np.linalg.norm(laser_xy - CENTER))
    return dist_mm, cond, laser_xy, center_dist_um

def load_pair(spike_path: Path):
    pos_path = spike_path.with_name(spike_path.name.replace('_spikes', '_positions'))
    with open(spike_path, 'rb') as f:
        spikes = pickle.load(f)  # list-of-arrays of spike times (ms)
    with open(pos_path, 'rb') as f:
        pos = pickle.load(f)     # (N, 2) positions in mm on 1x1 sheet
    N = len(spikes)
    if isinstance(pos, list):
        pos = np.array(pos)
    if pos.shape[0] != N:
        raise ValueError('Positions and spikes have inconsistent lengths')
    # infer duration if not provided
    if SIM_DURATION_MS_OVERRIDE is not None:
        dur_ms = float(SIM_DURATION_MS_OVERRIDE)
    else:
        max_t = 0.0
        for s in spikes:
            if len(s):
                tmax = float(s[-1]) if np.all(np.diff(s) >= 0) else float(np.max(s))
                if tmax > max_t:
                    max_t = tmax
        dur_ms = max_t + 1.0  # 1 ms cushion
        if dur_ms <= 0:
            warnings.warn('No spikes found; using 1000 ms as a placeholder duration')
            dur_ms = 1000.0
    # per-neuron firing rate (Hz)
    rates_hz = np.array([len(s) / (dur_ms / 1000.0) for s in spikes], dtype=float)
    is_E = np.arange(N) < 8000
    is_PV = ~is_E
    return pos, spikes, rates_hz, is_E, is_PV, dur_ms

def find_all_spike_files():
    files = []
    for root in SEARCH_ROOTS:
        files.extend(root.rglob('PV_dist*_condZT*_spikes.pickle'))
    return sorted(files)


all_spike_files = find_all_spike_files()
print(f'Found {len(all_spike_files)} runs')
for p in all_spike_files[:10]:
    print('  -', p)

#%%

def center_square_mask(pos, center=CENTER, tile_mm=TILE_MM):
    lo = center - tile_mm/2.0
    hi = center + tile_mm/2.0
    return (pos[:,0] >= lo[0]) & (pos[:,0] < hi[0]) & (pos[:,1] >= lo[1]) & (pos[:,1] < hi[1])

records = []
for spike_path in all_spike_files:
    dist_mm, cond, laser_xy, center_dist_um = parse_meta(spike_path)
    pos, spikes, rates_hz, is_E, is_PV, dur_ms = load_pair(spike_path)
    m_center = center_square_mask(pos)

    if CELLTYPE == 'E':
        m = m_center & is_E
    elif CELLTYPE == 'PV':
        m = m_center & is_PV
    else:
        m = m_center

    fr_center = float(np.mean(rates_hz[m])) if np.any(m) else np.nan
    records.append(dict(cond=cond, dist_mm=dist_mm, center_dist_um=center_dist_um,
                        fr_center_hz=fr_center, n_cells=int(np.sum(m)),
                        spike_file=str(spike_path)))

df_center = pd.DataFrame.from_records(records).sort_values(['cond','center_dist_um']).reset_index(drop=True)
display(df_center.head(10))

#%%

plt.figure(figsize=(6,4))
for cond, sub in df_center.groupby('cond'):
    sub = sub.sort_values('center_dist_um')
    plt.plot(sub['center_dist_um'], sub['fr_center_hz'], marker='o', label=cond)
plt.xlabel('Laser distance to visual center (µm)')
plt.ylabel(f'Mean FR in central {int(TILE_SIZE_UM)}×{int(TILE_SIZE_UM)} µm (Hz)')
plt.title(f'Distance → Center FR ({CELLTYPE})')
plt.legend()
out = FIG_DIR / f'distance_vs_centerFR_{CELLTYPE}.png'
plt.tight_layout()
plt.savefig(out, dpi=200)
print('Saved', out)
plt.show()

#%%

def tile_heatmap(pos, values, tile_mm=TILE_MM, celltype_mask=None, sheet_mm=SHEET_MM):
    if celltype_mask is not None:
        pos = pos[celltype_mask]
        values = values[celltype_mask]
    nx = int(round(sheet_mm / tile_mm))
    ny = int(round(sheet_mm / tile_mm))
    # bin indices [0 .. nx-1]
    ix = np.clip((pos[:,0] / tile_mm).astype(int), 0, nx-1)
    iy = np.clip((pos[:,1] / tile_mm).astype(int), 0, ny-1)
    tile_sum = np.zeros((ny, nx), dtype=float)
    tile_cnt = np.zeros((ny, nx), dtype=int)
    for x, y, v in zip(ix, iy, values):
        tile_sum[iy, ix]  # touch to ensure shape (lint-friendly)
        tile_sum[y, x] += v
        tile_cnt[y, x] += 1
    with np.errstate(invalid='ignore'):
        tile_mean = tile_sum / np.maximum(tile_cnt, 1)
        tile_mean[tile_cnt == 0] = np.nan
    return tile_mean  # shape [ny, nx]

def plot_heatmap(tile_mean, title, outfile):
    plt.figure(figsize=(5.4, 4.6))
    # origin='lower' so (0,0) is bottom-left
    plt.imshow(tile_mean, origin='lower', extent=(0, SHEET_MM*1000, 0, SHEET_MM*1000), aspect='equal')
    cb = plt.colorbar()
    cb.set_label('Mean FR (Hz)')
    plt.xlabel('x (µm)')
    plt.ylabel('y (µm)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print('Saved', outfile)
    plt.show()

#%%
# Pick a run to visualize:
# Example: the closest laser position for ZT0 (or change manually).
if len(df_center):
    pick = (df_center.sort_values('center_dist_um')
                    .groupby('cond')
                    .head(1)
                    .iloc[0])
    chosen_file = Path(pick['spike_file'])
else:
    chosen_file = all_spike_files[0] if all_spike_files else None

print('Chosen run for heatmap:', chosen_file)

if chosen_file is not None:
    dist_mm, cond, laser_xy, center_dist_um = parse_meta(chosen_file)
    pos, spikes, rates_hz, is_E, is_PV, dur_ms = load_pair(chosen_file)
    if CELLTYPE == 'E':
        mask = is_E
    elif CELLTYPE == 'PV':
        mask = is_PV
    else:
        mask = None  # both
    tile_mean = tile_heatmap(pos, rates_hz, celltype_mask=mask)
    out = FIG_DIR / f'heatmap_{cond}_dist{dist_mm:.3f}_{CELLTYPE}.png'
    plot_heatmap(tile_mean, title=f'FR heatmap ({cond}, dist={dist_mm:.3f} mm, {CELLTYPE})', outfile=out)


#%%

# Uncomment to export heatmaps for every run found
 for spike_path in all_spike_files:
     dist_mm, cond, laser_xy, center_dist_um = parse_meta(spike_path)
     pos, spikes, rates_hz, is_E, is_PV, dur_ms = load_pair(spike_path)
     if CELLTYPE == 'E':
         mask = is_E
     elif CELLTYPE == 'PV':
         mask = is_PV
     else:
         mask = None
     tile_mean = tile_heatmap(pos, rates_hz, celltype_mask=mask)
     out = FIG_DIR / f'heatmap_{cond}_dist{dist_mm:.3f}_{CELLTYPE}.png'
     plot_heatmap(tile_mean, title=f'FR heatmap ({cond}, dist={dist_mm:.3f} mm, {CELLTYPE})', outfile=out)



#%% Center effect-size (ZT12 − ZT0) vs distance
# Uses df_center built earlier (already filtered by CELLTYPE)

# Pivot to pair ZT0 and ZT12 at the same distance
pairs = (df_center
         .pivot_table(index='dist_mm', columns='cond', values='fr_center_hz', aggfunc='mean'))

# Keep only distances where both conditions exist
pairs = pairs.dropna(subset=['ZT0','ZT12'])

# Grab the center-distance (µm) for x-axis
dist_map = (df_center[['dist_mm','center_dist_um']]
            .drop_duplicates(subset='dist_mm')
            .set_index('dist_mm'))
eff = pairs.copy()
eff['center_dist_um'] = dist_map.loc[eff.index, 'center_dist_um'].values

# Compute effect sizes
eff['diff_hz'] = eff['ZT12'] - eff['ZT0']     # circadian difference
eff['ratio']   = eff['ZT12'] / eff['ZT0']     # optional: gain (commented in plot)

# --- Plot: difference only (cleanest) ---
plt.figure(figsize=(6,4))
plt.plot(eff['center_dist_um'], eff['diff_hz'], marker='o')
plt.axhline(0, color='k', linewidth=0.8, alpha=0.6)
plt.xlabel('Laser distance to visual center (µm)')
plt.ylabel('Center FR: ZT12 − ZT0 (Hz)')
plt.title(f'Circadian effect in center tile ({CELLTYPE})')
plt.tight_layout()
out = FIG_DIR / f'center_effectsize_diff_vs_distance_{CELLTYPE}.png'
plt.savefig(out, dpi=200); print('Saved', out)
plt.show()

# --- (Optional) ratio plot: uncomment to generate ---
# plt.figure(figsize=(6,4))
# plt.plot(eff['center_dist_um'], eff['ratio'], marker='o')
# plt.axhline(1.0, color='k', linewidth=0.8, alpha=0.6)
# plt.xlabel('Laser distance to visual center (µm)')
# plt.ylabel('Center FR ratio: ZT12 / ZT0')
# plt.title(f'Circadian gain in center tile ({CELLTYPE})')
# plt.tight_layout()
# out = FIG_DIR / f'center_effectsize_ratio_vs_distance_{CELLTYPE}.png'
# plt.savefig(out, dpi=200); print('Saved', out)
# plt.show()

#%% Difference heatmaps (ZT12 − ZT0) for each matched distance


def _index_runs_by_distance(files):
    """Return {dist_mm: {'ZT0': Path, 'ZT12': Path}} for distances that have both."""
    idx = {}
    for p in files:
        p = Path(p)
        dist_mm, cond, _, _ = parse_meta(p)
        d = idx.setdefault(dist_mm, {})
        d[cond] = p
    # keep only distances with both conditions
    return {d:conds for d,conds in idx.items() if {'ZT0','ZT12'}.issubset(conds.keys())}

def _plot_diff_heatmap(diff_tile, title, outfile):
    vmax = np.nanmax(np.abs(diff_tile))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-6
    plt.figure(figsize=(5.4, 4.6))
    im = plt.imshow(diff_tile, origin='lower',
                    extent=(0, SHEET_MM*1000, 0, SHEET_MM*1000),
                    aspect='equal', cmap='coolwarm', vmin=-vmax, vmax=vmax)
    cb = plt.colorbar(im); cb.set_label('ZT12 − ZT0 (Hz)')
    plt.xlabel('x (µm)'); plt.ylabel('y (µm)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200); print('Saved', outfile)
    plt.show()

# Build pairing index
pairs_idx = _index_runs_by_distance(all_spike_files)

# Generate a difference map for every paired distance
for dist_mm, conds in sorted(pairs_idx.items()):
    p0, p12 = conds['ZT0'], conds['ZT12']
    # Load both runs
    pos0, _, rates0, isE0, isPV0, _ = load_pair(p0)
    pos1, _, rates1, isE1, isPV1, _ = load_pair(p12)

    # Choose cell type
    if CELLTYPE == 'E':
        mask0, mask1 = isE0, isE1
    elif CELLTYPE == 'PV':
        mask0, mask1 = isPV0, isPV1
    else:
        mask0 = mask1 = None  # both

    # Bin to tiles (independent binning then subtract)
    tile0 = tile_heatmap(pos0, rates0, celltype_mask=mask0)
    tile1 = tile_heatmap(pos1, rates1, celltype_mask=mask1)
    diff_tile = tile1 - tile0  # ZT12 − ZT0

    # Plot and save
    title = f'Diff heatmap (ZT12 − ZT0), dist={dist_mm:.3f} mm, {CELLTYPE}'
    out = FIG_DIR / f'diff_heatmap_ZT12_minus_ZT0_dist{dist_mm:.3f}_{CELLTYPE}.png'
    _plot_diff_heatmap(diff_tile, title, out)

# %%
