# Synthetic degradations of the NE-US data (PLAN_UPDATE_v2 §2.5): sparse network, station outages, coarse grid.

import numpy as np
import pandas as pd


def spread_stations(lons, lats, n, seed):
    """Indices (sorted, i.e. shapefile row order) of n stations picked by farthest-point sampling from a seeded start."""
    lons, lats = np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
    if n >= len(lons):
        return np.arange(len(lons))
    x = lons * np.cos(np.deg2rad(lats.mean()))  # degrees of longitude are shorter away from the equator

    def dist(i):
        return np.hypot(x - x[i], lats - lats[i])

    idx = [int(np.random.default_rng(seed).integers(len(lons)))]
    d = dist(idx[0])
    for _ in range(n - 1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, dist(idx[-1]))
    return np.sort(idx)


def eval_stations(lons, lats, n_eval):
    """Fixed evaluation set of the E4 density sweep: n_eval spread stations, independent of the subsample seed."""
    return spread_stations(lons, lats, n_eval, seed=0)


def keep_fraction(lons, lats, fraction, n_eval, seed):
    """Sorted indices of round(fraction * n) stations: the fixed evaluation set plus a seeded random draw of the rest."""
    n = len(lons)
    keep_eval = eval_stations(lons, lats, n_eval)
    rest = np.setdiff1d(np.arange(n), keep_eval)
    n_rest = max(int(round(fraction * n)) - len(keep_eval), 0)
    drawn = np.random.default_rng(seed).choice(rest, n_rest, replace=False)
    return np.sort(np.concatenate([keep_eval, drawn]))


def inject_gaps(is_real, target_real_fraction, mean_gap_hours, seed):
    """Copy of is_real (stations, time) with contiguous outage blocks removed until the real fraction hits the target."""
    out = np.array(is_real, dtype=bool, copy=True)
    rng = np.random.default_rng(seed)
    n_st, n_t = out.shape
    need = int(out.sum()) - int(round(target_real_fraction * out.size))
    while need > 0:
        s, t = rng.integers(n_st), rng.integers(n_t)
        seg = out[s, t:t + rng.geometric(1 / mean_gap_hours)]  # view: writes go to out
        real = np.flatnonzero(seg)[:need]  # last block is clipped so the target is met exactly
        seg[real] = False
        need -= len(real)
    return out


def refill(values, is_real):
    """Fill non-real hours like Madis.createRawFile: ffill then bfill along time, per station."""
    filled = pd.DataFrame(np.where(is_real, values, np.nan).T).ffill().bfill().values.T
    return np.nan_to_num(filled, nan=np.nanmean(filled))  # stations with no real hour left


def coarsen_era5(ds, factor):
    """Block mean over factor x factor ERA5 cells; partial edge blocks are dropped."""
    if factor == 1:
        return ds
    return ds.coarsen(latitude=factor, longitude=factor, boundary='trim').mean()
