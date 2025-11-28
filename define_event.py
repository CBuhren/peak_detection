# --- Algorithm to detect extreme atmospheric moistening and drying in IWV timeseries from MWRs ---
# 
# Author: Christian Buhren (christian.buhren@uni-koeln.de)

import numpy as np
import xarray as xr
import glob
import pandas as pd
import warnings
import calendar
warnings.filterwarnings("ignore", message="Index.ravel returning ndarray is deprecated", category=FutureWarning)

Q_THRESH   = 0.95 # percentile threshold to define "extreme"
MAX_LINK_H = 48 # maximum distance (h) between Peaks within multiple peak events

def load_and_concatenate_data(filepath: str) -> xr.Dataset:
    """
    Import the MWR data from given `filepath` and load as one xr.Dataset

    Parameters:
        - filepath: filepath to MWR data (`str`)

    Returns:
        - ds_all: 10-min resolved MWR data of IWV of LWP (`xr.Dataset`)
    """

    files_all_years = sorted(glob.glob(f"{filepath}/20[1-2][0-9]/*.nc"))
    ds_all = xr.open_mfdataset(files_all_years, combine='nested', concat_dim='time')
    ds_all.load()
    print('Data loaded to memory.')
    return ds_all

def sliding_window_extrema(iwv_smoothed: xr.DataArray, window_size=72, valid_ratio=0.5):
    """
    Function which looks for local extrema within a 12-hour window.

    Parameters:
    -----------
        - iwv_smoothed: smoothed IWV timeseries data (smoothed with a 6-hour sliding window)
        - window_size: window size in which local minima and maxima are identified 
        to check for local extrema
        - valid_ratio: The minimum percentage of valid values that a time window must have in 
        order to be used for the extreme point search. Here: `50%`

    Returns:
    --------
        - extrema_max_idx: Indices of identified maxima of the smoothed profile
        - extrema_min_idx: Indices of identified minima of the smoothed profile
    """

    half_window = window_size // 2
    values = iwv_smoothed.values

    extrema_max_idx = []
    extrema_min_idx = []

    for i in range(half_window, len(iwv_smoothed) - half_window):
        window = values[i - half_window: i + half_window + 1]
        
        valid = ~np.isnan(window)
        if np.sum(valid) < valid_ratio * len(window):
            continue  # too many NaN within window
        
        center_val = values[i]
        if np.isnan(center_val):
            continue  # skip index if center is NaN
        
        # checks for maxima and minima
        if center_val == np.nanmax(window):
            extrema_max_idx.append(i)
        
        elif center_val == np.nanmin(window):
            extrema_min_idx.append(i)

    return extrema_max_idx, extrema_min_idx

def find_real_extremum(iwv: xr.DataArray, approx_idx, kind, window: int, direction='both'):
    """
    Function to find extrema in the 10-min resolved IWV data based on the 
    identified extrema from the smoothed IWV timeseries.

    Parameters:
        - iwv: `xr.DataArray` of the 10-min resolved IWV data
        - approx_idx: indices of the rough position of the local extrema from the smoothed profile
        - kind: either `min` or `max` search
        - window: window size to look around each smoothed extrema index. Here: `6 hours around each index`
        - direction: how the window should be placed with respect to the extrema search. Here: `both` setted, i.e., 
        looks in both directions

    Returns:
        - start + idx_in_seg: The true index of the actual extrema
    """

    if direction == 'before': # looks only for candidates before a peak (not in use yet)
        start = max(0, approx_idx - window)
        end = approx_idx + 1
    elif direction == 'after': # looks only for candidates after a peak (not in use yet)
        start = approx_idx
        end = min(len(iwv), approx_idx + window + 1)
    else:
        start = max(0, approx_idx - window) # (only in use yet)
        end = min(len(iwv), approx_idx + window + 1)
    
    segment = iwv[start:end]

    if np.all(np.isnan(segment)): # if window contains only nans, return  None
        return None
    
    if kind == 'min':
        idx_in_seg = np.nanargmin(segment)
    else:
        idx_in_seg = np.nanargmax(segment)
    return start + idx_in_seg

def build_segment_dict(mean_iwv, mean_iwv_df, q_thresh: float, idx_min1: int, idx_peak: int, idx_min2: int):
    """
    Function to collect information of MP-M or MP-D events (min1, min2, peak, duration, amplitudes)

    Parameters:
    ----------
        - mean_iwv: 10-min resolved IWV data (10-min means)
        - mean_iwv_df: same as mean_IWV but as pandas dataframe
        - q_thresh: percentile value (0.95 - or 95th percentile) to define extreme atmospheric moistening and drying
        - idx_min1: index value of min1 points
        - idx_peak: index value of peak points
        - idx_min2: index value of min2 points
    """
    amp_inc = float((mean_iwv[idx_peak] - mean_iwv[idx_min1]).values)
    amp_dec = float((mean_iwv[idx_peak] - mean_iwv[idx_min2]).values)
    t_inc_h = float((mean_iwv.time[idx_peak].values - mean_iwv.time[idx_min1].values) / np.timedelta64(1, 'h'))
    t_dec_h = float((mean_iwv.time[idx_min2].values - mean_iwv.time[idx_peak].values) / np.timedelta64(1, 'h'))

    w_inc = max(1, int(round(t_inc_h)))
    w_dec = max(1, int(round(t_dec_h)))
    roll_inc = mean_iwv_df.rolling(f'{w_inc}H', center=True)
    roll_dec = mean_iwv_df.rolling(f'{w_dec}H', center=True)
    thr_inc = float((roll_inc.max() - roll_inc.min()).quantile(q_thresh))
    thr_dec = float((roll_dec.max() - roll_dec.min()).quantile(q_thresh))

    #built up segment dictionary of MP-M and MP-D chains with multiple detections (peaks)
    seg_dic = {
        "idx_min1": int(idx_min1),
        "idx_peak": int(idx_peak),
        "idx_min2": int(idx_min2),
        "time_min1": mean_iwv.time[idx_min1].values,
        "time_peak": mean_iwv.time[idx_peak].values,
        "time_min2": mean_iwv.time[idx_min2].values,
        "min1_val": float(mean_iwv[idx_min1].values),
        "peak_val": float(mean_iwv[idx_peak].values),
        "amp_inc": round(amp_inc, 3),
        "amp_dec": round(amp_dec, 3),
        "dur_inc_hours": t_inc_h,
        "dur_dec_hours": t_dec_h,
        "inc_extreme": amp_inc >= thr_inc,
        "dec_extreme": amp_dec >= thr_dec,
    }

    return seg_dic

def grow_chain_left(chain_segments, real_max_idx, real_min_idx, mean_iwv,
                    mean_iwv_df, q_thresh=Q_THRESH, max_gap_h=MAX_LINK_H, mode="up"):
    """
    Extends a forward-built chain to the left, i.e., backward in time

    Parameters:
    ----------
        - chain_segments: 

    mode="up"  (MP-M):   peak_prev < peak_curr,  min1_prev < min1_curr,  min2_prev == min1_curr,
                         UND (Zwischen-Segment-Bedingung) min2_prev ≥ min1_prev
    mode="down"(MP-D):   peak_prev > peak_curr,  min1_prev > min1_curr,  min2_prev == min1_curr,
                         UND (Zwischen-Segment-Bedingung) min2_prev ≤ min1_prev
    """
    if not chain_segments:
        return chain_segments

    segs = list(chain_segments)
    while True:
        first = segs[0]
        peak_A = first["idx_peak"]
        min1_A = first["idx_min1"]

        prev_peaks = [p for p in real_max_idx if p < peak_A]
        if not prev_peaks:
            break
        peak_P = prev_peaks[-1]

        dt_h = float((mean_iwv.time[peak_A].values - mean_iwv.time[peak_P].values) / np.timedelta64(1, 'h'))
        if dt_h > max_gap_h:
            break

        peak_P_val = float(mean_iwv[peak_P].values)
        peak_A_val = float(mean_iwv[peak_A].values)

        if mode == "up":
            if not (peak_P_val < peak_A_val):
                break
        else:
            if not (peak_P_val > peak_A_val):
                break

        s = slice(peak_P, peak_A + 1)
        rel = int(np.nanargmin(mean_iwv[s].values))
        min2_P = int(s.start + rel)
        if min2_P != min1_A:
            break

        prev_prev_peaks = [pp for pp in real_max_idx if pp < peak_P]
        if prev_prev_peaks:
            pp = prev_prev_peaks[-1]
            s2 = slice(pp, peak_P + 1)
            rel2 = int(np.nanargmin(mean_iwv[s2].values))
            min1_P = int(s2.start + rel2)
        else:
            mins_before = [m for m in real_min_idx if m < peak_P]
            if not mins_before:
                break
            min1_P = int(mins_before[np.argmin([float(mean_iwv[m].values) for m in mins_before])])

        min1_P_val = float(mean_iwv[min1_P].values)
        min1_A_val = float(mean_iwv[min1_A].values)
        min2_P_val = float(mean_iwv[min2_P].values)

        if mode == "up":
            if not (min1_P_val < min1_A_val):
                break
            if min2_P_val < min1_P_val:
                break
        else:
            if not (min1_P_val > min1_A_val):
                break
            if min2_P_val > min1_P_val:
                break

        # check for cases where min and peak are > 3 days apart
        t_inc_h = float((mean_iwv.time[peak_P].values - mean_iwv.time[min1_P].values) / np.timedelta64(1, 'h'))
        t_dec_h = float((mean_iwv.time[min2_P].values - mean_iwv.time[peak_P].values) / np.timedelta64(1, 'h'))
        if t_inc_h > 72 or t_dec_h > 72:
            break

        seg_left = build_segment_dict(mean_iwv, mean_iwv_df, q_thresh,
                                      idx_min1=min1_P, idx_peak=peak_P, idx_min2=min2_P)
        segs.insert(0, seg_left)

    return segs

def grow_chain_right(peak_idx, min1_idx, direction_mode,
                       real_max_idx, real_min_idx, mean_iwv, mean_iwv_df,
                       q_thresh=Q_THRESH, MAX_LINK_H=MAX_LINK_H):
    """
    Extends a forward-built chain to the left, i.e., backward in time

    Parameters:
    ----------
        - peak_idx: index value of peak point
        - min1_idx: index value of min1 point
        - direction_mode: either `up` or `down` referring to moistening and drying
        - real_max_idx: maximum index value of the 10-min resolved IWV timeseries
        - real_min_idx: minimum index value of the 10-min resolved IWV timeseries
        - mean_iwv: 10-min resolved IWV data (10-min means)
        - mean_iwv_df: same as mean_IWV but as pandas dataframe
        - q_thresh: percentile value (0.95 - or 95th percentile) to define extreme atmospheric moistening and drying
        - MAX_LINK_H: denotes the maximum time difference that is allowed for next peak (not more than 48 hours apart)
    """
    
    chain_segments = []
    current_peak_idx = peak_idx
    current_min1_idx = min1_idx
    next_event_found = True

    while next_event_found:
        candidate_mins = [m for m in real_min_idx if m > current_peak_idx]
        if not candidate_mins:
            break

        next_maxima = [p for p in real_max_idx if p > current_peak_idx]
        if next_maxima:
            next_max = min(next_maxima)
            search_range = [m for m in candidate_mins if m < next_max]
        else:
            search_range = candidate_mins
        if not search_range:
            break
        min2_idx = search_range[np.argmin([mean_iwv[m].values for m in search_range])]

        val_peak = float(mean_iwv[current_peak_idx].values)
        val_min1 = float(mean_iwv[current_min1_idx].values)
        val_min2 = float(mean_iwv[min2_idx].values)
        if val_min1 > val_peak or val_min2 > val_peak:
            break

        t_inc = (mean_iwv.time[current_peak_idx].values - mean_iwv.time[current_min1_idx].values) / np.timedelta64(1, "h")
        t_dec = (mean_iwv.time[min2_idx].values - mean_iwv.time[current_peak_idx].values) / np.timedelta64(1, "h")
        if t_inc > 72 or t_dec > 72:
            break

        # find next peak, either for MP-M or MP-D event
        if direction_mode == "up":
            cand_peaks = [p for p in real_max_idx if p > min2_idx and float(mean_iwv[p].values) > val_peak]
        else:
            cand_peaks = [p for p in real_max_idx if p > min2_idx and float(mean_iwv[p].values) < val_peak]

        have_next = False
        if cand_peaks:
            next_peak = cand_peaks[0]
            dph = (mean_iwv.time[next_peak].values - mean_iwv.time[current_peak_idx].values) / np.timedelta64(1, "h")
            have_next = dph <= MAX_LINK_H

        # only enforce the intermediate segment condition if it continues
        if have_next:
            if direction_mode == "up" and (val_min2 < val_min1):
                break
            if direction_mode == "down" and (val_min2 > val_min1):
                break
        else:
            # end segment must match chain direction (no counterphase at the end)
            amp_inc = val_peak - val_min1
            amp_dec = val_peak - val_min2
            if direction_mode == "down":
                if not ((val_min2 <= val_min1) and (amp_dec >= amp_inc)):
                    break
            else: # keep last peak for MP-M
                pass

        seg = build_segment_dict(mean_iwv, mean_iwv_df, q_thresh,
                                 idx_min1=current_min1_idx, idx_peak=current_peak_idx, idx_min2=min2_idx)
        chain_segments.append(seg)

        if have_next:
            current_min1_idx = min2_idx
            current_peak_idx = next_peak
        else:
            next_event_found = False

    return chain_segments

def _nan_to_none(x):
    """
    This small function converts NaN to None (NaN != NaN)

    Parameters:
    -----------
        - x: any numpy array
    
    Returns:
    --------
        - converted any instances of NaN to None
    """
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else x

def pack_record(base: dict, month: int, event_type: str, extra: dict | None = None) -> dict:
    """
    Function which returns a dictionary-like object following a specific order

    Parameters:
    -----------
        - base: requires an input dictionary with entries which will be sorted here
        - month: month of respective event
        - event_type: refers to single or multiple detection events, i.e., SP-M, MP-M, SP-D, MP-D
        - extra: if extra True, additional information will be added to the dictionary (bool)
    
    Returns:
    --------
        - Dictionary with updated structure and with only the important information
    """

    rec = {
        "month": month,
        "time_peak": base.get("time_peak"),
        "time_min1": base.get("time_min1"),
        "time_min2": base.get("time_min2"),
        "amp_inc": base.get("amp_inc"),
        "amp_dec": base.get("amp_dec"),
        "dur_inc_hours": base.get("dur_inc_hours"),
        "dur_dec_hours": base.get("dur_dec_hours"),
        "event_type": event_type,
    }
    # if extra, than these additional informations will be added to the dictionary
    tail = {
        "inc_extreme": base.get("inc_extreme"),
        "dec_extreme": base.get("dec_extreme"),
    }
    if extra:
        tail.update(extra)
    rec.update(tail)
    return rec

def event_detection_algo(ds_mwr_concat_years, year, mo):
    """
    Runs the detection algorithm to find extreme moistening and drying cases.
    Specifically, all identified maxima and minima are taken to calculate
    IWV amplitudes (between min1 and peak -> moistening, and peak and min2 -> drying).
    Then each amplitude is checked if being extreme or not, i.e., 
    if the time- and amplitude-based threshold of IWV exceeds the 95th percentile 
    of typical amplitudes for a given time of the respective month. 

    For example, if an amplitude with a duration time t from min1 to peak (moistening)
    or from peak to min2 (drying) is detected in January, then the 95th percentile 
    of the IWV amplitude of all Januaries from 2012-2024 for time t is calculated.
    
    Parameters:
    -----------
        - ds_mwr_concat_years: `xr.Dataset` of the 10-min resolved MWR data
        - year: `int`, year to run
        - mo: `int`, month to run

    Returns:
    --------
        - results: `list[dict]` of detected events, including SP-M/SP-D and MP-M/MP-D
    """

    results = []

    month_str = f'{mo:02d}'
    num_days = calendar.monthrange(year, mo)[1]
    
    buffer = np.timedelta64(1, "D") * 2  # two days of buffer

    time_segment = dict(time=slice(
        np.datetime64(f"{year}-{month_str}-01") - buffer,
        np.datetime64(f"{year}-{month_str}-{num_days}") + buffer
    )) # monthly time segment with additional 2 days of buffer at the start and end of each month

    mean_iwv = ds_mwr_concat_years.sel(**time_segment).prw_10min[:, 0]

    if mean_iwv.size < 36:
        return []

    smoothed_iwv = mean_iwv.rolling(time=36, center=True, min_periods=18).mean() # smooth the IWV profile
    smoothed_max_idx, smoothed_min_idx = sliding_window_extrema(smoothed_iwv) # identify extrema of smoothed profile

    # refine extrema in original series
    real_min_idx = []
    for idx in smoothed_min_idx:
        real_idx = find_real_extremum(mean_iwv, idx, kind='min', window=36, direction='both')
        if real_idx is not None:
            real_min_idx.append(real_idx)
    real_min_idx = sorted(set(real_min_idx))

    real_max_idx = []
    for idx in smoothed_max_idx:
        real_idx = find_real_extremum(mean_iwv, idx, kind='max', window=36, direction='both')
        if real_idx is not None:
            real_max_idx.append(real_idx)
    real_max_idx = sorted(set(real_max_idx))
    if not real_min_idx or not real_max_idx:
        return []

    # --- Handles two closely located extrema (<12h) ---
    # Peaks: keep highest
    filtered_real_max_idx = []
    i = 0
    while i < len(real_max_idx):
        group = [real_max_idx[i]]
        t0 = mean_iwv.time[real_max_idx[i]].values
        j = i + 1
        while j < len(real_max_idx):
            tj = mean_iwv.time[real_max_idx[j]].values
            delta_h = (tj - t0) / np.timedelta64(1, "h")
            if delta_h < 12:
                group.append(real_max_idx[j]); j += 1
            else:
                break
        max_idx = max(group, key=lambda g: float(mean_iwv[g].values))
        filtered_real_max_idx.append(max_idx)
        i = j
    real_max_idx = sorted(set(filtered_real_max_idx))

    # Minima: keep deepest
    filtered_real_min_idx = []
    i = 0
    while i < len(real_min_idx):
        group = [real_min_idx[i]]
        t0 = mean_iwv.time[real_min_idx[i]].values
        j = i + 1
        while j < len(real_min_idx):
            tj = mean_iwv.time[real_min_idx[j]].values
            delta_h = (tj - t0) / np.timedelta64(1, "h")
            if delta_h < 12:
                group.append(real_min_idx[j]); j += 1
            else:
                break
        min_idx = min(group, key=lambda g: float(mean_iwv[g].values))
        filtered_real_min_idx.append(min_idx)
        i = j
    real_min_idx = sorted(set(filtered_real_min_idx))

    # --- Handles general cases with >1 peaks between two minima, and >1 minima between two peaks
    keep_peaks = set(real_max_idx)
    for i in range(len(real_min_idx) - 1):
        left_min = real_min_idx[i]
        right_min = real_min_idx[i + 1]
        inter_peaks = [p for p in real_max_idx if left_min < p < right_min]
        if len(inter_peaks) >= 2:
            vals = [mean_iwv[p].values for p in inter_peaks]
            keep_idx = inter_peaks[int(np.argmax(vals))]
            for p in inter_peaks:
                if p != keep_idx and p in keep_peaks:
                    keep_peaks.remove(p)

    keep_mins = set(real_min_idx)
    for i in range(len(real_max_idx) - 1):
        left_peak = real_max_idx[i]
        right_peak = real_max_idx[i + 1]
        inter_mins = [m for m in real_min_idx if left_peak < m < right_peak]
        if len(inter_mins) >= 2:
            vals = [mean_iwv[m].values for m in inter_mins]
            keep_idx = inter_mins[int(np.argmin(vals))]
            for m in inter_mins:
                if m != keep_idx and m in keep_mins:
                    keep_mins.remove(m)

    real_max_idx = sorted(keep_peaks)
    real_min_idx = sorted(keep_mins)

    # prepare for threshold evaluation (selects the month of current index i)
    iwv_sel_month = ds_mwr_concat_years.prw_10min[:, 0].sel(time=ds_mwr_concat_years.time.dt.month.isin([mo]))
    mean_iwv_df = iwv_sel_month.to_series()

    # --- chain logic ---
    emitted_segments = set()  # this set just makes sure to have non equal entries (min1,peak,min2,event_type)
    mp_committed_peaks = set()

    mp_m_windows = set()
    mp_d_windows = set()

    i = 0
    while i < len(real_max_idx):
        peak_idx = real_max_idx[i]

        if peak_idx in mp_committed_peaks:
            i += 1
            continue

        # look up for the closest min1 to current peak
        candidate_mins1 = [m for m in real_min_idx if m < peak_idx]
        if not candidate_mins1:
            i += 1
            continue

        prev_peaks = [p for p in real_max_idx if p < peak_idx]
        if prev_peaks:
            prev_peak = max(prev_peaks)
            search_range1 = [m for m in candidate_mins1 if m > prev_peak]
            if not search_range1:
                search_range1 = [max(candidate_mins1)]
        else:
            search_range1 = candidate_mins1

        # lowest minimum between prev_peak and peak_idx
        min1_idx = search_range1[np.argmin([mean_iwv[m].values for m in search_range1])]

        # collect chains of MP-M or MP-D cases
        chain_up = grow_chain_right(
            peak_idx=peak_idx, min1_idx=min1_idx, direction_mode="up",
            real_max_idx=real_max_idx, real_min_idx=real_min_idx,
            mean_iwv=mean_iwv, mean_iwv_df=mean_iwv_df,
            q_thresh=Q_THRESH, MAX_LINK_H=MAX_LINK_H
        )
        chain_up = grow_chain_left(chain_up, real_max_idx, real_min_idx,
                                   mean_iwv, mean_iwv_df,
                                   q_thresh=Q_THRESH, max_gap_h=MAX_LINK_H, mode="up")

        chain_down = grow_chain_right(
            peak_idx=peak_idx, min1_idx=min1_idx, direction_mode="down",
            real_max_idx=real_max_idx, real_min_idx=real_min_idx,
            mean_iwv=mean_iwv, mean_iwv_df=mean_iwv_df,
            q_thresh=Q_THRESH, MAX_LINK_H=MAX_LINK_H
        )
        chain_down = grow_chain_left(chain_down, real_max_idx, real_min_idx,
                                     mean_iwv, mean_iwv_df,
                                     q_thresh=Q_THRESH, max_gap_h=MAX_LINK_H, mode="down")

        # Falls beide leer → nächster Startpeak
        if not chain_up and not chain_down:
            i += 1
            continue

        # 2) Klassifizieren
        inc_any_up   = any(seg["inc_extreme"] for seg in chain_up)
        dec_any_down = any(seg["dec_extreme"] for seg in chain_down)
        is_mp_m = (len(chain_up)   >= 2) and inc_any_up
        is_mp_d = (len(chain_down) >= 2) and dec_any_down

        if is_mp_m or is_mp_d:
            # Längere Kette gewinnt (bei Gleichstand MP-M)
            if is_mp_m and (not is_mp_d or len(chain_up) >= len(chain_down)):
                chosen = chain_up;  ctype = "MP-M"; last_idx = chain_up[-1]["idx_peak"]
            else:
                chosen = chain_down; ctype = "MP-D"; last_idx = chain_down[-1]["idx_peak"]

            for k, seg in enumerate(chosen):
                rec = seg.copy()
                rec["month"] = mo
                rec["event_type"] = ctype

                # Halbsegment-Trim an der Wende
                if ctype == "MP-M" and k == len(chosen) - 1:
                    mp_m_windows.add( (seg["idx_min1"], seg["idx_peak"]) )
                    rec["amp_dec"] = np.nan
                    rec["dur_dec_hours"] = np.nan
                    rec["time_min2"] = np.nan
                    rec["dec_extreme"] = np.nan
                elif ctype == "MP-D" and k == 0:
                    mp_d_windows.add( (seg["idx_peak"], seg["idx_min2"]) )
                    rec["amp_inc"] = np.nan
                    rec["dur_inc_hours"] = np.nan
                    rec["time_min1"] = np.nan
                    rec["inc_extreme"] = np.nan

                key = (_nan_to_none(rec.get("idx_min1")), rec["idx_peak"],
                       _nan_to_none(rec.get("idx_min2")), rec["event_type"])
                if key in emitted_segments:
                    continue
                emitted_segments.add(key)

                mp_committed_peaks.add(seg["idx_peak"])
                results.append(
                    pack_record(rec, month=mo, event_type=ctype)
                )

                # in this block it is checked if we have extreme cases right after a MP event
                if ctype == "MP-M":
                    last_seg = chosen[-1]
                    if last_seg.get("dec_extreme", False) and pd.notna(last_seg.get("idx_min2")):
                        win = (last_seg["idx_peak"], last_seg["idx_min2"])
                        if win not in mp_d_windows:
                            sp = last_seg.copy()
                            sp["month"] = mo
                            sp["event_type"] = "SP-D"
                            sp["phase"] = "drying"
                            sp["delta_peak_hours"] = np.nan
                            sp["amp_inc"] = np.nan
                            sp["dur_inc_hours"] = np.nan
                            sp["time_min1"] = np.nan
                            sp["idx_min1"] = np.nan
                            sp["min1_val"] = np.nan
                            sp["inc_extreme"] = np.nan

                            key = (_nan_to_none(sp.get("idx_min1")), sp["idx_peak"],
                                _nan_to_none(sp.get("idx_min2")), sp["event_type"])
                            
                            if key not in emitted_segments:
                                emitted_segments.add(key)
                                results.append(sp)

                            mp_d_windows.add(win)

                elif ctype == "MP-D":
                    first_seg = chosen[0]
                    if first_seg.get("inc_extreme", False) and pd.notna(first_seg.get("idx_min1")):
                        win = (first_seg["idx_min1"], first_seg["idx_peak"])
                        if win not in mp_m_windows:
                            sp = first_seg.copy()
                            sp["month"] = mo
                            sp["event_type"] = "SP-M"
                            sp["phase"] = "moistening"
                            sp["delta_peak_hours"] = np.nan
                            sp["amp_dec"] = np.nan
                            sp["dur_dec_hours"] = np.nan
                            sp["time_min2"] = np.nan
                            sp["idx_min2"] = np.nan
                            sp["min2_val"] = np.nan
                            sp["dec_extreme"] = np.nan

                            key = (_nan_to_none(sp.get("idx_min1")), sp["idx_peak"],
                                _nan_to_none(sp.get("idx_min2")), sp["event_type"])
                            
                            if key not in emitted_segments:
                                emitted_segments.add(key)
                                results.append(sp)
                                
                            mp_m_windows.add(win)

            # jumps to the last peak of corresponding chain
            i = real_max_idx.index(last_idx) + 1
            continue

        # this part deals with the single peak events
        base_chain = chain_up if chain_up else chain_down
        for seg in base_chain:
            if seg["idx_peak"] in mp_committed_peaks: # if peak in MP chain already, skip
                continue
            if seg.get("inc_extreme", False):  # SP-M
                if (seg["idx_min1"], seg["idx_peak"]) in mp_m_windows:
                    continue
                out = seg.copy()
                out["month"] = mo
                out["event_type"] = "SP-M"
                out["amp_dec"] = np.nan
                out["dur_dec_hours"] = np.nan
                out["time_min2"] = np.nan
                out["dec_extreme"] = np.nan

                key = (_nan_to_none(out.get("idx_min1")), out["idx_peak"],
                       _nan_to_none(out.get("idx_min2")), out["event_type"])
                if key in emitted_segments:
                    continue
                emitted_segments.add(key)

                results.append(
                    pack_record(out, month=mo, event_type="SP-M")
                )

            if seg.get("dec_extreme", False):  # SP-D
                if (seg["idx_peak"], seg["idx_min2"]) in mp_d_windows:
                    continue
                out = seg.copy()
                out["month"] = mo
                out["event_type"] = "SP-D"
                out["amp_inc"] = np.nan
                out["dur_inc_hours"] = np.nan
                out["time_min1"] = np.nan
                out["inc_extreme"] = np.nan

                key = (_nan_to_none(out.get("idx_min1")), out["idx_peak"],
                       _nan_to_none(out.get("idx_min2")), out["event_type"])
                if key in emitted_segments:
                    continue
                emitted_segments.add(key)

                results.append(
                    pack_record(out, month=mo, event_type="SP-D")
                )

        # next peak
        i += 1

    return results

def save_results_to_csv(results, filepath='./', filename="iwv_detections.csv"):
    """
    The results from the detection algorithm are declared to a pandas Dataframe
    and saved as csv file to a specific location in `{filepath}/{filename}`

    Parameters:
        - results: `dict`, detected moistening and drying cases
        - filepath: `str`, filepath to store the csv file
        - filename: `str`, filename of detections file
    """

    df = pd.DataFrame(results)

    if df.empty:
        outpath = filepath + filename
        df.to_csv(outpath, index=False)
        print(f'File saved (empty) to {outpath}')
        return

    # --- Zeitspalten parsen
    for col in ["time_peak", "time_min1", "time_min2"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # =========================
    # KEIN peak-basiertes SP-Droppen mehr!
    # =========================

    # --- Segment-basierte Entdoppelung ---
    # MP-M Fenster: (time_min1, time_peak)
    mpm = df.loc[df["event_type"] == "MP-M", ["time_min1","time_peak"]].dropna()
    mpm["_key_mpm"] = (
        mpm["time_min1"].astype("datetime64[ns]").astype(str)
        + "||"
        + mpm["time_peak"].astype("datetime64[ns]").astype(str)
    )

    spm_mask = df["event_type"].eq("SP-M")
    tmp_spm = df.loc[spm_mask, ["time_min1","time_peak"]].copy()
    tmp_spm["_key_mpm"] = (
        tmp_spm["time_min1"].astype("datetime64[ns]").astype(str)
        + "||"
        + tmp_spm["time_peak"].astype("datetime64[ns]").astype(str)
    )
    dup_spm = tmp_spm["_key_mpm"].isin(set(mpm["_key_mpm"]))
    df = df.loc[~(spm_mask & dup_spm)].copy()

    # MP-D Fenster: (time_peak, time_min2)
    mpd = df.loc[df["event_type"] == "MP-D", ["time_peak","time_min2"]].dropna()
    mpd["_key_mpd"] = (
        mpd["time_peak"].astype("datetime64[ns]").astype(str)
        + "||"
        + mpd["time_min2"].astype("datetime64[ns]").astype(str)
    )

    spd_mask = df["event_type"].eq("SP-D")
    tmp_spd = df.loc[spd_mask, ["time_peak","time_min2"]].copy()
    tmp_spd["_key_mpd"] = (
        tmp_spd["time_peak"].astype("datetime64[ns]").astype(str)
        + "||"
        + tmp_spd["time_min2"].astype("datetime64[ns]").astype(str)
    )
    dup_spd = tmp_spd["_key_mpd"].isin(set(mpd["_key_mpd"]))
    df = df.loc[~(spd_mask & dup_spd)].copy()

    # --- Absicherung: zeilengleiche Dubs (gleiche Zeiten) weg
    keep_cols = [c for c in ["event_type","time_min1","time_peak","time_min2"] if c in df.columns]
    if keep_cols:
        df = df.drop_duplicates(subset=keep_cols, keep="first")

    # --- Sortierung: MP vor SP bei gleichem Peak-Zeitpunkt
    et_rank_map = {"MP-M": 0, "MP-D": 0, "SP-M": 1, "SP-D": 1}
    df["__etype_rank__"] = df.get("event_type", pd.Series(index=df.index)).map(et_rank_map).fillna(2).astype(int)
    df = (df.sort_values(["time_peak","__etype_rank__"], kind="mergesort")
            .drop(columns="__etype_rank__", errors="ignore")
            .reset_index(drop=True))

    # --- Schreiben
    df.to_csv(filepath + filename, index=False)
    print(f'File saved to {filepath}/{filename}')

def run_all_years(data_path, start_year=2012, end_year=2024, output_path='./', output_filename="iwv_detections.csv"):
    """
    Main function to run detection algorithm and store results as csv.

    Parameters:
        - data_path: `str`, path to the directory containing yearly MWR data folders
        - start_year: `int`, first year to process (default: 2012)
        - end_year: `int`, last year to process (default: 2024)
        - output_path: `str`, directory to save results (default: './')
        - output_filename: `str`, name of output CSV file (default: 'iwv_detections.csv')
    """

    ds_mwr_concat_years = load_and_concatenate_data(data_path)

    all_events = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            monthly_events = event_detection_algo(ds_mwr_concat_years, year, month)
            all_events.extend(monthly_events)
        print(f'Done with year {year}')

    save_results_to_csv(all_events, filepath=output_path, filename=output_filename)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python define_event.py <data_path> [start_year] [end_year]")
        print("Example: python define_event.py /path/to/mwr/data 2012 2024")
        sys.exit(1)

    data_path = sys.argv[1]
    start_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2012
    end_year = int(sys.argv[3]) if len(sys.argv) > 3 else 2024

    run_all_years(data_path, start_year, end_year)
