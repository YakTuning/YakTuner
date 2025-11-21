# C:/Users/Sam/PycharmProjects/YAKtunerCONVERTED/LPFP.py

"""
Low-Pressure Fuel Pump (LPFP) PWM Tuning Module

This module contains pure, non-UI functions to analyze engine logs and
recommend adjustments to the LPFP PWM control tables.
"""

import numpy as np
import pandas as pd
from scipy import stats, interpolate

# --- Helper Functions ---

def _process_and_filter_lpfp_log_data(log, logvars):
    """
    A pure function to prepare and filter log data for LPFP tuning.
    It returns a processed DataFrame and a list of warnings.
    """
    warnings = []
    df = log.copy()

    # Check for required variables first
    required_vars = ['FF_SP', 'LPFP_FP_SP', 'LPFP_PWM', 'RPM', 'LPFP_FP']
    missing_vars = [var for var in required_vars if var not in logvars]
    if missing_vars:
        warnings.append(f"The required log variable(s) were not found: {', '.join(missing_vars)}")
        return pd.DataFrame(), warnings

    # --- Convert FF_SP from per-stroke to per-minute ---
    df['FF_SP'] = df['FF_SP'] * 2 * df['RPM'] / 1000

    if "OILTEMP" in logvars:
        df = df[df['OILTEMP'] > 180]

    # Filter for rows where the pump is actively being controlled and in closed loop
    df = df[df['LPFP_PWM'] > 0].copy()
    df.dropna(subset=required_vars, inplace=True)

    # --- Filtering Logic for data stability ---
    df['LPFP_DELTA'] = df['LPFP_FP'] - df['LPFP_FP_SP']
    df['LPFP_DELTA_CHANGE'] = df['LPFP_DELTA'].diff().abs()
    is_small_delta = df['LPFP_DELTA'].abs() < 10
    is_steady_delta = df['LPFP_DELTA_CHANGE'] < 1.0
    final_mask = is_small_delta | is_steady_delta
    df = df[final_mask.fillna(False)]

    if df.empty:
        warnings.append("No data points met the criteria (small or steady LPFP delta).")

    return df, warnings

def _create_bins(log, xaxis, yaxis):
    """Discretizes log data into bins based on LPFP map axes."""
    xedges = [0] + [(xaxis[i] + xaxis[i + 1]) / 2 for i in range(len(xaxis) - 1)] + [np.inf]
    yedges = [0] + [(yaxis[i] + yaxis[i + 1]) / 2 for i in range(len(yaxis) - 1)] + [np.inf]

    log['X'] = pd.cut(log['FF_SP'], bins=xedges, labels=False)
    log['Y'] = pd.cut(log['LPFP_FP_SP'], bins=yedges, labels=False)
    return log

def _fit_surface_lpfp(log_data, xaxis, yaxis, max_interp_points=5000):
    """
    Fits a 3D surface to the LPFP PWM data using griddata.

    To improve performance on large logs, this function will take a random
    sample of the data if the number of points exceeds `max_interp_points`.
    This is much faster and preserves the data's distribution better than aggregation.
    """
    if log_data.empty or len(log_data) < 3:
        return np.zeros((len(yaxis), len(xaxis)))

    # --- START: Performance Optimization using Subsampling ---
    if len(log_data) > max_interp_points:
        # If the dataset is large, take a random sample to speed up interpolation.
        # Using a fixed random_state ensures that the sampling is repeatable
        # for the same input data, making the tuning process deterministic.
        interp_data = log_data.sample(n=max_interp_points, random_state=42)
    else:
        # If the dataset is small enough, use all of it.
        interp_data = log_data
    # --- END: Performance Optimization ---

    # Use the (potentially smaller) interp_data set for all interpolation steps
    points = interp_data[['FF_SP', 'LPFP_FP_SP']].values
    values = interp_data['LPFP_PWM'].values

    if len(points) < 4:
        # This condition is handled by the main orchestrator, but serves as a safeguard.
        return None

    grid_x, grid_y = np.meshgrid(xaxis, yaxis)

    # Perform the primary linear interpolation
    fitted_surface = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

    # Fill any remaining NaN values (outside the convex hull of the data)
    nan_mask = np.isnan(fitted_surface)
    if np.any(nan_mask):
        # Use the same subsampled points for the 'nearest' fill
        nearest_fill = interpolate.griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        fitted_surface[nan_mask] = nearest_fill

    return np.nan_to_num(fitted_surface)

def _calculate_lpfp_correction(log_data, blend_surface, old_table, xaxis, yaxis, confidence):
    """Applies confidence interval logic to determine the final correction table."""
    new_table = old_table.copy()
    changed_mask = np.zeros_like(old_table, dtype=bool)

    for i in range(len(xaxis)):
        for j in range(len(yaxis)):
            cell_data = log_data[(log_data['X'] == i) & (log_data['Y'] == j)]
            if len(cell_data) > 3:
                mean, std_dev = stats.norm.fit(cell_data['LPFP_PWM'])
                low_ci, high_ci = stats.norm.interval(confidence, loc=mean, scale=std_dev if std_dev > 0 else 1e-9)
                current_val = old_table[j, i]

                # --- FIX: Use the specific surface value for the current cell ---
                surface_val = blend_surface[j, i]

                # 1. Define the target value using the original 50/50 blend.
                target_val = (surface_val + mean) / 2

                # 2. Construct the CI around this new blended target.
                low_ci, high_ci = stats.norm.interval(confidence, loc=target_val, scale=std_dev if std_dev > 0 else 1e-9)

                # 3. Decide if a change is needed by comparing the current value to the new CI.
                if not (low_ci <= current_val <= high_ci):
                    # If outside, update the table directly to the target value.
                    new_table[j, i] = target_val
                    changed_mask[j, i] = True

    recommended_table = np.round(new_table * 655.3599999999997) / 655.3599999999997
    return recommended_table, changed_mask

# --- Main Orchestrator Function ---
def run_lpfp_analysis(log, xaxis, yaxis, old_table, logvars):
    """
    Main orchestrator for the LPFP tuning process. A pure computational function.

    Args:
        log (pd.DataFrame): The mapped log data.
        xaxis, yaxis (np.ndarray): The axes for the LPFP table.
        old_table (np.ndarray): The original LPFP table from the tune.
        logvars (list): A list of available variable names in the log.

    Returns:
        dict: A dictionary containing all results.
    """
    print(" -> Initializing LPFP analysis...")
    params = {'confidence': 0.8}

    print(" -> Preparing LPFP data from logs...")
    processed_log, warnings = _process_and_filter_lpfp_log_data(log, logvars)

    if processed_log.empty:
        return {'status': 'Failure', 'warnings': warnings, 'results_lpfp': None}

    print(" -> Creating data bins from LPFP axes...")
    log_binned = _create_bins(processed_log, xaxis, yaxis)

    print(" -> Fitting 3D surface to LPFP data...")
    blend_surface = _fit_surface_lpfp(log_binned, xaxis, yaxis)

    if blend_surface is None:
        warnings.append("Not enough valid data points to create a surface fit.")
        return {'status': 'Failure', 'warnings': warnings, 'results_lpfp': None}

    recommended_table, changed_mask = _calculate_lpfp_correction(
        log_binned, blend_surface, old_table, xaxis, yaxis, params['confidence']
    )

    print(" -> Preparing final results as DataFrame...")
    xlabels = [str(x) for x in xaxis]
    ylabels = [str(y) for y in yaxis]
    result_df = pd.DataFrame(recommended_table, columns=xlabels, index=ylabels)

    print(" -> LPFP analysis complete.")
    return {
        'status': 'Success',
        'warnings': warnings,
        'results_lpfp': result_df
    }