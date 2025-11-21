"""
Multiplicative Fuel Factor (MFF) Tuning Module

This module contains pure, non-UI functions to analyze engine logs and recommend
adjustments to the primary MFF table.
"""

import numpy as np
import pandas as pd
from scipy import stats, interpolate

# --- Helper Functions ---

def _process_and_filter_mff_data(log, logvars):
    """
    A pure function to prepare and filter log data for MFF tuning.
    It returns a processed DataFrame and a list of warnings.
    """
    warnings = []
    df = log.copy()

    # --- Pre-filtering ---
    if "OILTEMP" in logvars:
        df = df[df['OILTEMP'] > 180].copy()

    # --- Unified Correction Formula ---
    required_vars = ['LAMBDA', 'LAMBDA_SP', 'LOAD'] # Ensure LOAD is present
    if not all(v in df.columns for v in required_vars):
        raise ValueError(f"MFF analysis requires essential log variables: {required_vars}")

    mff_cor = df.get('MFF_COR', 1.0)
    stft = df.get('STFT', 0.0)
    ltft = df.get('LTFT', 0.0)

    ltft_correction_term = (1 + ltft / 100) if 'LTFT' in logvars else 1.0
    stft_correction_term = (1 + stft / 100) if 'STFT' in logvars else 1.0

    # Calculate the total target correction factor needed.
    total_ecu_factor = stft_correction_term * mff_cor * ltft_correction_term
    measured_error = df['LAMBDA'] / df['LAMBDA_SP']
    target_factor = total_ecu_factor * measured_error

    # Since the MAF module is removed, the MAF correction is assumed to be neutral (0).
    # The new MFF factor is simply the total target factor.
    df.loc[:, 'MFF_FACTOR'] = target_factor

    return df, warnings

def _create_bins(log, mffxaxis, mffyaxis):
    """Discretizes log data into bins based on MFF map axes."""
    xedges = [0] + [(mffxaxis[i] + mffxaxis[i + 1]) / 2 for i in range(len(mffxaxis) - 1)] + [np.inf]
    yedges = [0] + [(mffyaxis[i] + mffyaxis[i + 1]) / 2 for i in range(len(mffyaxis) - 1)] + [np.inf]

    log.loc[:, 'X'] = pd.cut(log['RPM'], bins=xedges, labels=False, duplicates='drop')
    # --- FIX: Use LOAD for the Y-axis ---
    log.loc[:, 'Y'] = pd.cut(log['LOAD'], bins=yedges, labels=False, duplicates='drop')
    return log

def _fit_surface_mff(log_data, mffxaxis, mffyaxis):
    """Fits a 3D surface to the MFF correction data using griddata."""
    if log_data.empty or len(log_data) < 3:
        return np.ones((len(mffyaxis), len(mffxaxis))) # Default to 1.0

    # --- FIX: Use LOAD for the Y-axis in fitting ---
    points = log_data[['RPM', 'LOAD']].values
    values = log_data['MFF_FACTOR'].values
    grid_x, grid_y = np.meshgrid(mffxaxis, mffyaxis)

    fitted_surface = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

    nan_mask = np.isnan(fitted_surface)
    if np.any(nan_mask):
        nearest_fill = interpolate.griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        fitted_surface[nan_mask] = nearest_fill

    filled_surface = np.nan_to_num(fitted_surface, nan=1.0)

    # Clamp the final surface to a believable range to prevent extreme values
    clamped_surface = np.clip(filled_surface, 0.92, 1.08)
    return clamped_surface

def _calculate_mff_correction(log_data, blend_surface, old_table, mffxaxis, mffyaxis, confidence, additive_mode=False):
    """
    Applies confidence interval logic to determine the final correction table.
    """
    new_table = old_table.copy()
    max_count = 50
    interp_factor = 0.25

    for i in range(len(mffxaxis)):
        for j in range(len(mffyaxis)):
            cell_data = log_data[(log_data['X'] == i) & (log_data['Y'] == j)]
            count = len(cell_data)

            if count > 3:
                mean, std_dev = stats.norm.fit(cell_data['MFF_FACTOR'])
                surface_val = blend_surface[j, i]
                target_val = (surface_val * interp_factor) + (mean * (1 - interp_factor))
                low_ci, high_ci = stats.norm.interval(confidence, loc=target_val, scale=std_dev if std_dev > 0 else 1e-9)
                current_val_from_table = old_table[j, i]
                comparison_val = 1.0 if additive_mode else current_val_from_table

                if not (low_ci <= comparison_val <= high_ci):
                    weight = min(count, max_count) / max_count
                    change_amount = (target_val - comparison_val) * weight
                    new_table[j, i] = current_val_from_table + change_amount

    recommended_table = np.round(new_table * 1024) / 1024
    return recommended_table

# --- Main Orchestrator Function ---
def run_mff_analysis(log, mffxaxis, mffyaxis, mfftable, logvars):
    """
    Main orchestrator for the MFF tuning process. A pure computational function.
    """
    print(" -> Initializing MFF analysis...")
    params = {'confidence': 0.7}

    print(" -> Preparing MFF data from logs...")
    processed_log, warnings = _process_and_filter_mff_data(log, logvars)

    additive_mode = 'MFF_COR' not in logvars
    if additive_mode:
        warnings.append("MFF_COR not found in logs. Switching to additive correction mode.")

    if processed_log.empty:
        return {'status': 'Failure', 'warnings': warnings, 'results_mff': None}

    print(" -> Creating data bins from MFF axes...")
    log_binned = _create_bins(processed_log, mffxaxis, mffyaxis)
    log_binned.dropna(subset=['RPM', 'LOAD', 'MFF_FACTOR'], inplace=True)

    print("   -> Fitting 3D surface...")
    blend_surface = _fit_surface_mff(log_binned, mffxaxis, mffyaxis)

    print("   -> Calculating correction map...")
    recommended_table = _calculate_mff_correction(
        log_binned, blend_surface, mfftable, mffxaxis, mffyaxis, params['confidence'], additive_mode=additive_mode
    )

    xlabels = [str(x) for x in mffxaxis]
    ylabels = [str(y) for y in mffyaxis]
    result_df = pd.DataFrame(recommended_table, columns=xlabels, index=ylabels)

    print(" -> MFF analysis complete.")
    return {
        'status': 'Success',
        'warnings': warnings,
        'results_mff': result_df
    }