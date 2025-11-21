# C:/Users/Sam/PycharmProjects/YAKtunerCONVERTED/MAF.py

"""
Mass Airflow (MAF) Correction Tuning Module

This module contains pure, non-UI functions to analyze engine logs and recommend
adjustments to the four primary MAF correction tables (IDX0-IDX3).
"""

import numpy as np
import pandas as pd
from scipy import stats, interpolate

# --- Helper Functions ---

def _process_and_filter_maf_data(log, logvars):
    """
    A pure function to prepare and filter log data for MAF tuning using the new
    unified correction formula.
    """
    warnings = []
    df = log.copy()

    # --- Pre-filtering (add your MAF-specific filters here) ---
    if "OILTEMP" in logvars:
        df = df[df['OILTEMP'] > 180].copy()

    # --- FIX: Unit conversion for MAP (kPa to hPa/mbar) ---
    # The log provides MAP in kPa, but the table axes are in hPa.
    # Multiply by 10 to ensure data is binned and interpolated correctly.
    if 'MAP' in df.columns:
        df.loc[:, 'MAP'] = df['MAP'] * 10
    else:
        # This is a critical variable, so we should warn if it's missing.
        warnings.append("Log variable 'MAP' not found. MAF analysis will likely fail.")
        return pd.DataFrame(), warnings
    # --- END FIX ---

    # --- New Unified Correction Formula ---
    required_vars = ['LAMBDA', 'LAMBDA_SP']
    if not all(v in df.columns for v in required_vars):
        raise ValueError(f"MAF analysis requires essential log variables: {required_vars}")

    # Get all potential correction factors, with defaults
    maf_cor = df.get('MAF_COR', 0.0)
    if 'MAF_COR' not in logvars: warnings.append("Log 'MAF_COR' for best accuracy.")

    mff_cor = df.get('MFF_COR', 1.0)
    if 'MFF_COR' not in logvars: warnings.append("Log 'MFF_COR' for best accuracy.")

    # --- Retrieve all primary and fallback variables for STFT and LTFT ---
    fac_stft = df.get('FAC_STFT', 0.0)
    fac_lam_out = df.get('FAC_LAM_OUT', 0.0)
    stft = df.get('STFT', 0.0)

    fac_ltft = df.get('FAC_LTFT', 0.0)
    add_ltft = df.get('ADD_LTFT', 0.0)
    fac_mff_add = df.get('FAC_MFF_ADD', 0.0)
    ltft = df.get('LTFT', 0.0)

    # --- Construct LTFT Correction Term with Degradation ---
    ltft_correction_term = 1.0
    if 'FAC_LTFT' in logvars and 'ADD_LTFT' in logvars:
        ltft_correction_term = (1 + (fac_ltft + abs(fac_ltft) * add_ltft) / 100)
    elif 'FAC_MFF_ADD' in logvars:
        ltft_correction_term = (1 + fac_mff_add / 100)
        warnings.append("Using 'FAC_MFF_ADD' as fallback for LTFT correction.")
    elif 'LTFT' in logvars:
        ltft_correction_term = (1 + ltft / 100)
        warnings.append("Using 'LTFT' as fallback for LTFT correction.")
    else:
        warnings.append("No suitable LTFT correction variable found. Assuming neutral LTFT correction (1.0).")

    # --- Construct STFT Correction Term with Degradation ---
    stft_correction_term = 1.0
    if 'FAC_STFT' in logvars:
        stft_correction_term = (1 + fac_stft / 100)
    elif 'FAC_LAM_OUT' in logvars:
        stft_correction_term = (1 + fac_lam_out / 100)
        warnings.append("Using 'FAC_LAM_OUT' as fallback for STFT correction.")
    elif 'STFT' in logvars:
        stft_correction_term = (1 + stft / 100)
        warnings.append("Using 'STFT' as fallback for STFT correction.")
    else:
        warnings.append("No suitable STFT correction variable found. Assuming neutral STFT correction (1.0).")

    # Calculate the total target correction factor needed.
    # Target_Factor = (all current ECU factors) * (measured_error)
    total_ecu_factor = (1 + maf_cor/100) * stft_correction_term * mff_cor * ltft_correction_term
    measured_error = df['LAMBDA'] / df['LAMBDA_SP']
    target_factor = total_ecu_factor * measured_error

    # --- Calculate the new ADD_MAF value ---
    # In MAF tuning, we solve for the additive correction MAF_COR_NEW.
    # The MFF part of the new correction is assumed to be 1.
    # Target_Factor = (1 + MAF_COR_NEW) * 1
    maf_cor_new = target_factor - 1
    df.loc[:, 'ADD_MAF'] = maf_cor_new  # The rest of the MAF module works with 'ADD_MAF'

    return df, warnings

def _create_bins(log, mafxaxis, mafyaxis):
    """Discretizes log data into bins based on MAF map axes."""
    xedges = [0] + [(mafxaxis[i] + mafxaxis[i + 1]) / 2 for i in range(len(mafxaxis) - 1)] + [np.inf]
    yedges = [0] + [(mafyaxis[i] + mafyaxis[i + 1]) / 2 for i in range(len(mafyaxis) - 1)] + [np.inf]

    log.loc[:, 'X'] = pd.cut(log['RPM'], bins=xedges, labels=False, duplicates='drop')
    log.loc[:, 'Y'] = pd.cut(log['MAP'], bins=yedges, labels=False, duplicates='drop')
    return log

def _fit_surface_maf(log_data, mafxaxis, mafyaxis):
    """Fits a 3D surface to the MAF correction data using griddata."""
    if log_data.empty or len(log_data) < 3:
        return np.zeros((len(mafyaxis), len(mafxaxis)))

    points = log_data[['RPM', 'MAP']].values
    values = log_data['ADD_MAF'].values
    grid_x, grid_y = np.meshgrid(mafxaxis, mafyaxis)

    fitted_surface = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

    nan_mask = np.isnan(fitted_surface)
    if np.any(nan_mask):
        nearest_fill = interpolate.griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        fitted_surface[nan_mask] = nearest_fill

    return np.nan_to_num(fitted_surface)

def _calculate_maf_correction(log_data, blend_surface, old_table, mafxaxis, mafyaxis, confidence, additive_mode=False):
    """
    Applies confidence interval logic to determine the final correction table.
    Supports standard (replacement) and additive correction modes.
    """
    new_table = old_table.copy()
    max_count = 80
    interp_factor = 0.5

    for i in range(len(mafxaxis)):
        for j in range(len(mafyaxis)):
            cell_data = log_data[(log_data['X'] == i) & (log_data['Y'] == j)]
            count = len(cell_data)

            if count > 3:
                mean, std_dev = stats.norm.fit(cell_data['ADD_MAF'])
                surface_val = blend_surface[j, i]

                # 1. Define a 'target' correction by blending the global surface fit and the local cell mean.
                target_val = (surface_val * interp_factor) + (mean * (1 - interp_factor))

                # 2. Construct the Confidence Interval around this new blended target.
                low_ci, high_ci = stats.norm.interval(confidence, loc=target_val, scale=std_dev if std_dev > 0 else 1e-9)

                # The value from the original tune file
                current_val_from_table = old_table[j, i]

                # The value to compare against the confidence interval.
                # In additive mode, we compare against 0 to see if any correction is needed.
                # In standard mode, we compare against the table's current value to see if it's already correct.
                comparison_val = 0.0 if additive_mode else current_val_from_table

                # 3. Decide if a change is needed by comparing the comparison value to the new CI.
                if not (low_ci <= comparison_val <= high_ci):
                    # If a change is needed, calculate the amount of change.
                    # Weight the change by the number of data points to control aggressiveness.
                    weight = min(count, max_count) / max_count

                    # The change amount is the difference between the target and the value we compared against.
                    change_amount = (target_val - comparison_val) * weight

                    # Apply the change to the value from the original table.
                    new_table[j, i] = current_val_from_table + change_amount

    # Quantize the final table to the ECU's resolution (5.12 = 256 / 50)
    recommended_table = np.round(new_table * 5.12) / 5.12
    return recommended_table

# --- Main Orchestrator Function ---
def run_maf_analysis(log, mafxaxis, mafyaxis, maftables, combmodes_MAF, logvars):
    """
    Main orchestrator for the MAF tuning process. A pure computational function.

    Args:
        log (pd.DataFrame): The mapped log data.
        mafxaxis, mafyaxis (np.ndarray): The axes for the MAF tables.
        maftables (list[np.ndarray]): A list of the four original MAF tables.
        combmodes_MAF (np.ndarray): The combination modes map.
        logvars (list): A list of available variable names in the log.

    Returns:
        dict: A dictionary containing all results.
    """
    print(" -> Initializing MAF analysis...")
    params = {'confidence': 0.6} # Hardcoded parameter

    print(" -> Preparing MAF data from logs...")
    processed_log, warnings = _process_and_filter_maf_data(log, logvars)

    # Determine if we are in additive mode (if MAF_COR is not available in logs)
    additive_mode = 'MAF_COR' not in logvars
    if additive_mode:
        warnings.append("MAF_COR not found in logs. Switching to additive correction mode.")

    if processed_log.empty:
        return {'status': 'Failure', 'warnings': warnings, 'results_maf': None}

    print(" -> Creating data bins from MAF axes...")
    log_binned = _create_bins(processed_log, mafxaxis, mafyaxis)

    results = {}
    for idx in range(4):
        print(f" -> Processing MAF Table IDX{idx}...")
        current_table = maftables[idx]
        idx_modes = np.where(combmodes_MAF == idx)[0]
        log_filtered = log_binned[log_binned['CMB'].isin(idx_modes)].copy()
        log_filtered.dropna(subset=['RPM', 'MAP', 'ADD_MAF'], inplace=True)

        print(f"   -> Fitting 3D surface for IDX{idx}...")
        blend_surface = _fit_surface_maf(log_filtered, mafxaxis, mafyaxis)

        print(f"   -> Calculating correction map for IDX{idx}...")
        recommended_table = _calculate_maf_correction(
            log_filtered, blend_surface, current_table, mafxaxis, mafyaxis, params['confidence'], additive_mode=additive_mode
        )

        # Store results as a DataFrame
        xlabels = [str(x) for x in mafxaxis]
        ylabels = [str(y) for y in mafyaxis]
        results[f'IDX{idx}'] = pd.DataFrame(recommended_table, columns=xlabels, index=ylabels)

    # 3D plotting and table display are now handled by the UI (Streamlit).

    print(" -> MAF analysis complete.")
    return {
        'status': 'Success',
        'warnings': warnings,
        'results_maf': results
    }