
"""
Multiplicative Fuel Factor (MFF) Tuning Module

This module contains pure, non-UI functions to analyze engine logs and recommend
adjustments to the five primary MFF tables (IDX0-IDX4).
"""

import numpy as np
import pandas as pd
from scipy import stats, interpolate

# --- Helper Functions ---

def _get_mff_parameters():
    """Returns hardcoded parameters for MFF tuning."""
    params = {
        'confidence': 0.9,
        'show_3d_plot': True
    }
    return params


def _process_and_filter_mff_data(log, logvars, tuning_mode='MFF'):
    """
    A pure function to prepare and filter log data for MFF tuning using the new
    unified correction formula. It returns a processed DataFrame and a list of warnings.

    Args:
        log (pd.DataFrame): The log data, which may contain pre-calculated columns.
        logvars (list): A list of available variable names in the original log.
        tuning_mode (str): 'MFF' for standalone MFF tuning, or 'BOTH' when run
                           after a MAF correction stage.
    """
    warnings = []
    df = log.copy()

    # --- Pre-filtering ---
    if "OILTEMP" in logvars:
        df = df[df['OILTEMP'] > 180].copy()

    # --- New Unified Correction Formula ---
    # Check for essential variables for the new formula
    required_vars = ['LAMBDA', 'LAMBDA_SP']
    if not all(v in df.columns for v in required_vars):
        raise ValueError(f"MFF analysis requires essential log variables: {required_vars}")

    # Gracefully get all potential correction factors from the log.
    # Defaults are chosen to be neutral (0 for additive, 1 for multiplicative).
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

    # --- Calculate the new MFF_FACTOR based on the tuning mode ---
    if tuning_mode == 'MFF':
        # MFF-only mode: Assume MAF table is correct and unchanged.
        # Solve for MFF_COR_NEW = Target_Factor / (1 + MAF_COR_current)
        mff_cor_new = target_factor / (1 + maf_cor/100)
        df.loc[:, 'MFF_FACTOR'] = mff_cor_new

    elif tuning_mode == 'BOTH':
        # MFF-as-second-stage mode: A new MAF correction has already been determined.
        # This new correction is passed in the 'MAF_COR_NEW' column.
        if 'MAF_COR_NEW' not in df.columns:
            raise ValueError("'tuning_mode' is 'BOTH', but the log DataFrame is missing the 'MAF_COR_NEW' column.")

        maf_cor_new = df['MAF_COR_NEW']
        # Solve for MFF_COR_NEW = Target_Factor / (1 + MAF_COR_NEW)
        mff_cor_new = target_factor / (1 + maf_cor_new / 100)
        df.loc[:, 'MFF_FACTOR'] = mff_cor_new

    return df, warnings

def _create_bins(log, mffxaxis, mffyaxis):
    """Discretizes log data into bins based on MFF map axes."""
    xedges = [0] + [(mffxaxis[i] + mffxaxis[i + 1]) / 2 for i in range(len(mffxaxis) - 1)] + [np.inf]
    yedges = [0] + [(mffyaxis[i] + mffyaxis[i + 1]) / 2 for i in range(len(mffyaxis) - 1)] + [np.inf]

    log.loc[:, 'X'] = pd.cut(log['RPM'], bins=xedges, labels=False, duplicates='drop')
    log.loc[:, 'Y'] = pd.cut(log['MAF'], bins=yedges, labels=False, duplicates='drop') # Use Airmass (MAF) for Y-axis
    return log

def _fit_surface_mff(log_data, mffxaxis, mffyaxis):
    """Fits a 3D surface to the MFF correction data using griddata."""
    if log_data.empty or len(log_data) < 3:
        return np.ones((len(mffyaxis), len(mffxaxis))) # Default to 1.0 for multiplicative factor

    points = log_data[['RPM', 'MAF']].values
    values = log_data['MFF_FACTOR'].values
    grid_x, grid_y = np.meshgrid(mffxaxis, mffyaxis)

    fitted_surface = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')

    # Fill any NaN values that result from sparse data regions using the nearest data point.
    nan_mask = np.isnan(fitted_surface)
    if np.any(nan_mask):
        nearest_fill = interpolate.griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        fitted_surface[nan_mask] = nearest_fill

    # As a final safety net, convert any remaining NaNs to the neutral value of 1.0
    filled_surface = np.nan_to_num(fitted_surface, nan=1.0)

    # --- START: New Clamping Logic ---
    # Clamp the final surface to a believable range (e.g., 0.92 to 1.08).
    # This prevents extreme values that can be artifacts of linear interpolation
    # without affecting the well-behaved parts of the surface.
    believable_min = 0.92
    believable_max = 1.08
    clamped_surface = np.clip(filled_surface, believable_min, believable_max)
    # --- END: New Clamping Logic ---

    return clamped_surface

def _calculate_mff_correction(log_data, blend_surface, old_table, mffxaxis, mffyaxis, confidence, additive_mode=False):
    """
    Applies confidence interval logic to determine the final correction table.
    Supports standard (replacement) and additive correction modes.
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

                # 1. Define a 'target' correction by blending the global surface fit and the local cell mean.
                target_val = (surface_val * interp_factor) + (mean * (1 - interp_factor))

                # 2. Construct the Confidence Interval around this new blended target.
                low_ci, high_ci = stats.norm.interval(confidence, loc=target_val, scale=std_dev if std_dev > 0 else 1e-9)

                # The value from the original tune file
                current_val_from_table = old_table[j, i]

                # The value to compare against the confidence interval.
                # In additive mode, we compare against 1.0 (neutral) to see if any correction is needed.
                # In standard mode, we compare against the table's current value.
                comparison_val = 1.0 if additive_mode else current_val_from_table

                # 3. Decide if a change is needed by comparing the comparison value to the new CI.
                if not (low_ci <= comparison_val <= high_ci):
                    # If a change is needed, calculate the amount of change.
                    # Weight the change by the number of data points to control aggressiveness.
                    weight = min(count, max_count) / max_count

                    # The change amount is the difference between the target and the value we compared against.
                    change_amount = (target_val - comparison_val) * weight

                    # Apply the change to the value from the original table.
                    # This works for both modes because in standard mode, change_amount is relative to the current value.
                    new_table[j, i] = current_val_from_table + change_amount

    # Quantize the final table to a common ECU resolution for multiplicative factors (1/1024)
    recommended_table = np.round(new_table * 1024) / 1024
    return recommended_table

# --- Main Orchestrator Function ---
def run_mff_analysis(log, mffxaxis, mffyaxis, mfftables, combmodes_MFF, logvars, tuning_mode='MFF'):
    """
    Main orchestrator for the MFF tuning process. A pure computational function.

    Args:
        log (pd.DataFrame): The mapped log data.
        mffxaxis, mffyaxis (np.ndarray): The axes for the MFF tables.
        mfftables (list[np.ndarray]): A list of the five original MFF tables.
        combmodes_MFF (np.ndarray): The combination modes map.
        logvars (list): A list of available variable names in the log.
        tuning_mode (str): 'MFF' for standalone, or 'BOTH' for second-stage tuning.

    Returns:
        dict: A dictionary containing all results.
    """
    print(" -> Initializing MFF analysis...")
    params = {'confidence': 0.7}  # Hardcoded parameter

    print(" -> Preparing MFF data from logs...")
    processed_log, warnings = _process_and_filter_mff_data(log, logvars, tuning_mode=tuning_mode)

    # Determine if we are in additive mode (if MFF_COR is not available in logs)
    additive_mode = 'MFF_COR' not in logvars
    if additive_mode:
        warnings.append("MFF_COR not found in logs. Switching to additive correction mode.")

    if processed_log.empty:
        return {'status': 'Failure', 'warnings': warnings, 'results_mff': None}

    print(" -> Creating data bins from MFF axes...")
    log_binned = _create_bins(processed_log, mffxaxis, mffyaxis)

    results = {}
    # Loop through all 5 MFF tables
    for idx in range(5):
        print(f" -> Processing MFF Table IDX{idx}...")
        current_table = mfftables[idx]
        idx_modes = np.where(combmodes_MFF == idx)[0]
        log_filtered = log_binned[log_binned['CMB'].isin(idx_modes)].copy()
        log_filtered.dropna(subset=['RPM', 'MAF', 'MFF_FACTOR'], inplace=True)

        print(f"   -> Fitting 3D surface for IDX{idx}...")
        blend_surface = _fit_surface_mff(log_filtered, mffxaxis, mffyaxis)

        print(f"   -> Calculating correction map for IDX{idx}...")
        recommended_table = _calculate_mff_correction(
            log_filtered, blend_surface, current_table, mffxaxis, mffyaxis, params['confidence'], additive_mode=additive_mode
        )

        # Store results as a DataFrame
        xlabels = [str(x) for x in mffxaxis]
        ylabels = [str(y) for y in mffyaxis]
        results[f'IDX{idx}'] = pd.DataFrame(recommended_table, columns=xlabels, index=ylabels)

    # 3D plotting and table display are now handled by the UI (Streamlit).

    print(" -> MFF analysis complete.")
    return {
        'status': 'Success',
        'warnings': warnings,
        'results_mff': results
    }