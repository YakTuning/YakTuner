# C:/Users/Sam/PycharmProjects/YAKtunerCONVERTED/WG.py

"""
Wastegate (WG) Tuning Module for YAKtuner

This module contains pure, non-UI functions to analyze engine log data and
recommend adjustments to wastegate base tables.
"""

import numpy as np
import pandas as pd
from scipy import stats, interpolate
import matplotlib.pyplot as plt

# --- Core Calculation and Filtering Functions ---

def _process_and_filter_log_data(log_df, params, logvars, WGlogic, tempcomp, tempcompaxis, min_pedal=50.0):
    """
    A pure function to prepare and filter log data without UI interactions.
    """
    warnings = []
    processed_log = log_df.copy()

    # Determine SWG/FF logic and apply temperature correction
    if WGlogic:
        processed_log['EFF'] = processed_log['RPM']
        processed_log['IFF'] = processed_log['PUTSP'] * 10
        interp_func = interpolate.interp1d(tempcompaxis, tempcomp, kind='linear', fill_value='extrapolate')
        tempcorr = interp_func(processed_log['AMBTEMP'])
    else:
        tempcorr = 0

    # Create derived values for analysis
    processed_log['deltaPUT'] = processed_log['PUT'] - processed_log['PUTSP']
    processed_log['WGNEED_uncorrected'] = processed_log['WG_Final'] - processed_log['deltaPUT'] * params['fudge']
    processed_log['WGNEED'] = processed_log['WGNEED_uncorrected'] - tempcorr

    # Filter log data to valid conditions
    if 'I_INH' in logvars:
        processed_log = processed_log[processed_log['I_INH'] <= 0]
    else:
        warnings.append("Recommend logging 'PUT I Inhibit'. Using pedal position as a fallback.")
        processed_log = processed_log[processed_log['Pedal'] >= min_pedal]

    if 'DV' in logvars:
        processed_log = processed_log[processed_log['DV'] <= 50]
    else:
        warnings.append("Recommend logging 'DV position'. Otherwise, DV may impact accuracy.")

    if 'BOOST' in logvars:
        processed_log = processed_log[processed_log['BOOST'] >= params['minboost']]
    else:
        warnings.append("Recommend logging 'boost'. Otherwise, logs are not trimmed for min boost.")

    # Filtering Logic for PUT Delta stability
    processed_log['deltaPUT_CHANGE'] = processed_log['deltaPUT'].diff().abs()
    is_small_delta = processed_log['deltaPUT'].abs() < 10
    is_steady_delta = processed_log['deltaPUT_CHANGE'] < 1.0
    final_mask = is_small_delta | is_steady_delta
    processed_log = processed_log[final_mask.fillna(False)]

    if processed_log.empty:
        warnings.append("No data points met the criteria (small or steady PUT delta).")

    # Final filter for WG duty cycle range
    processed_log = processed_log[processed_log['WG_Final'] <= 98]

    return processed_log, warnings

def _create_bins_and_labels(log_df, wgxaxis, wgyaxis):
    """Creates bin edges from axes and assigns each log entry to a grid cell (X, Y)."""
    wgxedges = np.zeros(len(wgxaxis) + 1)
    wgxedges[0] = wgxaxis[0]
    wgxedges[-1] = wgxaxis[-1] + 2
    for i in range(len(wgxaxis) - 1):
        wgxedges[i + 1] = (wgxaxis[i] + wgxaxis[i + 1]) / 2

    wgyedges = np.zeros(len(wgyaxis) + 1)
    wgyedges[0] = wgyaxis[0]
    wgyedges[-1] = wgyaxis[-1] + 2
    for i in range(len(wgyaxis) - 1):
        wgyedges[i + 1] = (wgyaxis[i] + wgyaxis[i + 1]) / 2

    log_df['X'] = pd.cut(log_df['EFF'], wgxedges, labels=False)
    log_df['Y'] = pd.cut(log_df['IFF'], wgyedges, labels=False)
    return log_df

def create_wg_scatter_plot(log_VVL0, log_VVL1, wgxaxis, wgyaxis, WGlogic):
    """
    Creates a Matplotlib scatter plot figure of the filtered log data.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter1 = ax.scatter(
        log_VVL1['EFF'], log_VVL1['IFF'], s=abs(log_VVL1['WGNEED']),
        c=log_VVL1['deltaPUT'], marker='x', cmap='RdBu', label='VVL1'
    )
    ax.scatter(
        log_VVL0['EFF'], log_VVL0['IFF'], s=abs(log_VVL0['WGNEED']),
        c=log_VVL0['deltaPUT'], marker='o', cmap='RdBu', label='VVL0', alpha=0.7
    )
    cbar = fig.colorbar(scatter1, ax=ax)
    cbar.set_label('PUT - PUT SP (kPa)')
    ax.invert_yaxis()
    if WGlogic:
        ax.set_xlabel('RPM')
        ax.set_ylabel('PUT SP')
    else:
        ax.set_xlabel('Engine Efficiency (EFF)')
        ax.set_ylabel('Intake Flow Factor (IFF)')
    ax.set_title('Wastegate Duty Cycle Need vs. Operating Point')
    ax.grid(True)
    ax.set_xticks(wgxaxis)
    ax.set_xticklabels(labels=wgxaxis, rotation=45)
    ax.set_yticks(wgyaxis)
    ax.legend()
    fig.tight_layout()
    return fig

def _fit_surface(log_data, wgxaxis, wgyaxis):
    """
    Fits a 3D surface to the provided log data using scipy.interpolate.griddata.
    """
    if log_data.empty or len(log_data) < 3:
        return np.zeros((len(wgyaxis), len(wgxaxis)))
    points = log_data[['EFF', 'IFF']].values
    values = log_data['WGNEED'].values
    grid_x, grid_y = np.meshgrid(wgxaxis, wgyaxis)
    fitted_surface = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')
    nan_mask = np.isnan(fitted_surface)
    if np.any(nan_mask):
        nearest_fill = interpolate.griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        fitted_surface[nan_mask] = nearest_fill
    if np.all(np.isnan(fitted_surface)):
        return np.zeros((len(wgyaxis), len(wgxaxis)))
    return fitted_surface / 100.0

def _calculate_final_recommendations(log_data, blend, old_table, wgxaxis, wgyaxis, calculate_temp_coef=False):
    """
    Calculates the final recommended WG table by comparing the new fit with the old
    table and applying confidence intervals to each cell using a predictive model.
    """
    final_table = old_table.copy()
    total_coef, count_coef = 0, 0

    # Define blending factor and confidence level
    interp_factor = 0.5  # 50/50 blend of surface fit and local data
    confidence = 0.7  # Confidence level for the interval check

    for i in range(len(wgxaxis)):
        for j in range(wgyaxis.shape[0]):
            cell_data = log_data[(log_data['X'] == i) & (log_data['Y'] == j)]
            if len(cell_data) > 3:
                # Get stats from raw data. `mean` and `std_dev` are in percent (0-100).
                mean, std_dev = stats.norm.fit(cell_data['WGNEED'])

                # Get surface value. `blend` is a factor (0-1).
                surface_val_factor = blend[j, i]

                # Get current value. `old_table` is a factor (0-1).
                current_val_factor = old_table[j, i]

                # --- START: New Logic ---
                # 1. Define a 'target' by blending the global surface fit and the local cell mean.
                #    Both values are converted to factors (0-1) for the calculation.
                target_val_factor = (surface_val_factor * interp_factor) + ((mean / 100.0) * (1 - interp_factor))

                # 2. Construct the CI around this new blended target.
                #    The CI is calculated in factors, so the scale (std_dev) is also converted.
                low_ci_factor, high_ci_factor = stats.norm.interval(
                    confidence,
                    loc=target_val_factor,
                    scale=(std_dev / 100.0) if std_dev > 0 else 1e-9
                )

                # 3. Decide if a change is needed by comparing the current value (factor) to the new CI (factor).
                if np.isnan(current_val_factor) or not (low_ci_factor <= current_val_factor <= high_ci_factor):
                    # If outside the CI, update the table to the blended target value.
                    final_table[j, i] = target_val_factor
                # --- END: New Logic ---
            if calculate_temp_coef:
                temp_log = cell_data[cell_data['BOOST'] >= 8]
                if len(temp_log) > 2 and (temp_log['AMBTEMP'].max() > temp_log['AMBTEMP'].min() + 15):
                    coef = np.polyfit(temp_log['AMBTEMP'], temp_log['WGNEED_uncorrected'] / 100, 1)[0]
                    total_coef += coef * len(temp_log)
                    count_coef += len(temp_log)
    final_table = np.round(final_table * 16384) / 16384
    if calculate_temp_coef:
        avg_coef = total_coef / count_coef if count_coef > 0 else None
        return final_table, avg_coef
    else:
        return final_table

# --- Main Orchestrator Function ---
def run_wg_analysis(log_df, wgxaxis, wgyaxis, oldWG0, oldWG1, logvars, WGlogic, tempcomp, tempcompaxis, show_scatter_plot=True):
    """
    Main orchestrator for the WG tuning process. A pure computational function.

    Args:
        log_df (pd.DataFrame): The mapped log data.
        wgxaxis, wgyaxis (np.ndarray): The axes for the WG tables.
        oldWG0, oldWG1 (np.ndarray): The original WG tables from the tune.
        logvars (list): A list of available variable names in the log.
        WGlogic (bool): Flag for SWG/FF logic.
        tempcomp, tempcompaxis (np.ndarray): Temperature compensation data.
        show_scatter_plot (bool): Flag to control generation of the scatter plot.

    Returns:
        dict: A dictionary containing all results:
              - 'status' (str): 'Success' or 'Failure'.
              - 'warnings' (list): A list of warning messages.
              - 'scatter_plot_fig' (matplotlib.figure.Figure or None): The generated plot.
              - 'results_vvl0' (pd.DataFrame or None): The recommended VVL0 table.
              - 'results_vvl1' (pd.DataFrame or None): The recommended VVL1 table.
              - 'temp_comp_results' (pd.DataFrame or None): Temp comp recommendations.
    """
    print(" -> Initializing WG analysis...")
    # Parameters are now hardcoded here, but could be passed in from the UI.
    params = {'fudge': 0.71, 'minboost': 0}
    temp_comp_results = None
    original_intercept = 0.0
    _slope = 0.0

    if WGlogic and tempcomp is not None and tempcompaxis is not None:
        try:
            # Check if the arrays have enough points to fit a line
            if len(tempcomp) > 1 and len(tempcompaxis) > 1:
                _slope, original_intercept = np.polyfit(tempcompaxis, tempcomp, 1)
                print(f" -> Original Temp Comp Slope: {_slope}, Intercept: {original_intercept}")
            else:
                warnings.append("Not enough data points in tempcomp/tempcompaxis to calculate slope.")
        except (np.linalg.LinAlgError, TypeError) as e:
            warnings.append(f"Could not calculate temp comp slope due to an error: {e}")

    print(" -> Preparing and filtering log data...")
    processed_log, warnings = _process_and_filter_log_data(
        log_df=log_df, params=params, logvars=logvars, WGlogic=WGlogic,
        tempcomp=tempcomp, tempcompaxis=tempcompaxis
    )

    if processed_log.empty:
        return {'status': 'Failure', 'warnings': warnings, 'scatter_plot_fig': None,
                'results_vvl0': None, 'results_vvl1': None, 'temp_comp_results': None}

    print(" -> Creating data bins from WG axes...")
    log = _create_bins_and_labels(processed_log, wgxaxis, wgyaxis)

    print(" -> Separating data for VVL0 and VVL1...")
    log_VVL1 = log[log['VVL'] == 1].copy()
    log_VVL0 = log[log['VVL'] == 0].copy()

    scatter_fig = None
    if show_scatter_plot:
        print(" -> Generating raw WG data plot...")
        scatter_fig = create_wg_scatter_plot(log_VVL0, log_VVL1, wgxaxis, wgyaxis, WGlogic)

    print(" -> Fitting 3D surface for VVL1...")
    blend1 = _fit_surface(log_VVL1, wgxaxis, wgyaxis)

    print(" -> Fitting 3D surface for VVL0...")
    blend0 = _fit_surface(log_VVL0, wgxaxis, wgyaxis)

    print(" -> Calculating final recommendations for VVL1...")
    final_table_1 = _calculate_final_recommendations(log_VVL1, blend1, oldWG1, wgxaxis, wgyaxis)

    print(" -> Calculating final recommendations for VVL0 and Temp Comp...")
    final_table_0, avg_coef = _calculate_final_recommendations(
        log_VVL0, blend0, oldWG0, wgxaxis, wgyaxis, calculate_temp_coef=True
    )

    # NOTE: 3D plotting logic is removed from here. It should be handled in the
    # Streamlit script if desired, using a refactored `plot_3d_surface` function.

    temp_comp_results_df = None

    if WGlogic and tempcomp is not None and tempcompaxis is not None:
        print(" -> Preparing temperature compensation results...")
        if avg_coef is not None:
            _slope, original_intercept = np.polyfit(tempcompaxis, tempcomp, 1)
            new_tempcomp = (avg_coef * tempcompaxis) + original_intercept
            temp_df = pd.DataFrame({
                'Temperature': tempcompaxis,
                'Original Comp': tempcomp,
                'Recommended Comp': new_tempcomp
            })
            temp_comp_results_df = temp_df.set_index('Temperature').T.round(4)

    print(" -> Preparing final results as DataFrames...")
    exhlabels = [str(x) for x in wgxaxis]
    intlabels = [str(x) for x in wgyaxis]
    Res_1 = pd.DataFrame(final_table_1, columns=exhlabels, index=intlabels)
    Res_0 = pd.DataFrame(final_table_0, columns=exhlabels, index=intlabels)

    print(" -> WG analysis complete.")
    return {
        'status': 'Success',
        'warnings': warnings,
        'scatter_plot_fig': scatter_fig,
        'results_vvl0': Res_0,
        'results_vvl1': Res_1,
        'temp_comp_results': temp_comp_results_df
    }