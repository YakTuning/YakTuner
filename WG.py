# C:/Users/Sam/PycharmProjects/YakTunerSUPRA/WG.py

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

def _process_and_filter_log_data(log_df, params, logvars, WGlogic, min_pedal=50.0):
    """
    A pure function to prepare and filter log data without UI interactions.
    """
    warnings = []
    processed_log = log_df.copy()

    # --- FIX: Assign X and Y axis variables based on WGlogic ---
    # These columns will be used for binning and plotting later.
    if WGlogic:
        # Custom logic uses RPM vs PUTSP
        processed_log['wg_x_axis_var'] = processed_log['RPM']
        processed_log['wg_y_axis_var'] = processed_log['PUTSP']
    else:
        # Standard logic uses WG_DIS vs PUTSP
        if 'WG_DIS' not in processed_log.columns:
            raise KeyError("Log variable 'WG_DIS' is required for standard WG logic but was not found.")
        processed_log['wg_x_axis_var'] = processed_log['WG_DIS']
        processed_log['wg_y_axis_var'] = processed_log['PUTSP']
    # --- END FIX ---

    # Create derived values for analysis
    processed_log['deltaPUT'] = processed_log['PUT'] - processed_log['PUTSP']
    processed_log['WGNEED_uncorrected'] = processed_log['WG_Final'] - processed_log['deltaPUT'] * params['fudge']
    processed_log['WGNEED'] = processed_log['WGNEED_uncorrected']

    # Filter log data to valid conditions

    if 'BOOST' in logvars:
        processed_log = processed_log[processed_log['BOOST'] >= params['minboost']]
    else:
        warnings.append("Recommend logging 'boost'. Otherwise, logs are not trimmed for min boost.")

    # Filtering Logic for PUT Delta stability
    processed_log['deltaPUT_CHANGE'] = processed_log['deltaPUT'].diff().abs()
    is_small_delta = processed_log['deltaPUT'].abs() < 1
    is_steady_delta = processed_log['deltaPUT_CHANGE'] < 0.5
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

    # --- FIX: Use generic axis variables for binning ---
    log_df['X'] = pd.cut(log_df['wg_x_axis_var'], wgxedges, labels=False)
    log_df['Y'] = pd.cut(log_df['wg_y_axis_var'], wgyedges, labels=False)
    # --- END FIX ---
    return log_df


def create_wg_scatter_plot(log_data, wgxaxis, wgyaxis, WGlogic):
    """
    Creates a Matplotlib scatter plot figure of the filtered log data.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    # --- FIX: Use generic axis variables for plotting ---
    scatter = ax.scatter(
        log_data['wg_x_axis_var'], log_data['wg_y_axis_var'], s=abs(log_data['WGNEED']),
        c=log_data['deltaPUT'], marker='o', cmap='RdBu', label='Log Data'
    )
    # --- END FIX ---
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('PUT - PUT SP (kPa)')
    ax.invert_yaxis()
    # --- FIX: Dynamically set axis labels ---
    if WGlogic:
        ax.set_xlabel('RPM')
        ax.set_ylabel('Boost Setpoint (PUTSP)')
    else:
        ax.set_xlabel('WG Desired Position (%)')
        ax.set_ylabel('Boost Setpoint (PUTSP)')
    # --- END FIX ---
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
    # --- FIX: Use generic axis variables for fitting ---
    points = log_data[['wg_x_axis_var', 'wg_y_axis_var']].values
    # --- END FIX ---
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


def _calculate_final_recommendations(log_data, blend, old_table, wgxaxis, wgyaxis):
    """
    Calculates the final recommended WG table by comparing the new fit with the old
    table and applying confidence intervals to each cell using a predictive model.
    """
    final_table = old_table.copy()

    interp_factor = 0.5
    confidence = 0.7

    for i in range(len(wgxaxis)):
        for j in range(wgyaxis.shape[0]):
            cell_data = log_data[(log_data['X'] == i) & (log_data['Y'] == j)]
            if len(cell_data) > 3:
                mean, std_dev = stats.norm.fit(cell_data['WGNEED'])
                surface_val_factor = blend[j, i]
                current_val_factor = old_table[j, i]
                target_val_factor = (surface_val_factor * interp_factor) + ((mean / 100.0) * (1 - interp_factor))
                low_ci_factor, high_ci_factor = stats.norm.interval(
                    confidence,
                    loc=target_val_factor,
                    scale=(std_dev / 100.0) if std_dev > 0 else 1e-9
                )
                if np.isnan(current_val_factor) or not (low_ci_factor <= current_val_factor <= high_ci_factor):
                    final_table[j, i] = target_val_factor

    final_table = np.round(final_table * 16384) / 16384
    return final_table


# --- Main Orchestrator Function ---
def run_wg_analysis(log_df, wgxaxis, wgyaxis, oldWG, logvars, WGlogic, show_scatter_plot=True):
    """
    Main orchestrator for the WG tuning process. A pure computational function.
    """
    print(" -> Initializing WG analysis...")
    params = {'fudge': 0, 'minboost': 0}

    print(" -> Preparing and filtering log data...")
    processed_log, warnings = _process_and_filter_log_data(
        log_df=log_df, params=params, logvars=logvars, WGlogic=WGlogic
    )

    if processed_log.empty:
        return {'status': 'Failure', 'warnings': warnings, 'scatter_plot_fig': None,
                'results_wg': None}

    print(" -> Creating data bins from WG axes...")
    log = _create_bins_and_labels(processed_log, wgxaxis, wgyaxis)

    scatter_fig = None
    if show_scatter_plot:
        print(" -> Generating raw WG data plot...")
        scatter_fig = create_wg_scatter_plot(log, wgxaxis, wgyaxis, WGlogic)

    print(" -> Fitting 3D surface...")
    blend = _fit_surface(log, wgxaxis, wgyaxis)

    print(" -> Calculating final recommendations...")
    final_table = _calculate_final_recommendations(log, blend, oldWG, wgxaxis, wgyaxis)

    print(" -> Preparing final results as DataFrame...")
    exhlabels = [str(x) for x in wgxaxis]
    intlabels = [str(x) for x in wgyaxis]
    Res = pd.DataFrame(final_table, columns=exhlabels, index=intlabels)

    print(" -> WG analysis complete.")
    return {
        'status': 'Success',
        'warnings': warnings,
        'scatter_plot_fig': scatter_fig,
        'results_wg': Res
    }