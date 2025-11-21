"""
Knock (KNK) Analysis and Ignition Timing Correction Module

This module contains pure, non-UI functions to analyze engine logs for knock
events to recommend ignition timing corrections.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Set a non-interactive backend for matplotlib
matplotlib.use('Agg')

# --- Helper Functions ---

def _prepare_knock_data(log):
    """Creates derived columns and identifies knock events in the log data."""
    log['MAP'] = log['MAP'] * 10

    # --- Identify single cylinder knock outliers for statistical analysis ---
    all_cyl_knock = log[['KNK1', 'KNK2', 'KNK3', 'KNK4']].to_numpy()
    log['KNKAVG'] = np.mean(all_cyl_knock, axis=1)

    min_cyl_knock = np.min(all_cyl_knock, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        z_scores = stats.zscore(all_cyl_knock, axis=1)
        outlier_scores = z_scores * all_cyl_knock / min_cyl_knock[:, np.newaxis]

    outlier_scores = np.nan_to_num(outlier_scores)
    rows_with_outlier = np.where(np.any(outlier_scores > 0, axis=1))[0]

    log['singlecyl'] = 0
    if len(rows_with_outlier) > 0:
        outlier_cyl_indices = np.argmax(outlier_scores[rows_with_outlier], axis=1)
        log.loc[rows_with_outlier, 'singlecyl'] = outlier_cyl_indices + 1

    # --- Vectorized knock event detection ---
    knock_cols = ['KNK1', 'KNK2', 'KNK3', 'KNK4']
    knock_decreased = log[knock_cols].diff() < 0
    log['knkoccurred'] = knock_decreased.any(axis=1)

    # --- Identify source of knock for plotting ---
    knock_events_mask = log['knkoccurred']
    num_knocking_cyls = knock_decreased[knock_events_mask].sum(axis=1)
    single_knock_cyl_idx = np.argmax(knock_decreased[knock_events_mask].to_numpy(), axis=1)

    log['knock_source_cyl'] = np.nan
    log.loc[knock_events_mask & (num_knocking_cyls == 1), 'knock_source_cyl'] = single_knock_cyl_idx[num_knocking_cyls == 1] + 1
    log.loc[knock_events_mask & (num_knocking_cyls > 1), 'knock_source_cyl'] = 0

    return log

def _create_bins(log, igxaxis, igyaxis):
    """Discretizes log data into bins based on ignition map axes."""
    xedges = [0] + [(igxaxis[i] + igxaxis[i + 1]) / 2 for i in range(len(igxaxis) - 1)] + [float('inf')]
    yedges = [0] + [(igyaxis[i] + igyaxis[i + 1]) / 2 for i in range(len(igyaxis) - 1)] + [float('inf')]

    log['X'] = pd.cut(log['RPM'], bins=xedges, labels=False)
    log['Y'] = pd.cut(log['MAF'], bins=yedges, labels=False)
    return log

def _calculate_knock_correction(log, igxaxis, igyaxis, params):
    """Calculates the recommended ignition correction map based on knock data."""
    num_x, num_y = len(igxaxis), len(igyaxis)
    correction_map = np.zeros((num_y, num_x))
    max_count_for_full_advance = 100

    for i in range(num_x):
        for j in range(num_y):
            cell_data = log[(log['X'] == i) & (log['Y'] == j)]
            count = len(cell_data)

            if count > 3:
                knock_events = cell_data[cell_data['knkoccurred']]
                mean_cell_knock = cell_data['KNKAVG'].mean()

                if not knock_events.empty:
                    mean_knock_retard_during_events = knock_events['KNKAVG'].mean()
                    std_dev_cell_knock = cell_data['KNKAVG'].std()
                    _low_ci, high_ci = norm.interval(
                        params['confidence'],
                        loc=mean_cell_knock,
                        scale=std_dev_cell_knock if std_dev_cell_knock > 0 else 1e-9
                    )
                    if high_ci < 0:
                        correction_map[j, i] = (high_ci + mean_knock_retard_during_events) / 2
                elif knock_events.empty and igxaxis[i] > 2500 and igyaxis[j] > 700:
                    confidence_weight = min(count, max_count_for_full_advance) / max_count_for_full_advance
                    advance_amount = params['max_adv'] * confidence_weight
                    correction_map[j, i] = mean_cell_knock + advance_amount

    correction_map = np.nan_to_num(correction_map)
    correction_map = np.minimum(correction_map, params['max_adv'])

    intermediate = np.ceil(correction_map * (16 / 3)) / (16 / 3)
    final_correction = np.round(intermediate * (8 / 3)) / (8 / 3)
    return final_correction

def create_knock_scatter_plot(log, igxaxis, igyaxis):
    """Creates a Matplotlib scatter plot figure of the knock events."""
    knock_events = log[log['knkoccurred']].copy()
    if knock_events.empty:
        print("No knock events to plot.")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['grey', 'red', 'blue', 'green', 'purple']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    scatter = ax.scatter(
        knock_events['RPM'], knock_events['MAF'],
        s=abs(knock_events['KNKAVG']) * 100,
        c=knock_events['knock_source_cyl'],
        cmap=cmap,
        norm=norm
    )

    cbar = fig.colorbar(scatter, ax=ax, label='Knock Source')
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['Multiple', 'Cyl 1', 'Cyl 2', 'Cyl 3', 'Cyl 4'])

    ax.invert_yaxis()
    ax.set_xlabel('RPM')
    ax.set_ylabel('MAF')
    ax.set_title('Knock Events by Cylinder and Magnitude')
    ax.grid(True)
    ax.set_xticks(igxaxis)
    ax.set_xticklabels(labels=igxaxis, rotation=45)
    ax.set_yticks(igyaxis)
    fig.tight_layout()
    return fig

# --- Main Orchestrator Function ---
def run_knk_analysis(log, igxaxis, igyaxis, IGNmaps, max_adv, map_num):
    """
    Main orchestrator for the KNK tuning process. A pure computational function.

    Args:
        log (pd.DataFrame): The mapped log data.
        igxaxis, igyaxis (np.ndarray): The axes for the ignition tables.
        IGNmaps (list[np.ndarray]): A list of the original ignition tables.
        max_adv (float): The maximum advance to apply, from user settings.
        map_num (int): The index of the base map to correct.

    Returns:
        dict: A dictionary containing all results.
    """
    print(" -> Initializing KNK analysis...")
    params = {'max_adv': max_adv, 'confidence': 0.7}
    warnings = []

    if map_num == 0:
        base_ignition_map = np.zeros((len(igyaxis), len(igxaxis)))
    elif map_num - 1 < len(IGNmaps):
        base_ignition_map = IGNmaps[map_num - 1]
    else:
        warnings.append(f"Selected map index {map_num} is out of range. Using a zeroed base map.")
        base_ignition_map = np.zeros((len(igyaxis), len(igxaxis)))

    print(" -> Preparing knock data from logs...")
    log = _prepare_knock_data(log)

    print(" -> Creating data bins from ignition axes...")
    log = _create_bins(log, igxaxis, igyaxis)

    print(" -> Generating knock events scatter plot...")
    scatter_fig = create_knock_scatter_plot(log, igxaxis, igyaxis)
    if scatter_fig is None:
        warnings.append("No knock events were found in the log to generate a scatter plot.")

    print(" -> Calculating ignition correction map...")
    correction_map = _calculate_knock_correction(log, igxaxis, igyaxis, params)

    recommended_map = base_ignition_map + correction_map

    print(" -> Preparing final results as DataFrame...")
    xlabels = [str(x) for x in igxaxis]
    ylabels = [str(y) for y in igyaxis]
    result_df = pd.DataFrame(recommended_map, columns=xlabels, index=ylabels)

    print(" -> KNK analysis complete.")
    return {
        'status': 'Success',
        'warnings': warnings,
        'results_knk': result_df,
        'scatter_plot_fig': scatter_fig,
        'base_map': base_ignition_map
    }