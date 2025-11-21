import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import re  # Import the regular expression module

def _format_tta_att_suffix(map_suffix):
    """
    Formats a map suffix like 'H[STND][1][1]' or 'H_LFT_1_1_2' into a
    human-readable name like 'H[VVL0][INT 0][EXH 0]'.
    """
    # This regex is designed to parse strings like 'H[STND][1][1]' or 'H[LFT_1][1][1]'
    # It captures four groups:
    # 1. The prefix (e.g., 'H')
    # 2. The mode (e.g., 'STND' or 'LFT_1')
    # 3. The first index (e.g., '1')
    # 4. The second index (e.g., '1')
    pattern = re.compile(r"(\w+)\[(\w+(?:_\d+)?)\]\[(\d+)\]\[(\d+)\]")
    match = pattern.match(map_suffix)

    if not match:
        # If the input doesn't match the expected bracketed format,
        # it might be an already-correctly-formatted string or an unknown format.
        # In either case, we return it as-is to avoid errors.
        return map_suffix

    try:
        prefix, mode, int_index_str, exh_index_str = match.groups()

        # Decrement the indices as requested
        int_index = int(int_index_str) - 1
        exh_index = int(exh_index_str) - 1

        # Convert the mode to the desired VVL format
        if mode.upper() == 'STND':
            vvl_mode = "VVL0"
        elif mode.upper().startswith('LFT_'):
            # Extracts the '1' from 'LFT_1'
            vvl_num = mode.split('_')[1]
            vvl_mode = f"VVL{vvl_num}"
        else:
            # If the mode is not recognized, return the original to be safe
            return map_suffix

        return f"{prefix}[{vvl_mode}][INT {int_index}][EXH {exh_index}]"

    except (ValueError, IndexError):
        # In case of any unexpected parsing error (e.g., non-integer indices),
        # return the original suffix safely.
        return map_suffix


def run_tta_att_analysis(all_maps):
    """
    Compares the inverse of Torque-to-Airflow (TTA) tables against their
    corresponding Airflow-to-Torque (ATT) tables to check for consistency.
    This version discovers existing tables using simple string matching.
    """
    warnings = []
    results = {}
    processed_suffixes = set()
    debug_logs = []

    # --- Define base names ---
    tta_base = "IP_MAF_STK_SP_VVL_CAM_"
    att_base = "IP_TQI_REF_N_M_AIR_VVL_CAM_"

    debug_logs.append("-> Initializing TTA/ATT Consistency Check...")

    # Iterate through all map names that were actually loaded from the file
    for map_name in all_maps.keys():
        # Check if the current map is a potential TTA base table
        if map_name.startswith(tta_base) and not map_name.endswith(('_X', '_Y')):
            # Extract the unique suffix from the found table name
            map_suffix = map_name[len(tta_base):]

            if map_suffix in processed_suffixes:
                continue

            # --- Build the names of its 5 required partners. ---
            tta_name = map_name
            att_name = att_base + map_suffix
            tta_x_axis_name = tta_name + "_X"
            tta_y_axis_name = tta_name + "_Y"
            att_x_axis_name = att_name + "_X"
            att_y_axis_name = att_name + "_Y"

            required_map_names = [
                tta_name, att_name, tta_x_axis_name, tta_y_axis_name,
                att_x_axis_name, att_y_axis_name
            ]

            # Check which of the required partners are missing.
            missing_maps = [m for m in required_map_names if m not in all_maps]

            if missing_maps:
                # This is a high-quality warning for partially defined sets.
                warnings.append(f"Incomplete set for '{map_suffix}'. Missing: {', '.join(missing_maps)}")
                continue

            # --- If we get here, all 6 maps for this set were found. ---
            processed_suffixes.add(map_suffix)
            tab_name = _format_tta_att_suffix(map_suffix)
            debug_logs.append(f"  -> Found complete set for {tab_name}. Processing...")

            try:
                tta_table = all_maps[tta_name]
                tta_rpm_axis = all_maps[tta_x_axis_name]
                tta_torque_axis = all_maps[tta_y_axis_name]
                att_table = all_maps[att_name]
                att_rpm_axis = all_maps[att_x_axis_name]
                att_airmass_axis = all_maps[att_y_axis_name]

                # --- Perform 2D Interpolation ---
                rpm_mesh, _ = np.meshgrid(tta_rpm_axis, tta_torque_axis)
                source_points = np.array([rpm_mesh.flatten(), tta_table.flatten()]).T
                _, torque_mesh = np.meshgrid(tta_rpm_axis, tta_torque_axis)
                source_values = torque_mesh.flatten()
                att_rpm_mesh, att_airmass_mesh = np.meshgrid(att_rpm_axis, att_airmass_axis)
                query_points = np.array([att_rpm_mesh.flatten(), att_airmass_mesh.flatten()]).T
                interpolated_torques_flat = griddata(source_points, source_values, query_points, method='linear')
                interpolated_tta_inv_table = interpolated_torques_flat.reshape(att_table.shape)
                interpolated_tta_inv_table = np.nan_to_num(interpolated_tta_inv_table, nan=0.0)

                results[tab_name] = {
                    'original_att': pd.DataFrame(att_table, index=att_airmass_axis, columns=att_rpm_axis),
                    'recommended_tta_inv': pd.DataFrame(
                        interpolated_tta_inv_table, index=att_airmass_axis, columns=att_rpm_axis
                    )
                }

            except Exception as e:
                warnings.append(f"Error processing '{tab_name}' after finding all maps: {e}")

    debug_logs.append(
        f" -> TTA/ATT Consistency Check complete. Found and processed {len(results)} complete table pairs.")

    if not results:
        final_warnings = warnings
        if not warnings:
            final_warnings.append(
                "No complete TTA/ATT table pairs were found. Check if maps are named correctly and have all 6 required components (TTA, ATT, and 4 axes)."
            )
        return {'status': 'Failed', 'warnings': final_warnings, 'debug_logs': debug_logs, 'results': {}}

    return {
        'status': 'Success',
        'warnings': warnings,
        'debug_logs': debug_logs,
        'results': results
    }