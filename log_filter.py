import pandas as pd
import re
import streamlit as st
from thefuzz import process

COLUMN_FILTER_RULES = {
    'Time': 'Keep',
    'Accel. Lat (m/s2)': 'TC',
    'Accel. Long (m/s2)': 'Keep',
    'Airmass (mg/stk)': 'Keep',
    'Airmass SP (mg/stk)': 'Keep',
    'Ambient Press (kPa)': 'Keep',
    'Ambient Temp (Â°F)': 'Keep',
    'Battery Volts (V)': 'Filter',
    'Boost (psi)': 'Keep',
    'BPA Deviation (%)': 'Keep',
    'BPA Pos (%)': 'Keep',
    'WG Position (%)': 'Keep',
    'BPA SP (%)': 'Keep',
    'Brake Press (bar)': 'Filter',
    'Calc HP (hp)': 'Filter',
    'Calc TQ (lbft)': 'Filter',
    'Cat Temp (Â°F)': 'Keep',
    'Comb Mode ()': 'Keep',
    'Coolant Temp (Â°F)': 'Keep',
    'Cruise ()': 'Filter',
    'Current map ()': 'Keep',
    'DV Position (%)': 'Keep',
    'Eng State ()': 'Keep',
    'Engine Speed (rpm)': 'Keep',
    'EOI Actual (Â°)': 'Keep',
    'EOI Limit (Â°)': 'Keep',
    'Eth Content (%)': 'Keep',
    'Exh Flow Fac ()': 'Keep',
    'Exh Press Des (kPa)': 'Keep',
    'Exhaust Cam (Â°)': 'Keep',
    'FAC_MFF_ADD_FAC_LAM_AD (%)': 'Keep',
    'FAC_MFF_BAS_CUS ()': 'Keep',
    'FAC_TQ_REQ_DRIV_DROF ()': 'Keep',
    'FAC_TQ_REQ_PV ()': 'Keep',
    'Fastest Wheel (km/h)': 'TC',
    'Flex IAT add ()': 'Keep',
    'Flex IAT weight ()': 'Filter',
    'Flex Lambda add ()': 'Keep',
    'Flex Lambda weight ()': 'Filter',
    'Flex MPI add ()': 'Keep',
    'Flex MPI weight ()': 'Filter',
    'Flex PUT Adder (kPa)': 'Keep',
    'Flex PUT weight ()': 'Filter',
    'Flex Spark Adder (Â°)': 'Keep',
    'Flex Spark weight ()': 'Filter',
    'Flex Torque Adder (ft-lb)': 'Keep',
    'Flex Torque weight ()': 'Filter',
    'FP DI (bar)': 'Keep',
    'FP DI SP (bar)': 'Keep',
    'FP MPI (kPa)': 'Keep',
    'FP MPI SP (kPa)': 'Keep',
    'Fuel Flow (g/min)': 'Keep',
    'Fuel Flow SP (mg/stk)': 'Keep',
    'Fuel Split MPI ()': 'Keep',
    'fup_efp_ctl_ad (kPa)': 'Keep',
    'fup_efp_ctl_i (kPa)': 'Keep',
    'fup_efp_dif (kPa)': 'Keep',
    'Gear ()': 'Keep',
    'HPFP Eff Vol (%)': 'Keep',
    'IAT (Â°F)': 'Keep',
    'Ign Table Value (Â°)': 'Keep',
    'Ign Timing Avg (Â°)': 'Keep',
    'Inj Duty DI (%)': 'Keep',
    'Inj Duty MPI (%)': 'Keep',
    'Inj PW DI (ms)': 'Keep',
    'Inj PW MPI (ms)': 'Keep',
    'Int Flow Fac ()': 'Keep',
    'Intake Cam (Â°)': 'Keep',
    'knk nl0 (V)': 'CMD',
    'knk nl1 (V)': 'CMD',
    'knk nl2 (V)': 'CMD',
    'knk nl3 (V)': 'CMD',
    'knks_cmd_gain_ad0 ()': 'CMD',
    'knks_cmd_gain_ad1 ()': 'CMD',
    'knks_cmd_gain_ad2 ()': 'CMD',
    'knks_cmd_gain_ad3 ()': 'CMD',
    'knks_rng_h0 (V)': 'CMD',
    'knks_rng_h1 (V)': 'CMD',
    'knks_rng_h2 (V)': 'CMD',
    'knks_rng_h3 (V)': 'CMD',
    'knks_thd0 (V)': 'CMD',
    'knks_thd1 (V)': 'CMD',
    'knks_thd2 (V)': 'CMD',
    'knks_thd3 (V)': 'CMD',
    'Knock Avg (Â°)': 'Keep',
    'Knock Cyl 1 (Â°)': 'Keep',
    'Knock Cyl 2 (Â°)': 'Keep',
    'Knock Cyl 3 (Â°)': 'Keep',
    'Knock Cyl 4 (Â°)': 'Keep',
    'LACO State ()': 'Keep',
    'Lambda ()': 'Keep',
    'Lambda Delta ()': 'Keep',
    'Lambda PID (%)': 'Keep',
    'Lambda SP ()': 'Keep',
    'Lambda SP Filtered ()': 'Keep',
    'LPFP DC (%)': 'Keep',
    'LTFT (%)': 'Keep',
    'LV_PRS_IM_SP_UP_THD ()': 'Keep',
    'MAF_COR ()': 'Keep',
    'MAP (kPa)': 'Keep',
    'MAP SP (kPa)': 'Keep',
    'MBT Ignition (deg crank)': 'Keep',
    'Misfire Cyl 1 ()': 'Keep',
    'Misfire Cyl 2 ()': 'Keep',
    'Misfire Cyl 3 ()': 'Keep',
    'Misfire Cyl 4 ()': 'Keep',
    'Misfire Sum ()': 'Keep',
    'Muffler Temp (C)': 'Filter',
    'Oil Temp (Â°F)': 'Filter',
    'PCV Mean Voltage (V)': 'Filter',
    'PCV PSI (psi)': 'Filter',
    'Pedal Pos (%)': 'Keep',
    'Port Flap Pos ()': 'Keep',
    'PQ_THR_SP_1 ()': 'Keep',
    'Press Ratio ()': 'Keep',
    'PRS_UP_THR_WIDE_OPEN_THR (hPa)': 'Keep',
    'PUT (kPa)': 'Keep',
    'PUT I Inhibit ()': 'Keep',
    'PUT SP (kPa)': 'Keep',
    'PUT_DIF_PUT_CTL (hPa)': 'Keep',
    'PUT_INC_MMV (hPa)': 'Keep',
    'put_sp_optm_resp (kpa)': 'Keep',
    'Slowest Wheel (km/h)': 'TC',
    'SOI Actual (Â°)': 'Keep',
    'SOI Limit (Â°)': 'Keep',
    'STATE_ALFU ()': 'Keep',
    'STATE_ALFU_CTL ()': 'Keep',
    'Steering Angle (Â°)': 'TC',
    'STFT (%)': 'Keep',
    'Target Slip (km/h)': 'TC',
    'TC Active ()': 'TC',
    'TC D ()': 'TC',
    'TC I ()': 'TC',
    'TC Ignition pull (km/h)': 'TC',
    'TC P ()': 'TC',
    'TC PID ()': 'TC',
    'TC WG pull (%)': 'TC',
    'TC Wheel Slip (km/h)': 'TC',
    'TEG_DYN_UP_TUR (C)': 'Keep',
    'Torque (Nm)': 'Keep',
    'Torque Lim ()': 'Keep',
    'Torque Max (Nm)': 'Keep',
    'Torque Req (Nm)': 'Keep',
    'Total Trim (%)': 'Keep',
    'TPS (Â°)': 'Keep',
    'tqi_add_puc_open_thr[0] (nm)': 'Keep',
    'Turbine Temp (Â°F)': 'Keep',
    'Turbo Speed (krpm)': 'Keep',
    'Valve Lift Pos ()': 'Keep',
    'Vehicle Speed (mph)': 'TC',
    'WG Flow Des (kg/hr)': 'Keep',
    'WG I Value (%)': 'Keep',
    'WG P-D Value (%)': 'Keep',
    'WG Pos Base (%)': 'Keep',
    'WG Pos Final (%)': 'Keep',
    'WG Current (a)': 'Keep',
    'WG Voltage (V)': 'Keep',
    'Wheel Speed FL (mph)': 'TC',
    'Wheel Speed FR (mph)': 'TC',
    'Wheel Speed RL (mph)': 'TC',
    'Wheel Speed RR (mph)': 'TC',
    'FAC_LAM_AD_OUT_FMSP[0] (%)': 'Keep',
    'MFF_ADD_LAM_AD_OUT_FMSP[0] (mg/stk)': 'Keep',
    'FAC_LAM_LIM[1] (%)': 'Keep',
    'MFF_SP_CLC[0] (mg/stk)': 'Keep',
    'SimosTools….': 'Filter',
}

def find_best_column_match(columns, keyword, score_cutoff=80):
    """Finds the best column match for a keyword using fuzzy matching."""
    matches = process.extract(keyword, columns, limit=len(columns))
    best_match = None
    highest_score = 0
    for match, score in matches:
        if score >= score_cutoff and score > highest_score:
            best_match = match
            highest_score = score
    return best_match

def filter_log_data(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Intelligently filters a log DataFrame based on the user's query.
    Applies row filtering first, then applies column filtering based on COLUMN_FILTER_RULES.
    """
    original_rows = len(df)
    original_cols = len(df.columns)

    # --- Row Filtering ---
    # 1. Time-based filtering
    time_match = re.search(r"around time (\d+\.?\d*)", query, re.IGNORECASE)
    if time_match:
        target_time = float(time_match.group(1))
        st.info(f"Detected time-based query. Filtering log data around {target_time}s.")
        time_column = find_best_column_match(df.columns, 'Time')
        if time_column:
            df = df[(df[time_column] >= target_time - 5) & (df[time_column] <= target_time + 5)]

    # 2. WOT (Wide Open Throttle) filtering
    elif "pull" in query.lower() or "wot" in query.lower():
        st.info("Detected 'pull' or 'WOT' in query. Filtering for high throttle conditions.")
        throttle_col = find_best_column_match(df.columns, 'Pedal Pos (%)')
        if not throttle_col:
            throttle_col = find_best_column_match(df.columns, 'TPS (Â°)')
        if throttle_col:
            df = df[df[throttle_col] > 80]

    # --- Column Filtering ---
    cols_to_drop = []
    query_lower = query.lower()
    has_tc_query = "traction control" in query_lower or "tc" in query_lower
    has_cmd_query = any(keyword in query_lower for keyword in ["knock", "gain", "noise", "threshold"])

    for col_name in df.columns:
        match = process.extractOne(col_name, COLUMN_FILTER_RULES.keys())

        if match and match[1] >= 80:
            rule_key = match[0]
            rule = COLUMN_FILTER_RULES[rule_key]

            if rule == 'Filter':
                cols_to_drop.append(col_name)
            elif rule == 'TC' and not has_tc_query:
                cols_to_drop.append(col_name)
            elif rule == 'CMD' and not has_cmd_query:
                cols_to_drop.append(col_name)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        st.info(f"Filtered columns based on rules: {len(cols_to_drop)} columns removed.")

    # --- Final Downsampling ---
    if len(df.index) > 500:
        st.warning(f"Log data is still large ({len(df.index)} rows). Downsampling by taking every other row.")
        df = df.iloc[::2, :]

    st.success(f"Log file filtered: {original_rows} rows -> {len(df.index)} rows | {original_cols} columns -> {len(df.columns)} columns.")
    return df
