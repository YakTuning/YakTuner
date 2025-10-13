import pandas as pd
import re
import streamlit as st
from thefuzz import process

def find_best_column_match(columns, keyword, score_cutoff=80):
    """Finds the best column match for a keyword using fuzzy matching."""
    best_match, score = process.extractOne(keyword, columns)
    if score >= score_cutoff:
        return best_match
    return None

def filter_log_data(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Intelligently filters a log DataFrame based on the user's query.
    """
    original_rows = len(df)

    # 1. Time-based filtering
    time_match = re.search(r"around time (\d+\.?\d*)", query, re.IGNORECASE)
    if time_match:
        target_time = float(time_match.group(1))
        st.info(f"Detected time-based query. Filtering log data around {target_time}s.")
        time_column = find_best_column_match(df.columns, 'time')
        if time_column:
            df = df[(df[time_column] >= target_time - 5) & (df[time_column] <= target_time + 5)]

    # 2. WOT (Wide Open Throttle) filtering
    elif "pull" in query.lower() or "wot" in query.lower():
        st.info("Detected 'pull' or 'WOT' in query. Filtering for high throttle conditions.")
        throttle_col = find_best_column_match(df.columns, 'throttle') or find_best_column_match(df.columns, 'pedal position')
        if throttle_col:
            df = df[df[throttle_col] > 80]

    # 3. Column removal
    cols_to_remove = [col for col in df.columns if find_best_column_match([col], 'wheel speed')]
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)

    # 4. Row downsampling if still too large
    if len(df) > 500: # Arbitrary threshold for "too large"
        st.warning(f"Log data is still large ({len(df)} rows). Downsampling by taking every other row.")
        df = df.iloc[::2, :]

    st.success(f"Log file filtered: {original_rows} rows -> {len(df)} rows.")
    return df
