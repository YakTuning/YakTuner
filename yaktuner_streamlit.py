import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import re
import traceback
import sys
import difflib
from st_copy_button import st_copy_button
from io import BytesIO
from scipy import interpolate

# --- Add project root to sys.path ---
# This is a robust way to ensure that local modules can be imported.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Import the custom tuning modules ---
from WG import run_wg_analysis
from LPFP import run_lpfp_analysis
from MAF import run_maf_analysis
from MFF import run_mff_analysis
from KNK import run_knk_analysis
from TTA_ATT import run_tta_att_analysis
from tuning_loader import TuningData
from error_reporter import send_to_google_sheets

# --- Constants ---
default_vars = "variables.csv"
MAP_DEFINITIONS_CSV_PATH = "map_definitions.csv"
XDF_MAP_LIST_CSV = 'maps_to_parse.csv'
XDF_SUBFOLDER = "XDFs"
PREDEFINED_FIRMWARES = ['S50', 'A05', 'V30', 'O30', 'LB6']
ALL_FIRMWARES = PREDEFINED_FIRMWARES + ['Other']

# --- Page Configuration ---
st.set_page_config(
    page_title="YAKtuner Online",
    layout="wide"
)

st.title("☁️ YAKtuner Online")

# --- 1. Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Tuner Settings")

    # --- Module Selection ---
    run_wg = st.checkbox("Tune Wastegate (WG)", value=True, key="run_wg", help="Analyzes wastegate duty cycle (WGDC) and boost pressure to recommend adjustments for your base tables. Supports standard and SWG logic.")
    run_maf = st.checkbox("Tune Mass Airflow (MAF)", value=False, key="run_maf", help="Corrects MAF scaling based on short-term and long-term fuel trims during closed-loop operation.")
    run_mff = st.checkbox("Tune Mass Fuel Flow (MFF)", value=False, key="run_mff", help="Adjusts MFF tables based on fuel trims. If MAF tuning is also selected, this runs as a second stage for higher accuracy.")
    run_ign = st.checkbox("Tune Ignition (KNK)", value=False, key="run_ign", help="Detects knock events across all cylinders and recommends ignition timing corrections for a selected base map.")
    run_lpfp = st.checkbox("Tune Low Pressure Pump Duty (LPFP)", value=False, key="run_lpfp", help="Analyzes LPFP duty cycle and pressure to recommend table adjustments.")
    run_tta_att = st.checkbox("TTA/ATT Consistency Check", value=False, key="run_tta_att", help="Verifies that your Torque-to-Air (TTA) and Air-to-Torque (ATT) tables are consistent. It checks for >5% deviation between the expected and actual torque.")

    st.divider()

    # --- Firmware Selection ---
    firmware = st.radio(
        "Firmware Version",
        options=ALL_FIRMWARES,
        horizontal=True,
        help="Select your ECU's firmware. This loads the correct map definitions. If your version isn't listed, choose 'Other' and upload your XDF file.",
        key="firmware"
    )

    st.divider()

    st.subheader("Global Settings")
    oil_temp_unit = st.radio(
        "Oil Temperature Unit in Log File",
        ('F', 'C'),
        index=0,  # Default to Fahrenheit
        horizontal=True,
        help="Select the unit for the 'OILTEMP' column in your log file. "
             "If 'C' is selected, it will be converted to Fahrenheit for analysis."
    )

    st.divider()

    # --- Module-Specific Settings ---
    if run_wg:
        st.subheader("WG Settings")
        use_swg_logic = st.checkbox("Use SWG Logic", key="use_swg_logic", help="Check this if your tune uses the Simplified Wastegate (SWG) logic. This changes which maps are used for the analysis.")

    if run_lpfp:
        st.subheader("LPFP Settings")
        lpfp_drive = st.radio("2WD or 4WD", ('2WD', '4WD'), key="lpfp_drive", help="Select whether the vehicle is 2WD or 4WD to ensure the correct LPFP duty table is loaded from the tune.")


    if run_ign:
        st.subheader("Ignition Settings")
        ign_map_options = {
            "Stock (Base Correction)": 0, "SP Map 1": 1, "SP Map 2": 2, "SP Map 3": 3,
            "SP Map 4": 4, "SP Map 5": 5, "SP Flex Modifier": 6
        }
        selected_map_name = st.selectbox(
            "Ignition Map Selection", options=list(ign_map_options.keys()), key="selected_ign_map", help="Select the base ignition map you want to apply the knock corrections to. The tool will recommend changes for this specific map."
        )
        ign_map = ign_map_options[selected_map_name]
        max_adv = st.slider("Max Advance", 0.0, 2.0, 0.75, 0.25, key="max_adv", help="Set the maximum amount of timing advance to add back per knock event. A lower value is safer.")

    st.divider()
    st.page_link("pages/2_PID_Downloads.py", label="PID Lists for Download", icon="📄")

    st.divider()


    # --- Donation Link ---
    paypal_link = "https://www.paypal.com/donate/?hosted_button_id=MN43RKBR8AT6L"
    st.markdown(f"""
    <style>
        .paypal-button {{
            display: inline-block; padding: 8px 16px; font-size: 14px; font-weight: bold;
            color: #ffffff !important; background-color: #0070ba; border: none; border-radius: 5px;
            text-align: center; text-decoration: none; cursor: pointer; transition: background-color 0.3s;
        }}
        .paypal-button:hover {{
            background-color: #005ea6; color: #ffffff !important; text-decoration: none;
        }}
    </style>
    <div style="text-align: center; margin-top: 20px;">
        <a href="{paypal_link}" target="_blank" class="paypal-button">☕ Support YAKtuner</a>
    </div>
    """, unsafe_allow_html=True)

# --- 2. Main Area for File Uploads ---
st.subheader("1. Upload Tune & Log Files")
uploaded_bin_file = st.file_uploader("Upload .bin file", type=['bin', 'all'], help="Upload your tune file (e.g., my_tune.bin). This contains all the maps the tool will analyze.")
uploaded_log_files = st.file_uploader("Upload .csv log files", type=['csv'], accept_multiple_files=True, help="Upload one or more data logs from your vehicle. The tool will combine them for analysis.")

uploaded_xdf_file = None
if firmware == 'Other':
    st.subheader("2. Upload Configuration File")
    st.info("Since you selected 'Other' firmware, you must provide an XDF file.")
    uploaded_xdf_file = st.file_uploader("Upload .xdf definition file", type=['xdf'], help="Upload the XDF definition file corresponding to your tune. This is required for 'Other' firmware to locate the maps.")


# --- Helper Functions ---

# In C:/Users/Sam/PycharmProjects/YAKtunerCONVERTED/yaktuner_streamlit.py

def display_table_with_copy_button(title: str, styled_df, raw_df: pd.DataFrame):
    """
    Displays a title, a styled DataFrame with its index, and a button to copy
    the raw data (without index/header) to the clipboard.

    Args:
        title (str): The title to display above the table.
        styled_df: The Styler object for display (e.g., with highlighted cells).
        raw_df (pd.DataFrame): The raw, unstyled DataFrame whose values will be copied.
    """
    st.write(title)

    # Prepare the data for the clipboard: tab-separated, no index, no header.
    # This is the format TunerPro expects for clean pasting.
    clipboard_text = raw_df.to_csv(sep='\t', index=False, header=False)

    # Display the table with its index visible for context.
    st.dataframe(styled_df)

    # Use a unique key for the button based on the title to avoid conflicts
    button_label = f"📋 Copy {title.strip('# ')} Data"

    # --- FIX ---
    # Generate a unique key from the title. This is crucial when this function
    # is called inside a loop (e.g., for MAF or MFF tables), as each
    # st_copy_button widget needs a distinct key to avoid a DuplicateKeyError.
    # We create a simple, clean key by removing special characters from the title.
    button_key = f"copy_btn_{re.sub(r'[^a-zA-Z0-9]', '', title)}"

    st_copy_button(clipboard_text, button_label, key=button_key)
    # --- END FIX ---

    st.caption("Use the button above to copy data for pasting into TunerPro.")

def _apply_advanced_state_lam_filter(df):
    """
    Filters the DataFrame to keep rows where state_lam is 1,
    but REMOVES the first 5 rows immediately following each transition to state_lam=1.
    This helps to exclude data from the initial, potentially unstable, closed-loop period.
    """
    if 'state_lam' not in df.columns:
        # If the column doesn't exist, we can't filter. Return the dataframe as-is.
        st.warning("Log variable 'state_lam' not found. Skipping advanced closed-loop filtering.")
        return df

    # 1. Mask for all rows where state_lam is 1. This is our starting set.
    state_lam_is_1_mask = (df['state_lam'] == 1)

    # 2. Identify the start of each state_lam=1 block by checking if the previous value was different.
    is_transition_start = state_lam_is_1_mask & (df['state_lam'].shift(1) != 1)

    # 3. Get the integer indices of these transition points.
    transition_indices = np.where(is_transition_start)[0]

    # 4. Create a mask that is True for the 5 rows to be REMOVED after each transition.
    rows_to_remove_mask = pd.Series(False, index=df.index)
    for idx in transition_indices:
        # Set True for the slice from the transition index to index + 5
        rows_to_remove_mask.iloc[idx:idx + 5] = True

    # 5. The final mask keeps rows where state_lam is 1 AND the row is NOT in the removal mask.
    #    The `~` operator inverts the boolean mask (True becomes False, and vice-versa).
    final_mask = state_lam_is_1_mask & ~rows_to_remove_mask

    st.write(f"Applying advanced closed-loop filter (removing initial 5 rows). Original rows: {len(df)}, Filtered rows: {final_mask.sum()}")

    return df[final_mask]


def normalize_header(header_name):
    """Normalizes a log file header for case-insensitive and unit-agnostic comparison."""
    normalized = re.sub(r'\s*\([^)]*\)\s*$', '', str(header_name))
    return normalized.lower().strip()


def _find_alias_match(aliases, log_headers):
    """Finds an exact match for a list of aliases within the log headers."""
    normalized_aliases = {normalize_header(a) for a in aliases}
    for header in log_headers:
        if normalize_header(header) in normalized_aliases:
            return header
    return None


def _find_best_match(target_name, log_headers, cutoff=0.7):
    """Finds the best fuzzy match for a target name from a list of log headers."""
    normalized_target = normalize_header(target_name)
    normalized_headers = {h: normalize_header(h) for h in log_headers}
    search_space = list(normalized_headers.values())
    best_matches = difflib.get_close_matches(normalized_target, search_space, n=1, cutoff=cutoff)
    if best_matches:
        for original_header, normalized_header in normalized_headers.items():
            if normalized_header == best_matches[0]:
                return original_header
    return None

def map_log_variables_streamlit(log_df, varconv_df):
    """
    Performs a 3-tiered, robust, automatic, and interactive variable mapping.
    The automatic part runs in a status box, while the interactive form is
    rendered in the main script body to avoid cloud-specific rerun conflicts.
    """
    # This block runs once to perform auto-mapping and initialize the state.
    if 'mapping_initialized' not in st.session_state:
        with st.status("Automatically mapping log variables...", expanded=True) as status:
            st.session_state.mapping_initialized = True
            st.session_state.mapping_complete = False
            st.session_state.vars_to_map = []
            st.session_state.varconv_array = varconv_df.to_numpy()
            st.session_state.log_df_mapped = log_df.copy()

            varconv = st.session_state.varconv_array
            available_log_headers = log_df.columns.tolist()
            missing_vars_indices = []
            found_vars_indices = set()

            # --- Pass 1: Prioritize Exact Alias Matches ---
            for i in range(1, varconv.shape[1]):
                aliases_str = str(varconv[0, i])
                canonical_name = varconv[1, i]
                aliases = aliases_str.split(',')
                alias_match = _find_alias_match(aliases, available_log_headers)

                if alias_match:
                    st.session_state.log_df_mapped = st.session_state.log_df_mapped.rename(
                        columns={alias_match: canonical_name}
                    )
                    st.session_state.varconv_array[0, i] = alias_match
                    available_log_headers.remove(alias_match)
                    found_vars_indices.add(i)

            # --- Pass 2: Fuzzy Matching for Remaining Variables ---
            for i in range(1, varconv.shape[1]):
                if i in found_vars_indices:
                    continue

                canonical_name = varconv[1, i]
                friendly_name = varconv[2, i] if varconv.shape[0] > 2 and pd.notna(varconv[2, i]) else canonical_name
                fuzzy_match = _find_best_match(friendly_name, available_log_headers)

                if fuzzy_match:
                    st.session_state.log_df_mapped = st.session_state.log_df_mapped.rename(
                        columns={fuzzy_match: canonical_name}
                    )
                    st.session_state.varconv_array[0, i] = fuzzy_match
                    available_log_headers.remove(fuzzy_match)
                    found_vars_indices.add(i)
                else:
                    missing_vars_indices.append(i)

            st.session_state.updated_varconv_df = pd.DataFrame(st.session_state.varconv_array)

            if not missing_vars_indices:
                st.session_state.mapping_complete = True
                status.update(label="Variable mapping complete.", state="complete", expanded=False)
            else:
                st.session_state.vars_to_map = missing_vars_indices
                status.update(label="Manual input required...", state="complete", expanded=True)

    # This block handles the interactive part, showing one form at a time.
    if st.session_state.get('vars_to_map'): # Use .get for safety
        varconv = st.session_state.varconv_array
        current_var_index = st.session_state.vars_to_map[0]
        prompt_name = varconv[2, current_var_index] if varconv.shape[0] > 2 and pd.notna(
            varconv[2, current_var_index]) else varconv[1, current_var_index]

        st.warning(f"Could not find a match for: **{prompt_name}**")
        st.write(
            f"Please select the corresponding column from your log file. The last known name was `{varconv[0, current_var_index]}`.")

        with st.form(key=f"mapping_form_{current_var_index}"):
            options = ['[Not Logged]'] + log_df.columns.tolist()
            selected_header = st.selectbox("Select Log File Column:", options=options)
            submitted = st.form_submit_button("Confirm Mapping")

            if submitted:
                if selected_header != '[Not Logged]':
                    st.session_state.log_df_mapped = st.session_state.log_df_mapped.rename(
                        columns={selected_header: varconv[1, current_var_index]}
                    )
                    st.session_state.varconv_array[0, current_var_index] = selected_header

                st.session_state.vars_to_map.pop(0)
                if not st.session_state.vars_to_map:
                    st.session_state.mapping_complete = True
                    st.session_state.updated_varconv_df = pd.DataFrame(st.session_state.varconv_array)

                st.rerun()
        return None

    # This block returns the final result only when the entire mapping process is finished.
    if st.session_state.get('mapping_complete'):
        return st.session_state.log_df_mapped

    return None


@st.cache_resource(show_spinner=False)
def load_all_maps_streamlit(bin_content, xdf_content, xdf_name, firmware_setting):
    """Loads all ECU maps from file contents. Accepts bytes to be cache-friendly."""
    st.write("Loading tune data from binary file...")
    try:
        loader = TuningData(bin_content)
    except Exception as e:
        st.error(f"Failed to initialize the binary file loader. Error: {e}")
        return None

    if xdf_content is not None:
        st.write(f"Parsing maps from XDF: {xdf_name}...")
        tmp_xdf_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp_xdf:
                tmp_xdf.write(xdf_content)
                tmp_xdf_path = tmp_xdf.name
            loader.load_from_xdf(tmp_xdf_path, XDF_MAP_LIST_CSV)
        except Exception as e:
            st.error(f"Failed to parse XDF file.")
            st.exception(e)
            return None
        finally:
            if os.path.exists(tmp_xdf_path):
                os.remove(tmp_xdf_path)

    if firmware_setting != 'Other':
        st.write(f"Loading additional maps for '{firmware_setting}' from '{MAP_DEFINITIONS_CSV_PATH}'...")
        try:
            firmware_col = f"address_{firmware_setting}"
            loader.load_from_manual_config(MAP_DEFINITIONS_CSV_PATH, firmware_col)
        except FileNotFoundError:
            st.error(f"'{MAP_DEFINITIONS_CSV_PATH}' not found.")
            return None
        except KeyError:
            st.warning(f"Firmware column for '{firmware_setting}' not found in '{MAP_DEFINITIONS_CSV_PATH}'. Skipping.")
        except Exception as e:
            st.error(f"Failed to load manual map definitions. Error: {e}")
            return None

    all_maps = loader.maps
    if not all_maps:
        st.error("No maps were loaded. Check your XDF and manual configuration files.")
        return None

    st.write(f"Successfully loaded {len(all_maps)} maps into memory.")
    with st.expander("Click to view list of all loaded maps"):
        sorted_map_names = sorted(list(all_maps.keys()))
        num_columns = 3
        columns = st.columns(num_columns)
        maps_per_column = (len(sorted_map_names) + num_columns - 1) // num_columns
        for i in range(num_columns):
            with columns[i]:
                for map_name in sorted_map_names[i * maps_per_column:(i + 1) * maps_per_column]:
                    st.code(map_name, language=None)
    return all_maps


def style_changed_cells(new_df: pd.DataFrame, old_df: pd.DataFrame):
    """Compares two DataFrames and returns a Styler object with changed cells highlighted."""
    try:
        new_df_c = new_df.copy().astype(float)
        old_df_c = old_df.copy().astype(float)
        old_df_aligned, new_df_aligned = old_df_c.align(new_df_c, join='outer', axis=None)
        style_df = pd.DataFrame('', index=new_df.index, columns=new_df.columns)
        increase_style = 'background-color: #2B442B'
        decrease_style = 'background-color: #442B2B'
        style_df[new_df_aligned > old_df_aligned] = increase_style
        style_df[new_df_aligned < old_df_aligned] = decrease_style
        return new_df.style.apply(lambda x: style_df, axis=None)
    except (ValueError, TypeError):
        return new_df.style


def style_deviation_cells(new_df: pd.DataFrame, old_df: pd.DataFrame, threshold=0.05):
    """
    Compares two DataFrames and returns a Styler object with cells highlighted
    if their relative deviation exceeds a threshold.
    """
    try:
        new_df_c = new_df.copy().astype(float)
        old_df_c = old_df.copy().astype(float)

        # To avoid division by zero, we treat a zero in the old table as a special case.
        # Any non-zero new value where the old was zero is a significant deviation.
        # If both are zero, deviation is zero.
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation = np.abs((new_df_c - old_df_c) / old_df_c)

        style_df = pd.DataFrame('', index=new_df.index, columns=new_df.columns)

        # Highlight cells where deviation is > threshold.
        # Also highlight where old was 0 and new is not, but ignore if both are 0.
        highlight_style = 'background-color: #442B2B'  # Red for deviation
        style_df[
            (deviation > threshold) |
            ((old_df_c == 0) & (new_df_c != 0))
            ] = highlight_style

        # Return the new dataframe with the calculated styles applied
        return new_df.style.apply(lambda x: style_df, axis=None).format("{:.2f}")
    except (ValueError, TypeError):
        return new_df.style.format("{:.2f}")

@st.cache_data(show_spinner="Running WG analysis...")
def cached_run_wg_analysis(*args, **kwargs):
    return run_wg_analysis(*args, **kwargs)

@st.cache_data(show_spinner="Running MAF analysis...")
def cached_run_maf_analysis(*args, **kwargs):
    return run_maf_analysis(*args, **kwargs)

@st.cache_data(show_spinner="Running MFF analysis...")
def cached_run_mff_analysis(*args, **kwargs):
    return run_mff_analysis(*args, **kwargs)

@st.cache_data(show_spinner="Running KNK analysis...")
def cached_run_knk_analysis(*args, **kwargs):
    return run_knk_analysis(*args, **kwargs)

@st.cache_data(show_spinner="Running LPFP analysis...")
def cached_run_lpfp_analysis(*args, **kwargs):
    return run_lpfp_analysis(*args, **kwargs)

#@st.cache_data(show_spinner="Running TTA/ATT Consistency Check...")
def cached_run_tta_att_analysis(*args, **kwargs):
    return run_tta_att_analysis(*args, **kwargs)

# --- 3. Run Button and Logic ---
st.divider()

if st.button("🚀 Run YAKtuner Analysis", type="primary", use_container_width=True):
    st.session_state.run_analysis = True
    # Clear previous mapping state on a new run
    for key in ['mapping_initialized', 'mapping_complete', 'vars_to_map', 'varconv_array', 'log_df_mapped']:
        if key in st.session_state:
            del st.session_state[key]

if 'run_analysis' in st.session_state and st.session_state.run_analysis:
    required_files = {"BIN file": uploaded_bin_file, "Log file(s)": uploaded_log_files}
    missing_files = [name for name, file in required_files.items() if not file]

    if missing_files:
        st.error(f"Please upload all required files. Missing: {', '.join(missing_files)}")
        st.session_state.run_analysis = False
    else:
        try:
            wg_results, maf_results, mff_results, knk_results, lpfp_results = None, None, None, None, None
            all_maps_data = {}

            # --- Phase 1: Interactive Variable Mapping (MODIFIED BLOCK) ---
            # The st.status wrapper has been removed from here. The function now handles it internally.
            log_df = pd.concat((pd.read_csv(f, encoding='latin1').iloc[:, :-1] for f in uploaded_log_files),
                               ignore_index=True)

            if 'OILTEMP' in log_df.columns and oil_temp_unit == 'C':
                st.write("Converting Oil Temperature from Celsius to Fahrenheit...")
                log_df['OILTEMP'] = log_df['OILTEMP'] * 1.8 + 32
                st.toast("Oil Temperature converted to Fahrenheit.", icon="🌡️")

            if not os.path.exists(default_vars):
                raise FileNotFoundError(
                    f"Critical file missing: The default '{default_vars}' could not be found.")

            logvars_df = pd.read_csv(default_vars, header=None)
            mapped_log_df = map_log_variables_streamlit(log_df, logvars_df)

            # --- This is the key change: The rest of the script only runs if mapping is complete ---
            if mapped_log_df is not None:

                mapped_log_df = _apply_advanced_state_lam_filter(mapped_log_df).copy()
                # --- Phase 2: Main Analysis Pipeline ---
                with st.status("Starting YAKtuner analysis...", expanded=True) as status:
                    if 'updated_varconv_df' in st.session_state:
                        with st.expander("View Variable Mapping Results"):
                            varconv_array = st.session_state.updated_varconv_df.to_numpy()
                            mapping_summary_df = pd.DataFrame({
                                "Required Variable": varconv_array[2, 1:] if varconv_array.shape[0] > 2 else varconv_array[1, 1:],
                                "Matched Log Column": varconv_array[0, 1:],
                                "Internal App Name": varconv_array[1, 1:]
                            })
                            st.dataframe(mapping_summary_df, use_container_width=True)

                    status.update(label="Loading tune files...")
                    bin_content = uploaded_bin_file.getvalue()
                    xdf_content = None
                    xdf_name = None

                    if firmware in PREDEFINED_FIRMWARES:
                        local_xdf_path = os.path.join(XDF_SUBFOLDER, f"{firmware}.xdf")
                        if os.path.exists(local_xdf_path):
                            with open(local_xdf_path, "rb") as f:
                                xdf_content = f.read()
                            xdf_name = os.path.basename(local_xdf_path)
                        else:
                            raise FileNotFoundError(f"The pre-packaged XDF for {firmware} was not found at '{local_xdf_path}'.")
                    elif firmware == 'Other':
                        if uploaded_xdf_file is None:
                            raise FileNotFoundError("Please upload an XDF file for the 'Other' firmware option.")
                        xdf_content = uploaded_xdf_file.getvalue()
                        xdf_name = uploaded_xdf_file.name

                    all_maps = load_all_maps_streamlit(
                        bin_content=bin_content, xdf_content=xdf_content, xdf_name=xdf_name, firmware_setting=firmware
                    )

                    if all_maps:
                        # Prepare a log copy for MFF, which might be modified by the MAF stage
                        log_for_mff = mapped_log_df.copy()

                        if run_wg:
                            with st.status("Running Wastegate (WG) analysis...", expanded=True) as module_status:
                                try:
                                    if use_swg_logic:
                                        x_axis_key, y_axis_key, temp_comp_key, temp_comp_axis_key = 'swgpid0_X', 'swgpid0_Y', 'tempcomp', 'tempcompaxis'
                                    else:
                                        x_axis_key, y_axis_key, temp_comp_key, temp_comp_axis_key = 'wgpid0_X', 'wgpid0_Y', None, None

                                    essential_keys = [x_axis_key, y_axis_key, 'wgpid0', 'wgpid1']
                                    if use_swg_logic: essential_keys.extend([temp_comp_key, temp_comp_axis_key])

                                    module_maps = {key: all_maps.get(key) for key in essential_keys if key}
                                    missing = [key for key, val in module_maps.items() if val is None]
                                    if missing: raise KeyError(f"A required map is missing: {', '.join(missing)}")
                                    all_maps_data['wg'] = module_maps

                                    wg_results = cached_run_wg_analysis(
                                        log_df=mapped_log_df, wgxaxis=module_maps[x_axis_key], wgyaxis=module_maps[y_axis_key],
                                        oldWG0=module_maps['wgpid0'], oldWG1=module_maps['wgpid1'],
                                        logvars=mapped_log_df.columns.tolist(),
                                        WGlogic=use_swg_logic, tempcomp=module_maps.get(temp_comp_key),
                                        tempcompaxis=module_maps.get(temp_comp_axis_key)
                                    )
                                    if wg_results['status'] == 'Success':
                                        module_status.update(label="Wastegate (WG) analysis complete.", state="complete", expanded=False)
                                    else:
                                        st.error("WG analysis failed. Check warnings and console logs for details.")
                                        module_status.update(label="Wastegate (WG) analysis failed.", state="error", expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during WG tuning: {e}")
                                    module_status.update(label="Wastegate (WG) analysis failed.", state="error", expanded=True)

                        if run_maf:
                            with st.status("Running Mass Airflow (MAF) analysis...",
                                           expanded=True) as module_status:
                                try:
                                    keys = ['maftable0_X', 'maftable0_Y', 'combmodes_MAF'] + [f'maftable{i}'
                                                                                              for i in
                                                                                              range(4)]
                                    module_maps = {key: all_maps.get(key) for key in keys}
                                    missing = [key for key, val in module_maps.items() if val is None]
                                    if missing: raise KeyError(
                                        f"A required map is missing: {', '.join(missing)}")
                                    all_maps_data['maf'] = module_maps

                                    maf_results = cached_run_maf_analysis(
                                        log=mapped_log_df, mafxaxis=module_maps['maftable0_X'],
                                        mafyaxis=module_maps['maftable0_Y'],
                                        maftables=[module_maps[f'maftable{i}'] for i in range(4)],
                                        combmodes_MAF=module_maps['combmodes_MAF'],
                                        logvars=mapped_log_df.columns.tolist()
                                    )

                                    if maf_results['status'] == 'Success':
                                        module_status.update(label="Mass Airflow (MAF) analysis complete.",
                                                             state="complete", expanded=False)

                                        # If MFF is also selected, prepare the log for the second stage.
                                        if run_mff:
                                            module_status.update(
                                                label="Preparing data for MFF second-stage...",
                                                state="running")
                                            # Get the primary new MAF table (IDX0)
                                            new_maf_table_df = maf_results['results_maf']['IDX0']

                                            # --- New ECU-like Interpolation Logic ---
                                            # The ECU performs 2D linear interpolation and clamps to the edge
                                            # of the map if coordinates are out of bounds. We replicate that here.

                                            # 1. Define the axes and values for the interpolator.
                                            y_axis_map = new_maf_table_df.index.astype(float).values
                                            x_axis_rpm = new_maf_table_df.columns.astype(float).values
                                            table_values = new_maf_table_df.values

                                            # 2. Create a RegularGridInterpolator for linear interpolation.
                                            # We set bounds_error=False and fill_value=None, as we will handle
                                            # out-of-bounds values by clamping the inputs manually.
                                            interpolator = interpolate.RegularGridInterpolator(
                                                (y_axis_map, x_axis_rpm),  # Points are (MAP, RPM)
                                                table_values,
                                                method='linear',
                                                bounds_error=False,
                                                fill_value=None
                                            )

                                            # 3. Get the coordinates from the log file to be interpolated.
                                            # --- FIX: Apply the same kPa -> hPa conversion before interpolation ---
                                            # This ensures the MAP values from the log match the scale of the new MAF table's Y-axis.
                                            map_coords_to_interp = log_for_mff['MAP'].values * 10
                                            rpm_coords_to_interp = log_for_mff['RPM'].values
                                            # --- END FIX ---

                                            # 4. Clamp the coordinates to the boundaries of the table axes.
                                            # This replicates the ECU's behavior of using the last row/column
                                            # for any value outside the defined range.
                                            clamped_map = np.clip(map_coords_to_interp, y_axis_map.min(), y_axis_map.max())
                                            clamped_rpm = np.clip(rpm_coords_to_interp, x_axis_rpm.min(), x_axis_rpm.max())

                                            # 5. Create a combined array of points and apply the interpolator.
                                            points_to_interpolate = np.vstack((clamped_map, clamped_rpm)).T
                                            log_for_mff['MAF_COR_NEW'] = interpolator(points_to_interpolate)
                                            # --- End of New Interpolation Logic ---
                                    else:
                                        st.error("MAF analysis failed. Check warnings for details.")
                                        module_status.update(label="Mass Airflow (MAF) analysis failed.",
                                                             state="error", expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during MAF tuning: {e}")
                                    module_status.update(label="Mass Airflow (MAF) analysis failed.",
                                                         state="error", expanded=True)

                        if run_mff:
                            with st.status("Running Fuel Factor (MFF) analysis...",
                                           expanded=True) as module_status:
                                try:
                                    keys = ['MFFtable0_X', 'MFFtable0_Y', 'combmodes_MFF'] + [f'MFFtable{i}'
                                                                                              for i in
                                                                                              range(5)]
                                    module_maps = {key: all_maps.get(key) for key in keys}
                                    missing = [key for key, val in module_maps.items() if val is None]
                                    if missing: raise KeyError(
                                        f"A required map is missing: {', '.join(missing)}")
                                    all_maps_data['mff'] = module_maps

                                    # Determine the tuning mode for MFF based on whether MAF ran successfully
                                    mff_tuning_mode = 'BOTH' if run_maf and maf_results and maf_results.get(
                                        'status') == 'Success' else 'MFF'

                                    mff_results = cached_run_mff_analysis(
                                        log=log_for_mff,  # Use the potentially modified log
                                        mffxaxis=module_maps['MFFtable0_X'],
                                        mffyaxis=module_maps['MFFtable0_Y'],
                                        mfftables=[module_maps[f'MFFtable{i}'] for i in range(5)],
                                        combmodes_MFF=module_maps['combmodes_MFF'],
                                        logvars=mapped_log_df.columns.tolist(),
                                        tuning_mode=mff_tuning_mode  # Pass the mode
                                    )
                                    if mff_results['status'] == 'Success':
                                        module_status.update(label="Fuel Factor (MFF) analysis complete.",
                                                             state="complete", expanded=False)
                                    else:
                                        st.error("MFF analysis failed. Check warnings for details.")
                                        module_status.update(label="Fuel Factor (MFF) analysis failed.",
                                                             state="error", expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during MFF tuning: {e}")
                                    module_status.update(label="Fuel Factor (MFF) analysis failed.",
                                                         state="error", expanded=True)


                        if run_ign:
                            with st.status("Running Ignition (KNK) analysis...", expanded=True) as module_status:
                                try:
                                    keys = ['igxaxis', 'igyaxis'] + [f'igmap{i}' for i in range(6)]
                                    module_maps = {key: all_maps.get(key) for key in keys}
                                    missing = [key for key, val in module_maps.items() if val is None and key in ['igxaxis', 'igyaxis', 'igmap0']]
                                    if missing: raise KeyError(f"A required map for KNK tuning is missing: {', '.join(missing)}")
                                    all_maps_data['knk'] = module_maps

                                    knk_results = cached_run_knk_analysis(
                                        log=mapped_log_df, igxaxis=module_maps['igxaxis'],
                                        igyaxis=module_maps['igyaxis'],
                                        IGNmaps=[module_maps.get(f'igmap{i}') for i in range(6)], max_adv=max_adv,
                                        map_num=ign_map
                                    )
                                    if knk_results['status'] == 'Success':
                                        module_status.update(label="Ignition (KNK) analysis complete.", state="complete", expanded=False)
                                    else:
                                        st.error("KNK analysis failed. Check warnings for details.")
                                        module_status.update(label="Ignition (KNK) analysis failed.", state="error", expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during KNK tuning: {e}")
                                    module_status.update(label="Ignition (KNK) analysis failed.", state="error", expanded=True)

                        if run_lpfp:
                            with st.status("Running Fuel Pump (LPFP) analysis...", expanded=True) as module_status:
                                try:
                                    table_key = 'lpfppwm' if lpfp_drive == '2WD' else 'lpfppwm4wd'
                                    keys = ['lpfppwm_X', 'lpfppwm_Y', table_key]
                                    module_maps = {key: all_maps.get(key) for key in keys}
                                    missing = [key for key, val in module_maps.items() if val is None]
                                    if missing:
                                        raise KeyError(
                                            f"A required map for LPFP tuning is missing: {', '.join(missing)}")
                                    all_maps_data['lpfp'] = module_maps
                                    all_maps_data['lpfp']['table_key'] = table_key

                                    lpfp_results = cached_run_lpfp_analysis(
                                        log=mapped_log_df, xaxis=module_maps['lpfppwm_X'],
                                        yaxis=module_maps['lpfppwm_Y'],
                                        old_table=module_maps[table_key], logvars=mapped_log_df.columns.tolist()
                                    )
                                    if lpfp_results['status'] == 'Success':
                                        module_status.update(label="Fuel Pump (LPFP) analysis complete.",
                                                             state="complete", expanded=False)
                                    else:
                                        st.error("LPFP analysis failed. Check warnings for details.")
                                        module_status.update(label="Fuel Pump (LPFP) analysis failed.",
                                                             state="error", expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during LPFP tuning: {e}")
                                    module_status.update(label="Fuel Pump (LPFP) analysis failed.", state="error",
                                                         expanded=True)

                        if run_tta_att:
                            with st.status("Running TTA/ATT Consistency Check...",
                                           expanded=True) as module_status:
                                try:
                                    tta_att_results = cached_run_tta_att_analysis(all_maps=all_maps)
                                    if tta_att_results['status'] == 'Success':
                                        module_status.update(label="TTA/ATT Check complete.",
                                                             state="complete", expanded=False)
                                    else:
                                        # --- START: New, more detailed error display ---
                                        all_warnings = tta_att_results.get('warnings', [])
                                        debug_logs = tta_att_results.get('debug_logs', [])

                                        for warning in all_warnings:
                                            st.warning(f"TTA/ATT Check Warning: {warning}")

                                        if debug_logs:
                                            with st.expander("Click to view detailed TTA/ATT debug log"):
                                                st.code('\n'.join(debug_logs), language=None)
                                        # --- END: New, more detailed error display ---

                                        module_status.update(label="TTA/ATT Check failed.", state="error",
                                                             expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during TTA/ATT Check: {e}")
                                    module_status.update(label="TTA/ATT Check failed.", state="error",
                                                         expanded=True)

                    status.update(label="Analysis complete!", state="complete", expanded=False)

                # On successful completion of the 'with' block, show balloons
                st.balloons()

                # --- Phase 3: Display All Results ---
                st.header("📈 Analysis Results")

                if wg_results and wg_results.get('status') == 'Success':
                    with st.expander("Wastegate (WG) Tuning Results", expanded=True):
                        if wg_results['warnings']:
                            for warning in wg_results['warnings']: st.warning(f"WG Analysis Warning: {warning}")

                        res_vvl0, res_vvl1 = wg_results['results_vvl0'], wg_results['results_vvl1']
                        scatter_plot, temp_comp = wg_results['scatter_plot_fig'], wg_results['temp_comp_results']
                        module_maps = all_maps_data['wg']
                        x_axis_key = 'swgpid0_X' if use_swg_logic else 'wgpid0_X'
                        y_axis_key = 'swgpid0_Y' if use_swg_logic else 'wgpid0_Y'

                        exh_labels = [str(x) for x in module_maps[x_axis_key]]
                        int_labels = [str(y) for y in module_maps[y_axis_key]]

                        original_vvl0_df = pd.DataFrame(module_maps['wgpid0'], index=int_labels, columns=exh_labels)
                        original_vvl1_df = pd.DataFrame(module_maps['wgpid1'], index=int_labels, columns=exh_labels)
                        styled_vvl0 = style_changed_cells(res_vvl0, original_vvl0_df)
                        styled_vvl1 = style_changed_cells(res_vvl1, original_vvl1_df)

                        tab1, tab2, tab3 = st.tabs(["📈 Recommended Tables", "📊 Scatter Plot", "🌡️ Temp Comp"])
                        with tab1:
                            display_table_with_copy_button("#### Recommended WGPID0 (VVL0)", styled_vvl0, res_vvl0)
                            st.divider()
                            display_table_with_copy_button("#### Recommended WGPID1 (VVL1)", styled_vvl1, res_vvl1)
                        with tab2:
                            if scatter_plot:
                                st.pyplot(scatter_plot)
                            else:
                                st.info("Scatter plot was not generated.")
                        with tab3:
                            if temp_comp is not None:
                                display_table_with_copy_button("#### Recommended Temperature Compensation",
                                                               temp_comp.style, temp_comp)
                            else:
                                st.info("No temperature compensation adjustments were recommended.")

                if maf_results and maf_results.get('status') == 'Success':
                    with st.expander("Mass Airflow (MAF) Tuning Results", expanded=True):
                        if maf_results['warnings']:
                            for warning in maf_results['warnings']: st.warning(f"MAF Analysis Warning: {warning}")
                        recommended_maf_dfs = maf_results['results_maf']
                        module_maps = all_maps_data['maf']
                        tabs = st.tabs([f"📈 MAF Table IDX{i}" for i in range(4)])
                        for i, tab in enumerate(tabs):
                            with tab:
                                original_df = pd.DataFrame(module_maps[f'maftable{i}'],
                                                           index=[str(y) for y in module_maps['maftable0_Y']],
                                                           columns=[str(x) for x in module_maps['maftable0_X']])
                                recommended_df = recommended_maf_dfs[f'IDX{i}']
                                styled_table = style_changed_cells(recommended_df, original_df)
                                display_table_with_copy_button(f"#### Recommended `maftable{i}`", styled_table,
                                                               recommended_df)

                if mff_results and mff_results.get('status') == 'Success':
                    with st.expander("Multiplicative Fuel Factor (MFF) Tuning Results", expanded=True):
                        if mff_results['warnings']:
                            for warning in mff_results['warnings']: st.warning(f"MFF Analysis Warning: {warning}")
                        recommended_mff_dfs = mff_results['results_mff']
                        module_maps = all_maps_data['mff']
                        tabs = st.tabs([f"📈 MFF Table IDX{i}" for i in range(5)])
                        for i, tab in enumerate(tabs):
                            with tab:
                                original_df = pd.DataFrame(module_maps[f'MFFtable{i}'],
                                                           index=[str(y) for y in module_maps['MFFtable0_Y']],
                                                           columns=[str(x) for x in module_maps['MFFtable0_X']])
                                recommended_df = recommended_mff_dfs[f'IDX{i}']
                                styled_table = style_changed_cells(recommended_df, original_df)
                                display_table_with_copy_button(f"#### Recommended `MFFtable{i}`", styled_table,
                                                               recommended_df)

                if knk_results and knk_results.get('status') == 'Success':
                    with st.expander("Ignition Timing (KNK) Tuning Results", expanded=True):
                        if knk_results['warnings']:
                            for warning in knk_results['warnings']: st.warning(f"KNK Analysis Warning: {warning}")
                        recommended_knk_df, scatter_plot, base_map_np = knk_results['results_knk'], knk_results[
                            'scatter_plot_fig'], knk_results['base_map']
                        module_maps = all_maps_data['knk']
                        tab1, tab2 = st.tabs(["📈 Recommended Table", "📊 Knock Scatter Plot"])
                        with tab1:
                            original_df = pd.DataFrame(base_map_np,
                                                       index=[str(y) for y in module_maps['igyaxis']],
                                                       columns=[str(x) for x in module_maps['igxaxis']])
                            styled_table = style_changed_cells(recommended_knk_df, original_df)
                            display_table_with_copy_button(
                                f"#### Recommended Ignition Table (Correcting `{selected_map_name}`)",
                                styled_table, recommended_knk_df)
                        with tab2:
                            if scatter_plot:
                                st.pyplot(scatter_plot)
                            else:
                                st.info("Scatter plot was not generated (no knock events found).")

                if lpfp_results and lpfp_results.get('status') == 'Success':
                    with st.expander("Low-Pressure Fuel Pump (LPFP) Tuning Results", expanded=True):
                        if lpfp_results['warnings']:
                            for warning in lpfp_results['warnings']: st.warning(f"LPFP Analysis Warning: {warning}")
                        recommended_lpfp_df = lpfp_results['results_lpfp']
                        module_maps = all_maps_data['lpfp']
                        table_key = module_maps['table_key']
                        original_lpfp_df = pd.DataFrame(module_maps[table_key],
                                                        index=[str(y) for y in module_maps['lpfppwm_Y']],
                                                        columns=[str(x) for x in module_maps['lpfppwm_X']])
                        styled_lpfp_table = style_changed_cells(recommended_lpfp_df, original_lpfp_df)
                        display_table_with_copy_button(f"#### Recommended {table_key.upper()} Table",
                                                       styled_lpfp_table, recommended_lpfp_df)

                if 'tta_att_results' in locals() and tta_att_results and tta_att_results.get(
                        'status') == 'Success':
                    with st.expander("TTA/ATT Consistency Check Results", expanded=True):
                        if tta_att_results['warnings']:
                            for warning in tta_att_results['warnings']:
                                st.warning(f"TTA/ATT Check Warning: {warning}")

                        st.info(
                            "The table below shows the expected torque values calculated from the TTA table. Cells are highlighted in red if they deviate by more than 5% from your tune's actual ATT table.")

                        result_tabs = st.tabs(sorted(tta_att_results['results'].keys()))
                        for i, tab in enumerate(result_tabs):
                            with tab:
                                tab_name = sorted(tta_att_results['results'].keys())[i]
                                data = tta_att_results['results'][tab_name]
                                original_att_df = data['original_att']
                                recommended_tta_inv_df = data['recommended_tta_inv']

                                styled_table = style_deviation_cells(recommended_tta_inv_df, original_att_df,
                                                                     threshold=0.05)
                                display_table_with_copy_button(f"#### Recommended Inverse TTA for {tab_name}",
                                                               styled_table, recommended_tta_inv_df)

                # --- ADDED ---
                # After a successful run, reset the flag to prevent re-running on the next interaction.
                st.session_state.run_analysis = False

        except Exception as e:
            st.error(f"An unexpected error occurred during the analysis: {e}")
            st.write("You can help improve YAKtuner by sending this error report to the developer.")
            traceback_str = traceback.format_exc()

            with st.form(key="error_report_form"):
                st.write("**An unexpected error occurred.** You can help by sending this report.")
                user_description = st.text_area(
                    "Optional: Please describe what you were doing when the error occurred."
                )
                user_contact = st.text_input(
                    "Optional: Email or username for follow-up questions."
                )
                st.text_area(
                    "Technical Error Details (for submission)",
                    value=traceback_str,
                    height=200,
                    disabled=True
                )
                submit_button = st.form_submit_button("Submit Error Report")

                if submit_button:
                    with st.spinner("Sending report..."):
                        success, message = send_to_google_sheets(traceback_str, user_description, user_contact)
                        if success:
                            st.success("Thank you! Your error report has been sent.")
                        else:
                            st.error(f"Sorry, the report could not be sent. Reason: {message}")
                            st.error("Please copy the details below and report it manually.")

            with st.expander("Click to view technical error details"):
                st.code(traceback_str, language=None)

            # --- ADDED ---
            # After an unsuccessful run, also reset the flag.
            st.session_state.run_analysis = False