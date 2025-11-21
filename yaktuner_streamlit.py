import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import re
import traceback
import sys
import difflib
import requests
from st_copy_button import st_copy_button
from io import BytesIO
from scipy import interpolate

# --- Add project root to sys.path ---
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Import the custom tuning modules ---
from WG import run_wg_analysis
from MFF import run_mff_analysis
from KNK import run_knk_analysis
from tuning_loader import TuningData
from error_reporter import send_to_google_sheets

# --- Constants ---
default_vars = "variables.csv"
XDF_MAP_LIST_CSV = 'maps_to_parse.csv'
XDF_SUBFOLDER = "XDFs"
LOG_METADATA_ROWS_TO_SKIP = 4
GITHUB_REPO_API = "https://api.github.com/repos/dmacpro91/BMW-XDFs/contents/B58gen2"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/dmacpro91/BMW-XDFs/master/B58gen2"

# --- Page Configuration ---
st.set_page_config(
    page_title="YAKtuner Online",
    layout="wide"
)

st.title("☁️ YAKtuner Online")

# --- 1. Sidebar Part 1: Initial Settings ---
with st.sidebar:
    # --- FIX: Add the application logo ---
    st.image("yaktune-website-favicon-black.png", use_container_width='auto')
    # --- END FIX ---

    st.header("⚙️ Tuner Settings")

    # --- Module Selection ---
    run_wg = st.checkbox("Tune Wastegate (WG)", value=True, key="run_wg",
                         help="Analyzes wastegate duty cycle (WGDC) and boost pressure to recommend adjustments for your base tables. Supports standard and Custom logic.")
    run_mff = st.checkbox("Tune Mass Fuel Flow (MFF)", value=False, key="run_mff",
                          help="Adjusts MFF tables based on fuel trims.")
    run_ign = st.checkbox("Tune Ignition (KNK)", value=False, key="run_ign",
                          help="Detects knock events across all cylinders and recommends ignition timing corrections for a selected base map.")

    st.divider()

# --- Helper Functions ---

def get_firmware_from_log(log_file):
    """
    Parses the uploaded log file to find the Ecu PRGID.
    """
    try:
        # Read the first few lines to find the metadata
        # We decode as latin1 to be safe, similar to the main read
        content = log_file.read().decode('latin1')
        # Reset pointer
        log_file.seek(0)

        match = re.search(r'#Ecu PRGID:\s*([A-Fa-f0-9]+)', content)
        if match:
             return match.group(1)

        # Fallback: check for Ecu CALID if PRGID isn't there?
        # The user specifically pointed out PRGID in the screenshot, so we stick to that primarily.
        match = re.search(r'#Ecu CALID:\s*([A-Fa-f0-9]+)', content)
        if match:
             return match.group(1)

        return None
    except Exception as e:
        print(f"Error parsing log for firmware: {e}")
        return None

def download_xdf(firmware_id):
    """
    Downloads the XDF for the given firmware ID from GitHub.
    """
    url = f"{GITHUB_RAW_BASE}/{firmware_id}/{firmware_id}.xdf"
    local_path = os.path.join(XDF_SUBFOLDER, f"{firmware_id}.xdf")

    if not os.path.exists(XDF_SUBFOLDER):
        os.makedirs(XDF_SUBFOLDER)

    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            st.error(f"Failed to download XDF from {url} (Status: {response.status_code})")
            return False
    except Exception as e:
        st.error(f"Error downloading XDF: {e}")
        return False

def display_table_with_copy_button(title: str, styled_df, raw_df: pd.DataFrame):
    """
    Displays a title, a styled DataFrame with its index, and a button to copy
    the raw data (without index/header) to the clipboard.
    """
    st.write(title)
    clipboard_text = raw_df.to_csv(sep='\t', index=False, header=False)
    st.dataframe(styled_df)
    button_label = f"📋 Copy {title.strip('# ')} Data"
    button_key = f"copy_btn_{re.sub(r'[^a-zA-Z0-9]', '', title)}"
    st_copy_button(clipboard_text, button_label, key=button_key)
    st.caption("Use the button above to copy data for pasting into TunerPro.")


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
    """
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

            # Pass 1: Prioritize Exact Alias Matches
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

            # Pass 2: Fuzzy Matching for Remaining Variables
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

    if st.session_state.get('vars_to_map'):
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

    all_maps = loader.maps
    if not all_maps:
        st.error("No maps were loaded. Check your XDF and the 'maps_to_parse.csv' file.")
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
        st.warning("Could not apply cell highlighting due to a data type mismatch. Displaying unstyled table.")
        return new_df.style


def style_deviation_cells(new_df: pd.DataFrame, old_df: pd.DataFrame, threshold=0.05):
    """
    Compares two DataFrames and returns a Styler object with cells highlighted
    if their relative deviation exceeds a threshold.
    """
    try:
        new_df_c = new_df.copy().astype(float)
        old_df_c = old_df.copy().astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation = np.abs((new_df_c - old_df_c) / old_df_c)
        style_df = pd.DataFrame('', index=new_df.index, columns=new_df.columns)
        highlight_style = 'background-color: #442B2B'
        style_df[
            (deviation > threshold) |
            ((old_df_c == 0) & (new_df_c != 0))
            ] = highlight_style
        return new_df.style.apply(lambda x: style_df, axis=None).format("{:.2f}")
    except (ValueError, TypeError):
        st.warning("Could not apply cell highlighting due to a data type mismatch. Displaying unstyled table.")
        return new_df.style.format("{:.2f}")


@st.cache_data(show_spinner="Running WG analysis...")
def cached_run_wg_analysis(*args, **kwargs):
    return run_wg_analysis(*args, **kwargs)


@st.cache_data(show_spinner="Running MFF analysis...")
def cached_run_mff_analysis(*args, **kwargs):
    return run_mff_analysis(*args, **kwargs)


@st.cache_data(show_spinner="Running KNK analysis...")
def cached_run_knk_analysis(*args, **kwargs):
    return run_knk_analysis(*args, **kwargs)


# --- 2. Main Area for File Uploads ---
st.subheader("1. Upload Tune & Log Files")
uploaded_bin_file = st.file_uploader("Upload .bin file", type=['bin', 'all'],
                                     help="Upload your tune file (e.g., my_tune.bin). This contains all the maps the tool will analyze.")
uploaded_log_files = st.file_uploader("Upload .csv log files", type=['csv'], accept_multiple_files=True,
                                      help="Upload one or more data logs from your vehicle. The tool will combine them for analysis.")

# --- Firmware Detection Logic ---
# Initialize session state
if 'firmware_id' not in st.session_state:
    st.session_state.firmware_id = None
if 'detected_firmware' not in st.session_state:
    st.session_state.detected_firmware = None
if 'available_firmwares' not in st.session_state:
    st.session_state.available_firmwares = []
if 'xdfs_fetched' not in st.session_state:
     st.session_state.xdfs_fetched = False

# Attempt detection if logs are uploaded
detected_fw = None
if uploaded_log_files:
    for log_file in uploaded_log_files:
        detected_fw = get_firmware_from_log(log_file)
        if detected_fw:
            st.session_state.detected_firmware = detected_fw
            break
else:
    st.session_state.detected_firmware = None # Reset if files are removed

# --- Sidebar Part 2: Firmware Display & Selection ---
with st.sidebar:
    st.subheader("Firmware")

    # Manual Override / Fallback
    manual_mode = False
    if uploaded_log_files:
        manual_mode = st.checkbox("Manually Select Firmware", key="manual_fw_selection")

    active_firmware = None

    if manual_mode:
        # Fetch available firmwares only if we haven't already
        if not st.session_state.xdfs_fetched:
            try:
                with st.spinner("Fetching firmware list from GitHub..."):
                    response = requests.get(GITHUB_REPO_API)
                    if response.status_code == 200:
                        contents = response.json()
                        # Filter for directories only
                        dirs = [item['name'] for item in contents if item['type'] == 'dir']
                        st.session_state.available_firmwares = sorted(dirs)
                        st.session_state.xdfs_fetched = True
                    else:
                         st.error(f"Failed to fetch firmware list: {response.status_code}")
            except Exception as e:
                 st.error(f"Error connecting to GitHub: {e}")

        # Combine local XDFs with GitHub ones (deduplicate)
        local_xdfs = []
        if os.path.exists(XDF_SUBFOLDER):
             local_xdfs = [f.replace('.xdf', '') for f in os.listdir(XDF_SUBFOLDER) if f.endswith('.xdf')]

        all_options = sorted(list(set(local_xdfs + st.session_state.available_firmwares)))

        selected_fw = st.selectbox("Select Firmware", options=all_options, index=0 if all_options else None)
        active_firmware = selected_fw
    else:
        # Automatic Mode
        if st.session_state.detected_firmware:
            active_firmware = st.session_state.detected_firmware
        else:
            active_firmware = None

    # Update Session State
    st.session_state.firmware_id = active_firmware

    # Display Info
    if active_firmware:
        st.info(f"**Active Firmware:**\n`{active_firmware}`")
        # Trigger download check immediately if active
        local_xdf_path = os.path.join(XDF_SUBFOLDER, f"{active_firmware}.xdf")
        if not os.path.exists(local_xdf_path):
            with st.spinner(f"Downloading XDF for {active_firmware}..."):
                    if download_xdf(active_firmware):
                        st.toast(f"Downloaded XDF for {active_firmware}", icon="✅")
                    else:
                        st.error(f"Could not find or download XDF for firmware {active_firmware}.")
    else:
        st.info("**Active Firmware:**\n`Waiting for log...`")

    st.divider()

    # --- Sidebar Part 3: Remaining Settings ---
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

    # --- Module-Specific Settings (Repeated logic or separate?) ---
    # The logic for displaying these settings was already executed in Part 1,
    # but the 'subheader' calls were inside 'if run_wg:' blocks.
    # We need to make sure the layout is correct.
    # Previously:
    #   Sidebar -> Settings -> Firmware -> Global -> Module Specific
    # Now:
    #   Sidebar Part 1 (Settings) -> Main -> Sidebar Part 2 (Firmware) -> Sidebar Part 3 (Global) -> Sidebar Part 4 (Module Specific)

    if run_wg:
        st.subheader("WG Settings")
        use_swg_logic = st.checkbox("Use Custom WGDC Logic", key="use_swg_logic",
                                    help="Check this if your tune uses the Custom WGDC logic. This changes which maps are used for the analysis.")

    if run_ign:
        st.subheader("Ignition Settings")
        max_adv = st.slider("Max Advance", 0.0, 2.0, 0.75, 0.25, key="max_adv",
                            help="Set the maximum amount of timing advance to add if clean logs are observed. A lower value is safer.")

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
            # Use the firmware detected/selected in the sidebar logic
            firmware = st.session_state.firmware_id

            if not firmware:
                 st.error("No firmware selected or detected. Please upload a log file or select firmware manually.")
                 st.session_state.run_analysis = False
                 st.stop()

            # Double check XDF existence (it should have been downloaded in the sidebar logic)
            local_xdf_path = os.path.join(XDF_SUBFOLDER, f"{firmware}.xdf")
            if not os.path.exists(local_xdf_path):
                 # Try one last time, just in case
                 if not download_xdf(firmware):
                     st.error(f"Could not find or download XDF for firmware {firmware}. Cannot proceed.")
                     st.session_state.run_analysis = False
                     st.stop()

            wg_results, mff_results, knk_results = None, None, None
            all_maps_data = {}

            # --- Phase 1: Interactive Variable Mapping ---
            log_df = pd.concat(
                (
                    pd.read_csv(f, encoding='latin1', skiprows=LOG_METADATA_ROWS_TO_SKIP).iloc[:, :-1]
                    for f in uploaded_log_files
                ),
                ignore_index=True
            )

            if 'OILTEMP' in log_df.columns and oil_temp_unit == 'C':
                st.write("Converting Oil Temperature from Celsius to Fahrenheit...")
                log_df['OILTEMP'] = log_df['OILTEMP'] * 1.8 + 32
                st.toast("Oil Temperature converted to Fahrenheit.", icon="🌡️")

            if not os.path.exists(default_vars):
                raise FileNotFoundError(
                    f"Critical file missing: The default '{default_vars}' could not be found.")

            logvars_df = pd.read_csv(default_vars, header=None)
            mapped_log_df = map_log_variables_streamlit(log_df, logvars_df)

            if mapped_log_df is not None:
                # --- Phase 2: Main Analysis Pipeline ---
                with st.status("Starting YAKtuner analysis...", expanded=True) as status:
                    if 'updated_varconv_df' in st.session_state:
                        with st.expander("View Variable Mapping Results"):
                            varconv_array = st.session_state.updated_varconv_df.to_numpy()
                            mapping_summary_df = pd.DataFrame({
                                "Required Variable": varconv_array[2, 1:] if varconv_array.shape[
                                                                                 0] > 2 else varconv_array[1, 1:],
                                "Matched Log Column": varconv_array[0, 1:],
                                "Internal App Name": varconv_array[1, 1:]
                            })
                            st.dataframe(mapping_summary_df, use_container_width=True)

                    status.update(label="Loading tune files...")
                    bin_content = uploaded_bin_file.getvalue()
                    xdf_content = None
                    xdf_name = None

                    # --- Simplified logic to load the single, hardcoded firmware XDF ---
                    local_xdf_path = os.path.join(XDF_SUBFOLDER, f"{firmware}.xdf")
                    if os.path.exists(local_xdf_path):
                        with open(local_xdf_path, "rb") as f:
                            xdf_content = f.read()
                        xdf_name = os.path.basename(local_xdf_path)
                    else:
                        raise FileNotFoundError(f"The required XDF for {firmware} was not found at '{local_xdf_path}'.")

                    all_maps = load_all_maps_streamlit(
                        bin_content=bin_content, xdf_content=xdf_content, xdf_name=xdf_name, firmware_setting=firmware
                    )

                    if all_maps:
                        # Prepare a log copy for MFF
                        log_for_mff = mapped_log_df.copy()

                        if run_wg:
                            with st.status("Running Wastegate (WG) analysis...", expanded=True) as module_status:
                                try:
                                    if use_swg_logic:
                                        x_axis_key = 'wgdc_cust_X'
                                        y_axis_key = 'wgdc_cust_Y'
                                        main_table_key = 'wgdc_cust'
                                    else:
                                        x_axis_key = 'wgdc_X'
                                        y_axis_key = 'wgdc_Y'
                                        main_table_key = 'wgdc'

                                    essential_keys = [x_axis_key, y_axis_key, main_table_key]

                                    module_maps = {key: all_maps.get(key) for key in essential_keys if key}
                                    missing = [key for key, val in module_maps.items() if val is None]
                                    if missing: raise KeyError(
                                        f"A required map for WG tuning is missing: {', '.join(missing)}")
                                    all_maps_data['wg'] = module_maps

                                    wg_results = cached_run_wg_analysis(
                                        log_df=mapped_log_df,
                                        wgxaxis=module_maps[x_axis_key],
                                        wgyaxis=module_maps[y_axis_key],
                                        oldWG=module_maps[main_table_key],
                                        logvars=mapped_log_df.columns.tolist(),
                                        WGlogic=use_swg_logic
                                    )

                                    if wg_results['status'] == 'Success':
                                        module_status.update(label="Wastegate (WG) analysis complete.",
                                                             state="complete", expanded=False)
                                    else:
                                        st.error("WG analysis failed. Check warnings and console logs for details.")
                                        module_status.update(label="Wastegate (WG) analysis failed.", state="error",
                                                             expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during WG tuning: {e}")
                                    module_status.update(label="Wastegate (WG) analysis failed.", state="error",
                                                         expanded=True)

                        if run_mff:
                            with st.status("Running Fuel Factor (MFF) analysis...",
                                           expanded=True) as module_status:
                                try:
                                    keys = ['MFFtable_X', 'MFFtable_Y', 'MFFtable']
                                    module_maps = {key: all_maps.get(key) for key in keys}
                                    missing = [key for key, val in module_maps.items() if val is None]
                                    if missing: raise KeyError(
                                        f"A required map for MFF tuning is missing: {', '.join(missing)}")
                                    all_maps_data['mff'] = module_maps

                                    mff_results = cached_run_mff_analysis(
                                        log=log_for_mff,
                                        mffxaxis=module_maps['MFFtable_X'],
                                        mffyaxis=module_maps['MFFtable_Y'],
                                        mfftable=module_maps['MFFtable'],
                                        logvars=mapped_log_df.columns.tolist()
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
                                    keys = ['igxaxis', 'igyaxis']
                                    module_maps = {key: all_maps.get(key) for key in keys}
                                    missing = [key for key, val in module_maps.items() if val is None]
                                    if missing: raise KeyError(
                                        f"A required map for KNK tuning is missing: {', '.join(missing)}")
                                    all_maps_data['knk'] = module_maps

                                    knk_results = cached_run_knk_analysis(
                                        log=mapped_log_df,
                                        igxaxis=module_maps['igxaxis'],
                                        igyaxis=module_maps['igyaxis'],
                                        max_adv=max_adv
                                    )

                                    if knk_results['status'] == 'Success':
                                        module_status.update(label="Ignition (KNK) analysis complete.",
                                                             state="complete", expanded=False)
                                    else:
                                        st.error("KNK analysis failed. Check warnings for details.")
                                        module_status.update(label="Ignition (KNK) analysis failed.",
                                                             state="error", expanded=True)
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during KNK tuning: {e}")
                                    module_status.update(label="Ignition (KNK) analysis failed.",
                                                         state="error", expanded=True)

                    status.update(label="Analysis complete!", state="complete", expanded=False)

                st.balloons()

                # --- Phase 3: Display All Results ---
                st.header("📈 Analysis Results")

                if wg_results and wg_results.get('status') == 'Success':
                    with st.expander("Wastegate (WG) Tuning Results", expanded=True):
                        if wg_results['warnings']:
                            for warning in wg_results['warnings']:
                                st.warning(f"WG Analysis Warning: {warning}")

                        recommended_wg_df = wg_results['results_wg']
                        scatter_plot = wg_results['scatter_plot_fig']
                        module_maps = all_maps_data['wg']

                        if use_swg_logic:
                            x_axis_key = 'wgdc_cust_X'
                            y_axis_key = 'wgdc_cust_Y'
                            main_table_key = 'wgdc_cust'
                        else:
                            x_axis_key = 'wgdc_X'
                            y_axis_key = 'wgdc_Y'
                            main_table_key = 'wgdc'

                        exh_labels = [str(x) for x in module_maps[x_axis_key]]
                        int_labels = [str(y) for y in module_maps[y_axis_key]]

                        original_wg_df = pd.DataFrame(module_maps[main_table_key], index=int_labels,
                                                      columns=exh_labels)
                        styled_wg_table = style_changed_cells(recommended_wg_df, original_wg_df)

                        tab1, tab2 = st.tabs(["📈 Recommended Table", "📊 Scatter Plot"])
                        with tab1:
                            display_table_with_copy_button("#### Recommended WGDC Base Table", styled_wg_table,
                                                           recommended_wg_df)
                        with tab2:
                            if scatter_plot:
                                st.pyplot(scatter_plot)
                            else:
                                st.info("Scatter plot was not generated.")

                if mff_results and mff_results.get('status') == 'Success':
                    with st.expander("Multiplicative Fuel Factor (MFF) Tuning Results", expanded=True):
                        if mff_results['warnings']:
                            for warning in mff_results['warnings']: st.warning(f"MFF Analysis Warning: {warning}")

                        recommended_mff_df = mff_results['results_mff']
                        module_maps = all_maps_data['mff']

                        original_df = pd.DataFrame(
                            module_maps['MFFtable'],
                            index=[str(y) for y in module_maps['MFFtable_Y']],
                            columns=[str(x) for x in module_maps['MFFtable_X']]
                        )
                        styled_table = style_changed_cells(recommended_mff_df, original_df)
                        display_table_with_copy_button(f"#### Recommended MFF Table", styled_table,
                                                       recommended_mff_df)

                if knk_results and knk_results.get('status') == 'Success':
                    with st.expander("Ignition Timing (KNK) Tuning Results", expanded=True):
                        if knk_results['warnings']:
                            for warning in knk_results['warnings']:
                                st.warning(f"KNK Analysis Warning: {warning}")

                        recommended_knk_df, scatter_plot = knk_results['results_knk'], knk_results[
                            'scatter_plot_fig']
                        module_maps = all_maps_data['knk']
                        tab1, tab2 = st.tabs(["📈 Recommended Correction Table", "📊 Knock Scatter Plot"])

                        with tab1:
                            styled_table = recommended_knk_df.style.format("{:.2f}").background_gradient(
                                cmap='viridis', axis=None
                            )
                            display_table_with_copy_button(
                                "#### Recommended Ignition Correction Table",
                                styled_table, recommended_knk_df
                            )

                        with tab2:
                            if scatter_plot:
                                st.pyplot(scatter_plot)
                            else:
                                st.info("Scatter plot was not generated (no knock events found).")

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

            st.session_state.run_analysis = False