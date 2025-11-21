import pandas as pd
import numpy as np
from asteval import Interpreter
import os
import io

from xdf_parser import parse_xdf_maps
from BinRead import read_maps_from_config


def _map_xdf_type_to_numpy(data_size_bits, is_signed, endian='<'):
    """Maps XDF data type info to a NumPy dtype string."""
    if data_size_bits == 8:
        return f"{endian}{'i' if is_signed else 'u'}1"
    elif data_size_bits == 16:
        return f"{endian}{'i' if is_signed else 'u'}2"
    elif data_size_bits == 32:
        return f"{endian}{'i' if is_signed else 'u'}4"
    elif data_size_bits == 64:
        return f"{endian}{'i' if is_signed else 'u'}8"
    else:
        raise ValueError(f"Unsupported data size: {data_size_bits} bits")


def _apply_equation(raw_values, equation_str):
    """Safely applies the conversion equation to the raw data values."""
    if not equation_str or equation_str.strip().upper() == 'X':
        return raw_values

    aeval = Interpreter()
    aeval.eval(f'def convert(X): return {equation_str}')
    return np.vectorize(aeval.symtable['convert'])(raw_values)


def _read_data_from_xdf_definitions(xdf_definitions, binary_content):
    """Reads data from in-memory binary content using definitions parsed from an XDF."""
    processed_maps = {}
    bin_stream = io.BytesIO(binary_content)

    for name, definition in xdf_definitions.items():
        try:
            address = int(definition['address'], 16)
            rows, cols = definition['rows'], definition['cols']
            dtype_str = _map_xdf_type_to_numpy(definition['data_size_bits'], definition['signed'])
            dtype = np.dtype(dtype_str)

            bin_stream.seek(address)
            count = rows * cols
            byte_count = count * dtype.itemsize

            byte_data = bin_stream.read(byte_count)

            if len(byte_data) != byte_count:
                print(f"Warning: For '{name}', expected {byte_count} bytes but only read {len(byte_data)}.")
                continue

            raw_data = np.frombuffer(byte_data, dtype=dtype)
            reshaped_data = raw_data.reshape((rows, cols), order='F')
            physical_data = _apply_equation(reshaped_data, definition['equation'])

            if definition.get('is_axis', False):
                physical_data = physical_data.flatten()

            processed_maps[name] = physical_data
            print(f"  [XDF] -> Successfully read and processed '{name}'.")

        except Exception as e:
            print(f"Error processing XDF map '{name}': {e}")

    return processed_maps


class TuningData:
    """
    Manages loading tuning data from multiple sources (XDF, manual CSV)
    into a unified dictionary of NumPy arrays.
    """

    def __init__(self, binary_content):
        """
        Initializes the loader with the in-memory binary content.
        """
        if not isinstance(binary_content, bytes):
            raise TypeError("TuningData must be initialized with a bytes object.")
        self.binary_content = binary_content
        self.maps = {}
        print(f"TuningData loader initialized with {len(binary_content)} bytes of tune data.")

    def load_from_xdf(self, xdf_file_path, xdf_map_list_csv):
        """
        Parses an XDF to get map definitions, then reads the data from the binary content.
        """
        print(f"\n--- Loading maps from XDF: {os.path.basename(xdf_file_path)} ---")
        if not os.path.exists(xdf_file_path):
            print(f"Warning: XDF file not found at '{xdf_file_path}'. Skipping XDF load.")
            return

        if not os.path.exists(xdf_map_list_csv):
            print(f"Warning: XDF map list CSV not found at '{xdf_map_list_csv}'. Skipping XDF load.")
            return

        xdf_definitions = parse_xdf_maps(xdf_file_path, xdf_map_list_csv)
        if not xdf_definitions:
            print("No map definitions were parsed from the XDF. Nothing to load.")
            return

        xdf_maps_data = _read_data_from_xdf_definitions(xdf_definitions, self.binary_content)
        self.maps.update(xdf_maps_data)
        print(f"--- XDF loading complete. Loaded {len(xdf_maps_data)} maps. ---")

    def load_from_manual_config(self, manual_config_csv_path, firmware_address_col, overrides=None):
        """
        Reads maps defined in a manual CSV configuration file using the in-memory binary content.
        """
        print(f"\n--- Loading maps from manual config: {os.path.basename(manual_config_csv_path)} ---")
        if not os.path.exists(manual_config_csv_path):
            print(f"Warning: Manual config file not found at '{manual_config_csv_path}'. Skipping manual load.")
            return

        try:
            config_df = pd.read_csv(manual_config_csv_path)
        except Exception as e:
            print(f"Error reading manual config CSV '{manual_config_csv_path}': {e}")
            return

        manual_maps_data = read_maps_from_config(
            self.binary_content, config_df, firmware_address_col, overrides
        )

        for name, data in manual_maps_data.items():
            if name in self.maps:
                print(f"Warning: Map '{name}' from manual config is overwriting a previously loaded map.")
            else:
                print(f"  [Manual] -> Successfully read and processed '{name}'.")
            self.maps[name] = data

        print(f"--- Manual loading complete. Loaded {len(manual_maps_data)} maps. ---")