import numpy as np
import pandas as pd
import io


def read_maps_from_config(binary_content, config_df, firmware_address_col, overrides=None):
    """
    Reads multiple data maps from in-memory binary content based on a configuration DataFrame.

    Args:
        binary_content (bytes): The raw byte content of the .bin file.
        config_df (pd.DataFrame): DataFrame containing map definitions.
        firmware_address_col (str): The name of the column in config_df that contains
                                    the addresses for the selected firmware.
        overrides (dict, optional): A dictionary to override specific parameters.

    Returns:
        dict: A dictionary where keys are variable names and values are the read numpy arrays.
    """
    if overrides is None:
        overrides = {}

    results = {}

    # Wrap the binary content in a BytesIO stream to make it behave like a file
    bin_stream = io.BytesIO(binary_content)

    for _, row in config_df.iterrows():
        var_name = row['variable_name']

        try:
            params = {
                'address': int(str(row[firmware_address_col]), 16),
                'rows': int(row['rows']),
                'cols': int(row['cols']),
                'offset': float(row['offset']),
                'res': float(row['res']),
                'prec': row['prec']
            }
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not parse parameters for '{var_name}'. Skipping. Error: {e}")
            continue

        if var_name in overrides:
            params.update(overrides[var_name])

        # Read and process the data from the in-memory stream
        bin_stream.seek(params['address'])
        count = params['rows'] * params['cols']

        # Calculate the number of bytes to read based on the data type's size
        dtype = np.dtype(params['prec'])
        byte_count = count * dtype.itemsize

        byte_data = bin_stream.read(byte_count)

        if len(byte_data) != byte_count:
            print(
                f"Warning: Read wrong number of bytes for '{var_name}'. Expected {byte_count}, got {len(byte_data)}. Skipping.")
            continue

        # Use np.frombuffer to interpret the raw bytes as the specified numpy data type
        data = np.frombuffer(byte_data, dtype=dtype)

        data = data.reshape((params['rows'], params['cols']), order='F')
        data = data.astype(float)
        data = (data - params['offset']) / params['res']

        if params['rows'] == 1:
            data = np.ravel(data)

        results[var_name] = data

    return results