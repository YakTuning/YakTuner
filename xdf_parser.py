import xml.etree.ElementTree as ET
import csv
import re
import pprint  # Used for cleanly printing the output dictionary


def _parse_single_table(variable_name, description, description_to_table_map, base_offset, results_dict):
    """
    Parses a single XDFTABLE, extracts its data, and recursively parses its axes.

    This is a helper function that finds a table by its description, extracts
    all relevant parameters, and then inspects the description text to find
    and parse linked axis tables.

    Args:
        variable_name (str): The internal name to assign to the parsed map.
        description (str): The description text to search for in the XDF.
        description_to_table_map (dict): A pre-built mapping of all table
                                         descriptions to their XML elements for fast lookups.
        base_offset (int): The base memory offset from the XDF header.
        results_dict (dict): The main dictionary to store all parsed results.
    """
    # Avoid re-parsing if we've already processed this map
    if variable_name in results_dict:
        return

    table_element = description_to_table_map.get(description)
    if table_element is None:
        print(f"Warning: Could not find table with description '{description}' for variable '{variable_name}'")
        return

    # --- Extract Z-Axis (Value) Data ---
    z_axis = table_element.find("XDFAXIS[@id='z']")
    if z_axis is None:
        print(f"Warning: No z-axis found for table '{description}'")
        return

    embedded_data = z_axis.find('EMBEDDEDDATA')
    if embedded_data is None:
        print(f"Warning: No EMBEDDEDDATA found in z-axis for table '{description}'")
        return

    try:
        # Addresses in XDF are hex strings and relative to the base offset
        address = int(embedded_data.get('mmedaddress'), 16) + base_offset
        cols = int(embedded_data.get('mmedcolcount'))
        rows = int(embedded_data.get('mmedrowcount'))
        data_size_bits = int(embedded_data.get('mmedelementsizebits'))

        # Determine if the value is signed. Check axis, then table, then default to '0' (unsigned)
        signed_str = z_axis.get('signed', table_element.get('signed', '0'))
        is_signed = signed_str == '1'

        # Get the conversion equation
        math_element = z_axis.find('MATH')
        equation = math_element.get('equation') if math_element is not None else 'X'

        # Store the extracted data
        results_dict[variable_name] = {
            'address': hex(address),
            'cols': cols,
            'rows': rows,
            'data_size_bits': data_size_bits,
            'signed': is_signed,
            'equation': equation,
            'is_axis': (cols == 1 or rows == 1)  # Heuristic: 1D tables are axes
        }

    except (TypeError, ValueError) as e:
        print(f"Error parsing attributes for table '{description}': {e}")
        return

    # --- Recursively Parse Description for Axis Maps (X and Y) ---
    desc_element = table_element.find('description')
    full_description_text = desc_element.text if desc_element is not None else ''
    if not full_description_text:
        return

    # Process each line of the description to find axis definitions
    for line in full_description_text.splitlines():
        line = line.strip()
        # Regex to find "X: title" or "Y: title" patterns at the start of a line
        match = re.match(r'^([XY]):\s*(.*)', line)
        if match:
            axis_letter, axis_description = match.groups()
            axis_description = axis_description.strip()
            axis_variable_name = f"{variable_name}_{axis_letter}"

            print(
                f"  -> Found linked axis '{axis_letter}' for '{variable_name}': '{axis_description}'. Parsing as '{axis_variable_name}'...")

            # Recursively parse the found axis table
            _parse_single_table(
                variable_name=axis_variable_name,
                description=axis_description,
                description_to_table_map=description_to_table_map,
                base_offset=base_offset,
                results_dict=results_dict
            )


def parse_xdf_maps(xdf_file_path, map_list_csv_path):
    """
    Parses an XDF file to extract map definitions based on a provided CSV list.

    This function reads a CSV file containing variable names and their corresponding
    descriptions in the XDF. It then finds each map in the XDF, extracts key
    parameters (address, dimensions, data type, equation), and also parses
    linked axis tables.

    Args:
        xdf_file_path (str): The path to the .xdf file.
        map_list_csv_path (str): The path to the CSV file which maps
                                 internal variable names to XDF descriptions.
                                 Expected columns: 'variable_name', 'xdf_description'.

    Returns:
        dict: A dictionary where keys are the variable names (including derived
              axis names like 'map_X') and values are dictionaries containing
              the parsed parameters. Returns an empty dictionary if parsing fails.
    """
    try:
        # --- 1. Parse the XDF file and get the root ---
        tree = ET.parse(xdf_file_path)
        root = tree.getroot()
        print(f"Successfully parsed XDF file: {xdf_file_path}")

        # --- 2. Extract the base offset from the header ---
        base_offset_element = root.find('XDFHEADER/BASEOFFSET')
        if base_offset_element is not None and 'offset' in base_offset_element.attrib:
            base_offset = int(base_offset_element.get('offset'), 16)
            print(f"Found base offset: {hex(base_offset)}")
        else:
            base_offset = 0
            print("Warning: BASEOFFSET not found in XDFHEADER. Defaulting to 0.")

        # --- 3. Build a fast lookup map from description to table element ---
        print("Building description-to-table map for fast lookups...")
        description_to_table_map = {}
        for table in root.findall('XDFTABLE'):
            desc_element = table.find('description')
            # Ensure the description element and its text exist and are not empty
            if desc_element is not None and desc_element.text and desc_element.text.strip():
                # The "key" for a table is the first line of its description.
                # Subsequent lines often contain axis definitions.
                first_line = desc_element.text.strip().splitlines()[0].strip()
                description_to_table_map[first_line] = table
        print(f"Map built with {len(description_to_table_map)} entries.")

        # --- 4. Read the target maps from the CSV file ---
        try:
            with open(map_list_csv_path, mode='r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                maps_to_parse = list(reader)
        except FileNotFoundError:
            print(f"Error: Map list CSV file not found at '{map_list_csv_path}'")
            return {}
        except Exception as e:
            print(f"Error reading CSV file '{map_list_csv_path}': {e}")
            return {}

        # --- 5. Iterate through the target maps and parse each one ---
        results_dict = {}
        for map_info in maps_to_parse:
            variable_name = map_info.get('variable_name')
            xdf_description = map_info.get('xdf_description')

            if not variable_name or not xdf_description:
                print(f"Warning: Skipping invalid row in CSV: {map_info}")
                continue

            print(f"\nParsing map '{variable_name}' with description '{xdf_description}'...")
            _parse_single_table(
                variable_name=variable_name,
                description=xdf_description,
                description_to_table_map=description_to_table_map,
                base_offset=base_offset,
                results_dict=results_dict
            )

        print("\n--- Parsing complete. ---")
        return results_dict

    except FileNotFoundError:
        print(f"Error: XDF file not found at '{xdf_file_path}'")
        return {}
    except ET.ParseError as e:
        print(f"Error: Failed to parse XML in '{xdf_file_path}'. Details: {e}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}

