import xml.etree.ElementTree as ET
import csv
import os
import pprint  # Used for cleanly printing the output dictionary


def _parse_table_element(variable_name, table_element, base_offset, results_dict):
    """
    Parses a single XDFTABLE XML element and extracts its data.
    """
    if variable_name in results_dict:
        return

    z_axis = table_element.find("XDFAXIS[@id='z']")
    if z_axis is None:
        print(f"Warning: No z-axis found for table with title '{table_element.find('title').text}'")
        return

    embedded_data = z_axis.find('EMBEDDEDDATA')
    if embedded_data is None:
        print(f"Warning: No EMBEDDEDDATA found in z-axis for table '{table_element.find('title').text}'")
        return

    try:
        address = int(embedded_data.get('mmedaddress'), 16) + base_offset

        # Provide a default value of '1' for missing row/col counts
        # This handles 1D tables where one of these attributes may be omitted.
        cols = int(embedded_data.get('mmedcolcount', '1'))
        rows = int(embedded_data.get('mmedrowcount', '1'))

        data_size_bits = int(embedded_data.get('mmedelementsizebits'))
        signed_str = z_axis.get('signed', table_element.get('signed', '0'))
        is_signed = signed_str == '1'
        math_element = z_axis.find('MATH')
        equation = math_element.get('equation') if math_element is not None else 'X'

        results_dict[variable_name] = {
            'address': hex(address),
            'cols': cols,
            'rows': rows,
            'data_size_bits': data_size_bits,
            'signed': is_signed,
            'equation': equation,
            'is_axis': (cols == 1 or rows == 1)
        }
        print(f"  -> Successfully parsed '{variable_name}' from title '{table_element.find('title').text}'.")

    except (TypeError, ValueError) as e:
        print(f"Error parsing attributes for table '{table_element.find('title').text}': {e}")
        return


def parse_xdf_maps(xdf_file_path, map_list_csv_path):
    """
    Parses an XDF file to extract map definitions based on a provided CSV list of titles.
    """
    try:
        tree = ET.parse(xdf_file_path)
        root = tree.getroot()
        print(f"Successfully parsed XDF file: {os.path.basename(xdf_file_path)}")

        base_offset_element = root.find('XDFHEADER/BASEOFFSET')
        if base_offset_element is not None and 'offset' in base_offset_element.attrib:
            base_offset = int(base_offset_element.get('offset'), 16)
            print(f"Found base offset: {hex(base_offset)}")
        else:
            base_offset = 0
            print("Warning: BASEOFFSET not found in XDFHEADER. Defaulting to 0.")

        print("Building title-to-table map for fast lookups...")
        title_to_table_map = {
            table.find('title').text.strip(): table
            for table in root.findall('XDFTABLE')
            if table.find('title') is not None and table.find('title').text
        }
        print(f"Map built with {len(title_to_table_map)} entries.")

        # Implement a robust, multi-encoding read strategy for the CSV
        maps_to_parse = []
        try:
            # First, try the standard 'utf-8-sig' which handles BOM from Excel.
            with open(map_list_csv_path, mode='r', encoding='utf-8-sig') as infile:
                reader = csv.DictReader(infile)
                if 'variable_name' not in reader.fieldnames or 'xdf_title' not in reader.fieldnames:
                    print(f"\n[CRITICAL ERROR] The CSV file '{map_list_csv_path}' is missing required headers.")
                    print("Please ensure the first line of the CSV is exactly: variable_name,xdf_title")
                    return {}
                maps_to_parse = list(reader)
        except UnicodeDecodeError:
            # If UTF-8 fails, it's likely a legacy Windows encoding. Fall back to 'latin-1'.
            print(
                f"Warning: Could not decode '{os.path.basename(map_list_csv_path)}' as UTF-8. Retrying with 'latin-1' encoding.")
            try:
                with open(map_list_csv_path, mode='r', encoding='latin-1') as infile:
                    reader = csv.DictReader(infile)
                    if 'variable_name' not in reader.fieldnames or 'xdf_title' not in reader.fieldnames:
                        print(f"\n[CRITICAL ERROR] The CSV file '{map_list_csv_path}' is missing required headers.")
                        print("Please ensure the first line of the CSV is exactly: variable_name,xdf_title")
                        return {}
                    maps_to_parse = list(reader)
            except Exception as e_fallback:
                print(f"Error reading CSV file '{map_list_csv_path}' with fallback encoding: {e_fallback}")
                return {}
        except FileNotFoundError:
            print(f"Error: Map list CSV file not found at '{map_list_csv_path}'")
            return {}
        except Exception as e:
            print(f"Error reading CSV file '{map_list_csv_path}': {e}")
            return {}

        if not maps_to_parse:
            print(
                f"Error: Failed to read any data from '{os.path.basename(map_list_csv_path)}'. The file might be empty or formatted incorrectly.")
            return {}

        results_dict = {}
        for map_info in maps_to_parse:
            variable_name = map_info.get('variable_name')
            xdf_title = map_info.get('xdf_title')

            if not variable_name or not xdf_title:
                print(f"Warning: Skipping row in CSV with missing data: {map_info}")
                continue

            table_element = title_to_table_map.get(xdf_title)
            if table_element:
                _parse_table_element(
                    variable_name=variable_name,
                    table_element=table_element,
                    base_offset=base_offset,
                    results_dict=results_dict
                )
            else:
                print(f"Warning: Could not find table with title '{xdf_title}' for variable '{variable_name}'")

        print("\n--- Parsing complete. ---")
        return results_dict

    except FileNotFoundError:
        print(f"Error: XDF file not found at '{xdf_file_path}'")
        return {}
    except ET.ParseError as e:
        print(f"Error: Failed to parse XML in '{xdf_file_path}'. Details: {e}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred in parse_xdf_maps: {e}")
        return {}