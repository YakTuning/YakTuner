import streamlit as st
import os

# --- Page Configuration ---
st.set_page_config(page_title="Resource Downloads", layout="wide")
st.title("📄 Resource Downloads")
st.markdown("Here you can download predefined XDF and PID lists for use with your tuning software.")

# --- Constants for Directories ---
PID_LISTS_DIR = "PID Lists"
FULL_LISTS_SUBDIR = "Full PID Lists"
ADDITIONS_SUBDIR = "Additions"
XDF_DIR = "XDFs"

def create_download_section(title, directory_path, file_extension, mime_type):
    """
    A generic helper function to find files in a directory and display
    download buttons for them in a structured, collapsible layout.

    Args:
        title (str): The subheader title for the section.
        directory_path (str): The path to the folder containing the files.
        file_extension (str): The file extension to look for (e.g., '.csv').
        mime_type (str): The MIME type for the download button (e.g., 'text/csv').
    """
    # Use an expander, which is collapsible and defaults to collapsed.
    with st.expander(title, expanded=False):
        if not os.path.isdir(directory_path):
            st.error(f"Directory not found: '{directory_path}'. Please check your folder structure.")
            return

        # Find all files with the specified extension
        files_to_list = sorted([f for f in os.listdir(directory_path) if f.endswith(file_extension)])

        if not files_to_list:
            st.info(f"No {file_extension} files found in this category.")
            return

        # Create a clean layout for the download links
        for file_name in files_to_list:
            file_path = os.path.join(directory_path, file_name)
            try:
                with open(file_path, "rb") as f:
                    # Reading as bytes is the most robust way for st.download_button
                    file_data = f.read()

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{file_name}**")
                with col2:
                    st.download_button(
                        label="📥 Download",
                        data=file_data,
                        file_name=file_name,
                        mime=mime_type,
                        use_container_width=True,
                        key=f"download_{file_name}"
                    )
                st.divider()
            except FileNotFoundError:
                st.warning(f"Could not read file: {file_name}")
            except Exception as e:
                st.error(f"An error occurred while processing {file_name}: {e}")

# --- Main Page Logic ---
# Create the download sections for each category by calling the generic helper.
# The order has been adjusted for better user experience.

# --- XDF Section ---
create_download_section(
    title="XDF Definition Files",
    directory_path=XDF_DIR,
    file_extension=".xdf",
    mime_type="application/xml"  # XDF files are XML-based
)

# --- PID Sections ---
full_list_path = os.path.join(PID_LISTS_DIR, FULL_LISTS_SUBDIR)
additions_path = os.path.join(PID_LISTS_DIR, ADDITIONS_SUBDIR)

create_download_section(
    title="Full PID Lists",
    directory_path=full_list_path,
    file_extension=".csv",
    mime_type="text/csv"
)
create_download_section(
    title="PID List Additions",
    directory_path=additions_path,
    file_extension=".csv",
    mime_type="text/csv"
)