import pandas as pd

# --- Configuration ---
# Use a raw string (r"...") to handle backslashes in Windows file paths.
file_path = r"C:\Users\obs\OneDrive\ETH\ETH_MSc\Masters Thesis\CIPS_variations\BW vars\20250610_0004544\20250610_0004569\Visit_projector_1_data.pkl"

# Define the rows you want to print.
# Replace these with the actual row indices or labels you need.

# Option 1: By integer position (e.g., 0th, 2nd, 5th row)
rows_by_position = [78,79,80, 99,100,101]

# Option 2: By index label (if your DataFrame has named indices)
# For example, if your index is a series of dates or custom strings:
# rows_by_label = ['2023-01-01', '2023-01-15', '2023-02-01']
# If the index is just the default integers, labels and positions are the same.
rows_by_label = [10, 20, 30]


# --- Script ---
try:
    # Load the data from the .pkl file into a pandas DataFrame
    print(f"🔄 Loading data from: {file_path}\n")
    df = pd.read_pickle(file_path)

    print("✅ Data loaded successfully. Here's a preview of the first 5 rows:")
    print(df.head())
    print("-" * 50)

    # --- Select and Print Rows ---

    # Using .iloc for selection by integer position
    if rows_by_position:
        print("\nPrinting rows by their integer position using .iloc:")
        selected_rows_pos = df.iloc[rows_by_position]
        print(selected_rows_pos)
        print("-" * 50)


    # Using .loc for selection by index label
    if rows_by_label:
        # Check if all requested labels exist in the DataFrame's index
        valid_labels = [label for label in rows_by_label if label in df.index]
        if not valid_labels:
             print("\n⚠️ None of the specified labels in 'rows_by_label' exist in the DataFrame's index.")
        else:
            print("\nPrinting rows by their index label using .loc:")
            selected_rows_lab = df.loc[valid_labels]
            print(selected_rows_lab)
            print("-" * 50)


except FileNotFoundError:
    print(f"❌ ERROR: The file was not found at the specified path:\n{file_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")