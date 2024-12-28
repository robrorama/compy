import sys
import pandas as pd

# Check if a column name has been provided
if len(sys.argv) < 2:
    print("Usage: python3 script_name.py <column_name>")
    sys.exit(1)

# Get the column name to sort by from the command line argument
sort_column = sys.argv[1]
file_path = sys.argv[2]
output_file_path = sys.argv[3]

# Load the dataset
file_path = 'your_input_file.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Function to sort the DataFrame based on a specified column
def sort_by_column(df, column_name):
    if column_name in df.columns:
        # Handle non-numeric columns by converting to numeric if needed
        df[column_name] = pd.to_numeric(df[column_name].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce')
        df_sorted = df.sort_values(by=column_name, ascending=False)
        return df_sorted
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")
        sys.exit(1)

# Sort the DataFrame based on the specified column
df_sorted = sort_by_column(df, sort_column)

# Save the sorted DataFrame to a new CSV file
#output_file_path = 'sorted_stock_data.csv'
df_sorted.to_csv(output_file_path, index=False)

print(f"Sorted data by '{sort_column}' saved to {output_file_path}")

