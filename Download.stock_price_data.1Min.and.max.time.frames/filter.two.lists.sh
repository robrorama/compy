#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <full_list_file> <exceptions_list_file>"
    exit 1
fi

# Assign input parameters to variables
full_list_file=$1
exceptions_list_file=$2
output_file="filtered_list.txt"

# Create the filtered list
grep -Fxv -f "$exceptions_list_file" "$full_list_file" > "$output_file"

# Notify the user
echo "Filtered list created: $output_file"

