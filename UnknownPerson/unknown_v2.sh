#!/bin/bash

# List of notebooks to be converted
notebooks=(
    "unknown_baseline_p21.ipynb"
    "unknown_baseline_p22.ipynb"
    "unknown_baseline_p25.ipynb"
    "unknown_baseline_p26.ipynb"
    "unknown_baseline_p27.ipynb"
    "unknown_baseline_p28.ipynb"
    "unknown_baseline_p29.ipynb"
)

# Loop through the notebooks and convert them
for notebook in "${notebooks[@]}"; do
    # Extracting the filename without the extension for the output
    output_name=$(basename "$notebook" .ipynb)

    echo "Converting $notebook..."
    jupyter nbconvert --execute "$notebook" --to notebook --output "${output_name}_converted.ipynb"
    
    if [ $? -ne 0 ]; then
        echo "Error converting $notebook. Exiting."
        exit 1
    fi
done

echo "All notebooks converted successfully."

