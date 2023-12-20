#!/bin/bash

# List of notebooks to be converted
notebooks=(
    "unknown_baseline_p1_few_shot.ipynb"
    "unknown_baseline_p2_few_shot.ipynb"
    "unknown_baseline_p4_few_shot.ipynb"
    "unknown_baseline_p5_few_shot.ipynb"
    "unknown_baseline_p6_few_shot.ipynb"
    "unknown_baseline_p8_few_shot.ipynb"
    "unknown_baseline_p9_few_shot.ipynb"
    "unknown_baseline_p10_few_shot.ipynb"
    "unknown_baseline_p11_few_shot.ipynb"
    "unknown_baseline_p12_few_shot.ipynb"
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

