#!/bin/bash

# List of notebooks to be converted
notebooks=(
    "Feature_extraction.ipynb"
    "Feature_extraction_only_autocorr.ipynb"
    "Feature_extraction_only_cwt.ipynb"
    "Feature_extraction_only_fft.ipynb"
    "Feature_extraction_only_energy.ipynb"
    "Feature_extraction_only_gait.ipynb"
    "Feature_extraction_only_spe.ipynb"
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

