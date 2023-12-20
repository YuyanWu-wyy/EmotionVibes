#!/bin/bash

# List of notebooks to be converted
notebooks=(
    "AdptivePrune_remove_autocorr.ipynb"
    "AdptivePrune_remove_cwt.ipynb"
    "AdptivePrune_remove_energy.ipynb"
    "AdptivePrune_remove_fft.ipynb"
    "AdptivePrune_remove_spe.ipynb"
    "AdptivePrune_remove_gaitpara.ipynb"
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

