#!/bin/bash

# List of notebooks to be converted
notebooks=(
    "unknown_baseline_p8_few_shot_evaluation.ipynb"
    "unknown_baseline_p9_few_shot_evaluation.ipynb"
    "unknown_baseline_p10_few_shot_evaluation.ipynb"
    "unknown_baseline_p11_few_shot_evaluation.ipynb"
    "unknown_baseline_p12_few_shot_evaluation.ipynb"
    "unknown_baseline_p13_few_shot_evaluation.ipynb"
    "unknown_baseline_p17_few_shot_evaluation.ipynb"
    "unknown_baseline_p19_few_shot_evaluation.ipynb"
    "unknown_baseline_p21_few_shot_evaluation.ipynb"
    "unknown_baseline_p22_few_shot_evaluation.ipynb"
    "unknown_baseline_p25_few_shot_evaluation.ipynb"
    "unknown_baseline_p26_few_shot_evaluation.ipynb"
    "unknown_baseline_p27_few_shot_evaluation.ipynb"
    "unknown_baseline_p28_few_shot_evaluation.ipynb"
    "unknown_baseline_p29_few_shot_evaluation.ipynb"
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

