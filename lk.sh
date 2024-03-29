#!/bin/bash
if [ -z "$(which fzf)" ]; then
    echo "Please install fzf!"
    echo ""
else
    model_dir="$HOME/Downloads/ggml_models/v3/" #Changeme
    if [ ! -d "$model_dir" ]; then
        echo "Path does not exist: $model_dir"
        echo "Update this script with a real path."
        exit 1
    fi

    model="$(ls ${model_dir} | fzf)"
    pyinterpreter="ptpython"

    "$pyinterpreter" -i ./llama_knife.py -- ${model_dir}${model}
fi
