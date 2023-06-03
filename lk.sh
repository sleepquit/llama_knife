#!/bin/bash
if [ -z "$(which fzf)" ]; then
    echo "Please install fzf!"
    echo ""
else
    model_dir="$HOME/Downloads/ggml_models/v3/" #Changeme
    model="$(ls ${model_dir} | fzf)"
    pyinterpreter="ptpython"

    "$pyinterpreter" -i ./llama_knife.py -- ${model_dir}${model}
fi
