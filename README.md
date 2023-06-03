# llama_knife

## Usage
Run Python interactively using llama_knife.py. I'm using ptpython, there's also ipython and I guess vanilla python3.
Provide the model as the first argument.
Inference.

```python
# "Reverse prompt", stop generating when these strings are found
stop_words = ["User:", "Human:", "Instruction:"]

# Edit the prompt and generated text with $EDITOR (set before you run if unset)
# "Stream" input to the interpreter. end_signal will contain "" or a reason generation stopped. Add "%QUIT" anywhere in the text to abort generation.
end_signal = inference(stop_words, max_tokens=124)

# To print the text again.
print(
    tokens_to_str(
        file_to_tokens()
    )
)
```
