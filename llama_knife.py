#!/usr/bin/env python3
import subprocess
import os
import sys
import json
import pathlib
from llama_cpp import Llama

default_token_file = ".token_store.json"
default_session_file = ".session.tmp"
input_signals = ["%QUIT"]

def str_to_tokens(strarg):
    string_bytes = bytes(strarg, "utf-8")
    string_tokens = llm.tokenize(string_bytes)
    return string_tokens

def tokens_to_str(tokarg):
    return llm.detokenize(tokarg).decode("utf-8")

def tokens_to_file(tokarg, output_file=default_token_file):
    tok_object_string = json.dumps(tokarg)
    pathlib.Path(output_file).write_text(tok_object_string)
    return

def file_to_tokens(filearg=default_token_file):
    return json.loads(pathlib.Path(filearg).read_text())

def save_session(
    tokarg, token_file=default_token_file, session_file=default_session_file
):
    tokens_to_file(tokarg)
    pathlib.Path(session_file).write_text(tokens_to_str(tokarg))
    return

def input_from_editor():
    cmd = f"{os.environ['EDITOR']} " + default_session_file
    subprocess.call(cmd.split(" "))
    session_text = pathlib.Path(default_session_file).read_text()
    return session_text

def posix_clear():
    print(os.popen("clear").read())
    return

if __name__ == "__main__":
    data_fname = "data.json"
    data_file = pathlib.Path(data_fname)
    ggml_path = sys.argv[1] if len(sys.argv) > 1 else None

    ngl_count = 0
    if "-ngl" in sys.argv:
        ngl_count = int(sys.argv[sys.argv.index("-ngl") + 1])

    data_file.write_text("{}") if not data_file.exists() else 0
    run_data = json.loads(data_file.read_text())

    if not ggml_path and "last_model" in run_data:
        print("\t Use previous model?")
        print("\t ", run_data["last_model"])
        uinput = input("> ").lower()
        if "y" in uinput:
            ggml_path = run_data["last_model"]
        else:
            print("Bye!")
            quit()
    elif not ggml_path and not "last_model" in run_data:
        print("Provide model path as an arg.")
        quit()

    if not pathlib.Path(ggml_path).exists():
        print("Model doesn't exist?\n", ggml_path)
        quit()

    run_data["last_model"] = ggml_path
    data_file.write_text(json.dumps(run_data, indent=4))
    
    llm = Llama(
        model_path=ggml_path,
        seed=0,
        n_ctx=2048,
        use_mmap=True,
        n_threads=4,
        n_gpu_layers=ngl_count
    )

    def inference(stop_words=[], max_tokens=512, mirostat=0, temp=.8, tts=False):
        input_signal = ""
        session_text = input_from_editor()
        for sig in input_signals:
            if sig in session_text:
                input_signal = sig
                return input_signal

        if session_text[-1] == "\n":
            session_text = session_text[0:-1]
        session_tokens = str_to_tokens(session_text)
        text_gen = llm.generate(session_tokens, mirostat_mode=mirostat, temp=temp)
        print(session_text)

        new_text = ""
        posix_clear()
        token_count = len(session_tokens)
        if (token_count + max_tokens) > 2048:
            input_signal = "%TOKLIMIT"
            return input_signal

        for i in range(max_tokens):
            next_token = text_gen.__next__()
            if next_token == llm.token_eos():
                #print("EOS token detected")
                break
            session_tokens.append(next_token)

            new_text += tokens_to_str([next_token])
            print(new_text, end="\r")
            if any([word for word in stop_words if word in new_text]):
                break
        print("")
        save_session(session_tokens)
        print("Token count: ", len(session_tokens)) #debug?
        return input_signal
