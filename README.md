# Local_Llama_Chat 

Only Python code for a chat with history.
You still have to install dependencies.

On Windows create virtual environment:
python -m venv .venv  # Creates a virtual environment in a directory named .venv

Do not forget to set ENV for windows, for Nvidia CUDA cores usage at inference: 
set CMAKE_ARGS=-DGGML_CUDA=ON

Install llama-cpp-python (force reinstall with no cached files, if needed)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123 --upgrade --force-reinstall --no-cache-dir
