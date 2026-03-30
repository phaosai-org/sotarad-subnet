pip3 install huggingface-hub

hf download xxx/yyy --local-dir zzz

pip3 install sglang

python -m sglang.launch_server --model-path local_model_path --host 0.0.0.0 --port 8000 --mem-fraction-static 0.9 --context-length 32768
