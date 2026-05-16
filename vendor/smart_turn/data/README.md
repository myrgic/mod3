# Smart Turn v3.1 ONNX Weight

Place `smart-turn-v3.1.onnx` in this directory before running inference.

## Download

```bash
# Using HuggingFace CLI
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='pipecat-ai/smart-turn', filename='smart-turn-v3.1.onnx', local_dir='vendor/smart_turn/data')"
```

Or see `vendor/MANIFEST.toml` for the pinned revision and checksum.
