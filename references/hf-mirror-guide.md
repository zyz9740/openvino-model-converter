# HuggingFace Mirror (hf-mirror.com) Configuration Guide

hf-mirror.com is a public mirror of huggingface.co for users in China where the original site is slow or inaccessible. All HuggingFace model and dataset downloads should go through this mirror.

## Method 1: Environment Variable (Recommended)

The simplest and least invasive approach. HuggingFace tools read the `HF_ENDPOINT` environment variable to determine the download URL.

**Windows PowerShell:**
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python your_script.py
```

**Linux / macOS:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python your_script.py
```

Or inline:
```bash
HF_ENDPOINT=https://hf-mirror.com python your_script.py
```

Add the export line to `~/.bashrc` for persistence.

This works with `transformers`, `diffusers`, `huggingface_hub`, and any library that calls HuggingFace download APIs internally.

## Method 2: hf CLI

The official HuggingFace CLI with built-in download management (resumable, retry, parallel workers). In recent `huggingface_hub` releases the command is `hf`; the older `huggingface-cli` entry point still works as a deprecated alias, but prefer `hf`.

1. Install: `pip install -U huggingface_hub`
2. Set mirror: `$env:HF_ENDPOINT = "https://hf-mirror.com"` (PowerShell) or `export HF_ENDPOINT=https://hf-mirror.com` (Linux)
3. Download model:
   ```
   hf download <org>/<model> --local-dir <model>
   ```
4. Download dataset:
   ```
   hf download <dataset> --repo-type dataset --local-dir <dataset>
   ```

Downloads are resumable by default, so no separate resume flag is needed. `--local-dir` writes plain files (no symlinks) directly into the target folder.

## Method 3: hfd (HuggingFace Downloader)

A dedicated download tool built on `aria2` for stable, high-speed, resumable downloads. Good for very large models.

1. Download the script:
   ```bash
   wget https://hf-mirror.com/hfd/hfd.sh
   chmod a+x hfd.sh
   ```
2. Set mirror: `export HF_ENDPOINT=https://hf-mirror.com`
3. Download model: `./hfd.sh <org>/<model>`
4. Download dataset: `./hfd.sh <dataset> --dataset`

Note: hfd is Linux/macOS only. On Windows, use Method 1 or Method 2.

## Gated Repos (Login Required)

Some models (e.g., Llama, Gemma) require accepting a license on HuggingFace first. Since hf-mirror.com does not support login:

1. Go to the original huggingface.co model page and accept the license agreement
2. Get an Access Token from https://huggingface.co/settings/tokens
3. Use the token with your download tool:

   **hf CLI:**
   ```
   hf download <org>/<model> --token hf_*** --local-dir <model>
   ```

   **hfd:**
   ```
   ./hfd.sh <org>/<model> --hf_username YOUR_USERNAME --hf_token hf_***
   ```

   **Python (from_pretrained):**
   ```python
   from huggingface_hub import login
   login(token="hf_***")
   ```

## In Python Code

When using `from_pretrained()` or any HuggingFace download API in Python, just set the environment variable before importing:

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
```

## Quick Decision Guide

| Situation | Method |
|-----------|--------|
| Normal model download | Method 1 (env var) + pip/transformers |
| Large model (multi-GB) | Method 3 (hfd with aria2) on Linux, Method 2 on Windows |
| Gated model (needs license) | Get token first, then Method 2 with `--token` |
| In a Python script | Set `os.environ["HF_ENDPOINT"]` before imports |
