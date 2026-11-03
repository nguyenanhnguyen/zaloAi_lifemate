"""Tải lại đúng hai model mà bản 22.52 đã dùng, theo revision đã pin.

    python restore.py            # tải cả hai vào baseline_2252/models/
    python restore.py --check    # chỉ báo thiếu gì, không tải

Revision lấy từ pipeline/src/llm_agent.py của chính commit 7002744, nên bản tải về là đúng
bit đã chạy -- không phải "bản mới nhất của cùng repo".
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "models" / "hf-cache"

LLM_REPO = "unsloth/Qwen3.5-4B-GGUF"
LLM_REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
LLM_FILE = "Qwen3.5-4B-Q8_0.gguf"
ASR_REPO = "Qualcomm-AI-Research/PhoASR-whisper-small"
ASR_REVISION = "da2d9ae0ac0a6a6faba20add61c2e980ccd85037"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="chỉ kiểm tra đã có sẵn trong cache chưa")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    CACHE.mkdir(parents=True, exist_ok=True)
    local = args.check

    gguf = hf_hub_download(LLM_REPO, LLM_FILE, revision=LLM_REVISION,
                           cache_dir=str(CACHE), local_files_only=local)
    print(f"LLM: {gguf}")
    print(f"     {Path(gguf).stat().st_size / 2**30:.2f} GiB")

    asr = snapshot_download(ASR_REPO, revision=ASR_REVISION, cache_dir=str(CACHE),
                            local_files_only=local,
                            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model",
                                            "pytorch_model.bin"])
    print(f"ASR: {asr}")

    # llama-server không nằm trên HuggingFace nên script này không lấy được; chỉ báo.
    binary = next((HERE / "models" / "llama.cpp").rglob("llama-server*"), None)
    if binary is None:
        print("\nCòn thiếu llama-server của llama.cpp. Đặt nó vào "
              f"{HERE / 'models' / 'llama.cpp'} hoặc trỏ biến LLAMA_SERVER tới nó.")
    else:
        print(f"llama-server: {binary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
