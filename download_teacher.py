import os
from huggingface_hub import snapshot_download

def main():
    print("[Download] Downloading SmolLM2-360M teacher model offline cache via python api...")
    snapshot_download(
        repo_id="HuggingFaceTB/SmolLM2-360M",
        revision="f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        local_dir="../local_smollm2",
        local_dir_use_symlinks=False
    )
    print("[Download] Teacher model downloaded successfully to ../local_smollm2.")

if __name__ == "__main__":
    main()
