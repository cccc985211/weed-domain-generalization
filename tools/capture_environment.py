"""EXP-001 Step A: capture reproducible environment information.

Writes reports/EXP-001/environment.json. Run with the project venv:
    .venv/Scripts/python tools/capture_environment.py
"""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")
OUT = os.path.join(REPO, "reports", "EXP-001", "environment.json")


def scrub(obj):
    """Remove private filesystem paths from recorded strings."""
    if isinstance(obj, str):
        return (obj.replace(REPO, "<REPO>")
                   .replace(HOME, "<HOME>")
                   .replace(HOME.replace("\\", "/"), "<HOME>"))
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def _run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=60).strip()
    except Exception as e:  # noqa: BLE001 - record failure verbatim
        return f"<failed: {e}>"


def main():
    info = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "git": {
            "repo": REPO,
            "commit": _run(["git", "-C", REPO, "rev-parse", "HEAD"]),
            "branch": _run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": _run(["git", "-C", REPO, "status", "--porcelain"]),
        },
        "nvidia_smi": _run(["nvidia-smi"]),
        "nvidia_smi_gpu": _run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"]
        ),
    }

    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            info["torch"]["device_name_0"] = torch.cuda.get_device_name(0)
            info["torch"]["device_capability_0"] = list(torch.cuda.get_device_capability(0))
            x = torch.randn(4, 3, 64, 64, device="cuda")
            info["torch"]["cuda_tensor_smoke"] = bool(torch.isfinite(x.sum()).item())
    except Exception as e:  # noqa: BLE001
        info["torch"] = {"import_error": str(e)}

    for pkg in ("torchvision", "ultralytics", "cv2", "PIL", "numpy"):
        try:
            mod = __import__(pkg)
            info.setdefault("packages", {})[pkg] = getattr(mod, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            info.setdefault("packages", {})[pkg] = f"<missing: {e}>"

    info["pip_freeze"] = _run([sys.executable, "-m", "pip", "freeze"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(scrub(info), f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
