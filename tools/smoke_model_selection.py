"""EXP-001 Step D (revised): model-selection smoke test.

Roster:  yolo26n.pt (primary candidate) | yolo11n.pt (historical baseline)
         yolo26s.pt (stronger baseline)

For each model, on the SAME deterministic synthetic split
(datasets_derived/smoke_weed_v1) with imgsz=640, seed=42, workers=0, <=2 epochs:
  * weight provenance + SHA-256
  * architecture / output-structure notes (head type, dual-head, params)
  * explicit forward + backward health check
  * 2-epoch training: wall time, peak CUDA memory, val metrics (NOT a ranking)

Outputs -> reports/EXP-001/model_selection/smoke_metrics.json (+ per-model logs)
Run artifacts -> <WORKSPACE>/runs/EXP-001/model_selection
"""
import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.dirname(REPO)
WEIGHTS = os.path.join(EXP, "weights")
DATA_YAML = os.path.join(EXP, "datasets_derived", "smoke_weed_v1", "data.yaml")
RUNS = os.path.join(EXP, "runs", "EXP-001", "model_selection")
REPORT_DIR = os.path.join(REPO, "reports", "EXP-001", "model_selection")
MODELS = ["yolo26n.pt", "yolo11n.pt", "yolo26s.pt"]
if len(os.sys.argv) > 1:  # optional: restrict to given model names
    MODELS = [m for m in MODELS if m in os.sys.argv[1:]]

COMMON = dict(
    data=DATA_YAML, epochs=2, imgsz=640, batch=8, seed=42, workers=0,
    device=0, deterministic=True, pretrained=True, project=RUNS,
    exist_ok=True, verbose=False,
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_gpu_mib():
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def describe_model(yolo):
    import torch

    m = yolo.model
    last = m.model[-1]
    info = {
        "task": getattr(yolo, "task", None),
        "model_class": type(m).__name__,
        "head_class": type(last).__name__,
        "head_repr": str(last)[:2000],
        "params": sum(p.numel() for p in m.parameters()),
        "n_modules": len(list(m.modules())),
        "head_attrs": {
            a: (v if isinstance(v, (int, float, str, bool, list, tuple,
                                    dict, type(None)))
                else type(v).__name__)
            for a in ("end2end", "one2one", "nc", "reg_max", "no", "nl",
                      "stride")
            if hasattr(last, a)
            for v in [getattr(last, a)]
        },
    }
    # eval-mode forward output structure
    m.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 640, 640, device=next(m.parameters()).device)
        out = m(x)
    def shape_of(o):
        if isinstance(o, torch.Tensor):
            return list(o.shape)
        if isinstance(o, (list, tuple)):
            return [shape_of(i) for i in o]
        if isinstance(o, dict):
            return {k: shape_of(v) for k, v in o.items()}
        return type(o).__name__
    info["eval_forward_structure"] = shape_of(out)
    return info


def _collect_tensors(o, acc):
    import torch

    if torch.is_tensor(o):
        acc.append(o)
    elif isinstance(o, (list, tuple)):
        for i in o:
            _collect_tensors(i, acc)
    elif isinstance(o, dict):
        for v in o.values():
            _collect_tensors(v, acc)


def fwd_bwd_check(yolo):
    """Forward + backward health on the raw inference graph (eval mode).

    Train-mode forward dispatches to model.loss() which needs a real batch,
    so we verify graph connectivity via _predict_once instead."""
    import torch

    try:
        m = yolo.model
        m.eval()
        for p in m.parameters():
            p.requires_grad_(True)
        x = torch.rand(1, 3, 640, 640,
                        device=next(m.parameters()).device)
        out = m._predict_once(x)  # raw graph, eval-mode detect decode
        flat = []
        _collect_tensors(out, flat)
        grad_leaves = [t for t in flat if t.requires_grad]
        if not grad_leaves:
            return {"ok": False,
                    "note": f"no grad-requiring output; type={type(out).__name__}"}
        loss = sum(t.float().sum() for t in grad_leaves)
        loss.backward()
        n_grad = sum(1 for p in m.parameters() if p.grad is not None)
        m.zero_grad(set_to_none=True)
        return {"ok": True, "n_output_tensors": len(flat),
                "n_grad_outputs": len(grad_leaves),
                "n_params_with_grad": n_grad}
    except BaseException as e:  # noqa: BLE001
        return {"ok": False, "note": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1500:]}


def run_one(name, torch, YOLO):
    rec = {"model": name, "started_utc": datetime.now(timezone.utc).isoformat()}
    wpath = os.path.join(WEIGHTS, name)
    t0 = time.time()
    try:
        yolo = YOLO(wpath)  # downloads to WEIGHTS if missing
        rec["weights_path"] = wpath
        rec["weights_sha256"] = sha256(wpath)
        rec["weights_mb"] = round(os.path.getsize(wpath) / 1e6, 2)
        rec["architecture"] = describe_model(yolo)
        rec["fwd_bwd"] = fwd_bwd_check(yolo)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem0 = peak_gpu_mib()
        ttrain = time.time()
        results = yolo.train(name=name.replace(".pt", ""), **COMMON)
        rec["train_wall_s"] = round(time.time() - ttrain, 1)
        rec["torch_peak_alloc_mib"] = round(
            torch.cuda.max_memory_allocated() / 2**20, 1)
        rec["torch_peak_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / 2**20, 1)
        mem1 = peak_gpu_mib()
        rec["nvsmi_used_before_mib"] = mem0
        rec["nvsmi_used_after_mib"] = mem1

        rec["train_ok"] = True
        rd = getattr(results, "results_dict", None) or {}
        rec["val_metrics"] = {k: float(v) for k, v in rd.items()
                              if isinstance(v, (int, float))}
        tr = getattr(yolo, "trainer", None)
        if tr is not None:
            rec["optimizer_used"] = type(getattr(tr, "optimizer", None)).__name__
            rec["train_args"] = {
                k: str(getattr(tr.args, k, None)) for k in
                ("optimizer", "lr0", "lrf", "momentum", "weight_decay",
                 "epochs", "batch", "imgsz", "seed", "workers", "mosaic",
                 "hsv_h", "hsv_s", "hsv_v", "close_mosaic")
            }
        rec["run_dir"] = str(getattr(tr, "save_dir", "")) if tr else ""
    except BaseException as e:  # noqa: BLE001 - record failure, keep going
        rec["train_ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["wall_s_total"] = round(time.time() - t0, 1)
    rec["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return rec


def main():
    import torch
    import ultralytics
    from ultralytics import YOLO

    os.makedirs(WEIGHTS, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    support = {
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "yolo26_cfg_present": os.path.exists(os.path.join(
            os.path.dirname(ultralytics.__file__), "cfg", "models", "26",
            "yolo26.yaml")),
    }
    with open(os.path.join(REPORT_DIR, "model_support.json"), "w") as f:
        json.dump(support, f, indent=2)
    print("support:", support)

    records = []
    prev_path = os.path.join(REPORT_DIR, "smoke_metrics.json")
    if os.path.exists(prev_path):  # merge with prior partial run
        try:
            prev = json.load(open(prev_path))
            records = [m for m in prev.get("models", [])
                       if m.get("model") not in MODELS]
        except Exception:
            pass
    for name in MODELS:
        print(f"=== {name} ===", flush=True)
        records.append(run_one(name, torch, YOLO))
        try:
            with open(os.path.join(REPORT_DIR, "smoke_metrics.json"), "w") as f:
                json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(),
                           "data_yaml": DATA_YAML,
                           "note": "synthetic placeholder split; metrics are a "
                                   "pipeline check, NOT an accuracy ranking",
                           "common_train_args": {k: str(v)
                                                for k, v in COMMON.items()},
                           "models": records}, f, indent=2, default=str)
        except BaseException:  # noqa: BLE001
            with open(os.path.join(REPORT_DIR, "smoke_metrics_dump_error.txt"),
                      "w") as f:
                f.write(traceback.format_exc())
    print(json.dumps([{k: r.get(k) for k in
                       ("model", "train_ok", "train_wall_s",
                        "torch_peak_alloc_mib", "error")}
                      for r in records], indent=2))


if __name__ == "__main__":
    main()
