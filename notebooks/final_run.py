# %% [markdown]
# # Final run: 4b (available?) + 4c (positive control)
#
# Paste each `# %%` block into its own Kaggle cell, then **Save Version ->
# Save & Run All (Commit)**. Nothing needs editing.
#
# Requires: GPU accelerator, Internet ON, both datasets added as Inputs.
#
# Runtime ~3.5 h. Results flush after every trained model and completed cells
# are skipped on re-entry, so a timeout costs one cell, not the run. Anything
# already finished (4b on fetal_planes) is skipped automatically.
#
# Produces `/kaggle/working/runs/FINAL_SUMMARY.md` plus per-experiment CSVs.

# %%
# ============================ CELL 1: setup ================================
import csv
import os
import shutil
import socket
import sys
from pathlib import Path

ENV = "kaggle" if Path("/kaggle/input").exists() else "local"
print("environment:", ENV)


def find_repo() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    if ENV == "kaggle":
        candidates += sorted(Path("/kaggle/input").glob("*"))
    for candidate in candidates:
        if (candidate / "src" / "asa" / "__init__.py").exists():
            return candidate
    if ENV == "kaggle":
        for marker in Path("/kaggle/input").rglob("asa/__init__.py"):
            repo = marker.parent.parent.parent
            if (repo / "src" / "asa").exists():
                return repo
    raise FileNotFoundError("could not find src/asa -- add the code dataset as an Input")


def find_packages() -> dict:
    roots = sorted(Path("/kaggle/input").glob("*")) if ENV == "kaggle" else [REPO / "work"]
    found = {}
    for root in roots:
        for meta in sorted(root.rglob("meta.json")):
            pack = meta.parent
            if not (pack / "index.csv").exists():
                continue
            with open(pack / "index.csv", newline="") as handle:
                first = next(csv.DictReader(handle), None)
            found.setdefault((first or {}).get("dataset") or pack.parent.name, pack)
    return found


REPO = find_repo()
sys.path.insert(0, str(REPO / "src"))
CONFIG_ROOT = REPO / "configs"
OUT_ROOT = Path("/kaggle/working/runs") if ENV == "kaggle" else REPO / "runs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PACKS = find_packages()

# torchvision fetches ImageNet weights at model-build time, so a missing
# checkpoint fails minutes into the run rather than at the start.
TORCH_HOME = Path("/kaggle/working/torch") if ENV == "kaggle" else Path.home() / ".cache/torch"
(TORCH_HOME / "hub" / "checkpoints").mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(TORCH_HOME)
if ENV == "kaggle":
    for source in Path("/kaggle/input").rglob("*.pth"):
        target = TORCH_HOME / "hub" / "checkpoints" / source.name
        if not target.exists():
            shutil.copy(source, target)

import torch  # noqa: E402

try:
    socket.create_connection(("download.pytorch.org", 443), timeout=5).close()
    internet = True
except OSError:
    internet = False

print("repo:    ", REPO)
print("output:  ", OUT_ROOT)
print("packages:", {k: str(v) for k, v in PACKS.items()})
print("torch    ", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("internet:", "ON" if internet else "OFF")

assert torch.cuda.is_available(), "No GPU. Settings -> Accelerator -> GPU, then restart."
assert "fetal_planes" in PACKS, f"fetal_planes not mounted; have {list(PACKS)}"
assert internet or any((TORCH_HOME / "hub" / "checkpoints").glob("*.pth")), (
    "No internet and no staged weights -- the run would die at model build. "
    "Turn Internet ON in Settings."
)
print("\nsetup OK")

# %%
# ===================== CELL 2: experiment runner ===========================
import pandas as pd  # noqa: E402

from asa.config import load_experiment  # noqa: E402
from asa.data.packaging import load_package  # noqa: E402
from asa.train.matrix import run_matrix  # noqa: E402

CELLS = ["a_full", "a_full__nomarker", "a_only", "a_removed", "ctrl_random"]


def real_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the paired-contrast rows. `is_contrast` is NaN on real rows and True
    on contrast rows, so the column is object dtype and `~` on it would be
    bitwise, not logical -- hence the explicit cast."""
    flag = frame.get("is_contrast", pd.Series(False, index=frame.index))
    return frame[~flag.fillna(False).astype(bool)]


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        real_rows(frame)
        .groupby("test_variant")
        .agg(
            bal_acc=("balanced_accuracy", "mean"),
            sd=("balanced_accuracy", "std"),
            chance=("chance_balanced_accuracy", "mean"),
            ece=("ece", "mean"),
            n=("balanced_accuracy", "size"),
        )
        .round(4)
    )


def run_experiment(experiment: str, datasets, tag: str, overrides=None, folds=(0, 1, 2)):
    """Train + evaluate, print a summary, return the raw frame.

    No gate logic anywhere: `evaluate_gate` is defined only for A-full-trained
    models and raises on everything else, which is what derailed earlier runs.
    """
    frames = []
    for name in datasets:
        if name not in PACKS:
            print(f"SKIP {name}: not mounted")
            continue
        cfg = load_experiment(
            CONFIG_ROOT / "experiments" / f"{experiment}.yaml",
            CONFIG_ROOT,
            {"dataset": name, **(overrides or {})},
        )
        index, arrays, _ = load_package(PACKS[name])
        frame = run_matrix(
            cfg, index, arrays,
            out_dir=OUT_ROOT / tag / name,
            folds=list(folds),
            resume=True,
        )
        frame = frame.assign(_experiment=experiment, _tag=tag, _dataset=name)
        frames.append(frame)
        print(f"\n=== {name} / {tag} ===")
        print(summarise(frame).to_string())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


print("runner ready")

# %%
# ============ CELL 3: 4b -- are the annotations learnable at all? ==========
# Trains ON A-only. fetal_planes is already done and resumes instantly.
# Expect a_only well above chance if the annotations carry usable signal.

res_4b = run_experiment(
    "step4b_available_vs_taken",
    ["fetal_planes", "bus_bra"],
    tag="step4b",
)

# %%
# ====== CELL 4: 4c -- positive control, sensitivity calibration ============
# Injects a marker matched to the real annotations in area (0.7% of pixels) and
# intensity, at four correlation strengths.
#
# The number that matters: a_full MINUS a_full__nomarker. That is what removing
# a shortcut the model DOES use costs. Compare it against the real
# annotation-removal cost of -0.0002 measured in the gate.
#
# Ordered strongest-first so a timeout still leaves the decisive point.

res_4c = {}
for strength in (1.0, 0.75, 0.5, 0.25):
    res_4c[strength] = run_experiment(
        "step4c_positive_control",
        ["fetal_planes"],
        tag=f"step4c/strength_{strength}",
        overrides={"synthetic": {"enabled": True, "strength": strength}},
    )

# %%
# ================== CELL 5: consolidated final summary =====================
import json  # noqa: E402

lines = ["# Final results", ""]

# ---- 4b -------------------------------------------------------------------
lines += ["## 4b -- trained ON A-only (is the annotation signal learnable?)", ""]
if len(res_4b):
    for name, group in res_4b.groupby("_dataset"):
        s = summarise(group)
        if "a_only" in s.index:
            row = s.loc["a_only"]
            lines.append(
                f"- **{name}**: A-only = {row.bal_acc:.4f} (sd {row.sd:.4f}) "
                f"vs chance {row.chance:.4f} -- {row.bal_acc / row.chance:.1f}x chance"
            )
    for name, group in res_4b.groupby("_dataset"):
        lines += ["", f"### {name}", "", "```", summarise(group).to_string(), "```"]
    lines.append("")

# ---- 4c -------------------------------------------------------------------
lines += ["## 4c -- positive control (what does removing a USED shortcut cost?)", ""]
lines += ["| strength | a_full | a_full__nomarker | drop | a_only | chance |", "|---|---|---|---|---|---|"]
curve = []
for strength, frame in res_4c.items():
    if not len(frame):
        continue
    s = summarise(frame)
    full = s.loc["a_full", "bal_acc"] if "a_full" in s.index else float("nan")
    nomark = s.loc["a_full__nomarker", "bal_acc"] if "a_full__nomarker" in s.index else float("nan")
    only = s.loc["a_only", "bal_acc"] if "a_only" in s.index else float("nan")
    chance = s.loc["a_full", "chance"] if "a_full" in s.index else float("nan")
    lines.append(f"| {strength} | {full:.4f} | {nomark:.4f} | **{full - nomark:+.4f}** | {only:.4f} | {chance:.4f} |")
    curve.append({"strength": strength, "a_full": full, "a_full_nomarker": nomark,
                  "drop": full - nomark, "a_only": only, "chance": chance})

lines += [
    "",
    "### How to read this",
    "",
    "`drop` is what removing a shortcut the model demonstrably uses costs.",
    "The real annotation-removal cost measured in the Step-4 gate was **-0.0002**",
    "on fetal_planes, against a run-to-run SD of 0.0037.",
    "",
    "- If `drop` at strength 1.0 is large, the A-removed cell CAN detect a",
    "  shortcut at 0.7% pixel area, so the real null is a genuine null.",
    "- If `drop` at strength 1.0 is also near zero, the design cannot detect any",
    "  feature of this size, and that limitation is the finding -- not the null.",
    "- The strength column gives the sensitivity floor: the weakest",
    "  annotation-label correlation this design can see.",
    "",
]

(OUT_ROOT / "FINAL_SUMMARY.md").write_text("\n".join(lines) + "\n")
(OUT_ROOT / "positive_control_curve.json").write_text(json.dumps(curve, indent=2))

print("\n".join(lines))
print(f"\nwritten: {OUT_ROOT / 'FINAL_SUMMARY.md'}")
print("Download the whole runs/ folder from the Output tab.")
