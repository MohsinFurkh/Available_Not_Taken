# %% [markdown]
# # Revision run — closing the review items that need compute
#
# Ordered by value per GPU-hour, so partial completion still helps. Each block
# resumes independently; a session boundary costs one cell, not the run.
#
# | block | closes | cost |
# |---|---|---|
# | A. glyph control sweep | **item 1** — learnability-matched calibration | ~3 h |
# | B. revision matrix | items 3, 7b, 12 — per-class, 5 seeds, clean-trained row | ~3 h |
# | C. HC18 + MMOTU gate | item 4 — the four-dataset claim | ~30 min |
# | D. 448 px replicate | item 2 — resolution confound | ~2 h + a 4.9 GB upload |
#
# **Block A is the one that changes the paper's standing.** If only one block
# finishes, make it that one.
#
# Requires: GPU, Internet ON, code + packs datasets added as Inputs.

# %%
# ======================= CELL 1: setup (unchanged) =========================
import csv, os, shutil, socket, sys
from pathlib import Path

ENV = "kaggle" if Path("/kaggle/input").exists() else "local"


def find_repo() -> Path:
    for c in [Path.cwd(), *Path.cwd().parents] + (
        sorted(Path("/kaggle/input").glob("*")) if ENV == "kaggle" else []
    ):
        if (c / "src" / "asa" / "__init__.py").exists():
            return c
    if ENV == "kaggle":
        for m in Path("/kaggle/input").rglob("asa/__init__.py"):
            if (m.parent.parent.parent / "src" / "asa").exists():
                return m.parent.parent.parent
    raise FileNotFoundError("add the code dataset as an Input")


def find_packages() -> dict:
    roots = sorted(Path("/kaggle/input").glob("*")) if ENV == "kaggle" else [REPO / "work"]
    found = {}
    for root in roots:
        for meta in sorted(root.rglob("meta.json")):
            pack = meta.parent
            if not (pack / "index.csv").exists():
                continue
            with open(pack / "index.csv", newline="") as h:
                first = next(csv.DictReader(h), None)
            # key on (dataset, resolution) so 224 and 448 packs coexist
            import json as _json
            size = _json.loads((pack / "meta.json").read_text())["size"]
            found[((first or {}).get("dataset") or pack.parent.name, size)] = pack
    return found


REPO = find_repo()
sys.path.insert(0, str(REPO / "src"))
CONFIG_ROOT = REPO / "configs"
OUT_ROOT = Path("/kaggle/working/runs") if ENV == "kaggle" else REPO / "runs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PACKS = find_packages()

TORCH_HOME = Path("/kaggle/working/torch") if ENV == "kaggle" else Path.home() / ".cache/torch"
(TORCH_HOME / "hub" / "checkpoints").mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(TORCH_HOME)
if ENV == "kaggle":
    for s in Path("/kaggle/input").rglob("*.pth"):
        t = TORCH_HOME / "hub" / "checkpoints" / s.name
        if not t.exists():
            shutil.copy(s, t)

import torch

try:
    socket.create_connection(("download.pytorch.org", 443), timeout=5).close()
    internet = True
except OSError:
    internet = False

print("repo:    ", REPO)
print("packages:", {f"{k[0]}@{k[1]}": str(v) for k, v in PACKS.items()})
print("gpu:     ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("internet:", "ON" if internet else "OFF")
assert torch.cuda.is_available(), "Settings -> Accelerator -> GPU"
assert internet or any((TORCH_HOME / "hub" / "checkpoints").glob("*.pth"))
print("\nsetup OK")

# %%
# ========================= CELL 2: runner ==================================
import pandas as pd

from asa.config import load_experiment
from asa.data.packaging import load_package
from asa.train.matrix import run_matrix


def real_rows(f):
    flag = f.get("is_contrast", pd.Series(False, index=f.index))
    return f[~flag.fillna(False).astype(bool)]


def summarise(f):
    return (real_rows(f).groupby("test_variant")
            .agg(bal_acc=("balanced_accuracy", "mean"), sd=("balanced_accuracy", "std"),
                 macro_f1=("macro_f1", "mean"), ece=("ece", "mean"),
                 chance=("chance_balanced_accuracy", "mean"), n=("balanced_accuracy", "size"))
            .round(4))


def run(experiment, dataset, tag, overrides=None, size=224, folds=(0, 1, 2)):
    key = (dataset, size)
    if key not in PACKS:
        print(f"SKIP {dataset}@{size}: not mounted")
        return pd.DataFrame()
    cfg = load_experiment(CONFIG_ROOT / "experiments" / f"{experiment}.yaml", CONFIG_ROOT,
                          {"dataset": dataset, **(overrides or {})})
    index, arrays, _ = load_package(PACKS[key])
    frame = run_matrix(cfg, index, arrays, out_dir=OUT_ROOT / tag / dataset,
                       folds=list(folds), resume=True)
    print(f"\n=== {dataset}@{size} / {tag} ===")
    print(summarise(frame).to_string())
    return frame.assign(_dataset=dataset, _tag=tag)


print("runner ready")

# %%
# === CELL 3 / BLOCK A: glyph control — the learnability-matched calibration ==
# The rectangle marker is a solid block at one of K fixed positions: far easier
# to read than a caliper or a burnt-in word, so its sweep is an UPPER bound on
# sensitivity. The glyph cue carries the class in its *content* (vendor-like
# strings), scattered in position, jittered in scale, 25% partially occluded.
# Report both curves; the real annotations belong against the harder one.

res_glyph = {}
for rho in (1.0, 0.75, 0.5, 0.25):
    res_glyph[rho] = run("step4c_positive_control", "fetal_planes",
                         tag=f"step4c_glyph/strength_{rho}",
                         overrides={"synthetic": {"enabled": True, "cue": "glyph", "strength": rho}})

# %%
# === CELL 4 / BLOCK B: revision matrix — per-class, 5 seeds, clean-trained ===
# Adds r_clean (so A-removed->A-removed exists), 5 seeds, and per-image
# predictions for the per-class breakdown.

res_rev = {}
for ds in ("fetal_planes", "bus_bra"):
    res_rev[ds] = run("revision_matrix", ds, tag="revision_matrix", folds=(0, 1, 2, 3, 4))

# %%
# === CELL 5 / BLOCK C: HC18 + MMOTU — the four-dataset claim ================
# HC18 at 28.9% annotation prevalence is the dose-response evidence and is
# currently unused. MMOTU is the declared multi-class stress case.
# NOTE: both need their packs uploaded; skipped cleanly if absent.

res_more = {}
for ds in ("mmotu", "hc18"):
    res_more[ds] = run("revision_matrix", ds, tag="revision_matrix", folds=(0, 1, 2))

# %%
# === CELL 6 / BLOCK D: 448 px replicate — the resolution confound ===========
# At 224 px a line of burnt-in text is ~6 px tall. Until this runs, the null on
# the text channel is a property of the resize step. Needs the 448 pack
# uploaded (~4.9 GB); skipped cleanly if absent.

res_448 = run("revision_matrix", "fetal_planes", tag="revision_matrix_448",
              size=448, folds=(0, 1, 2),
              overrides={"model": {"image_size": 448}, "train": {"batch_size": 24}})

# %%
# ===================== CELL 7: save everything =============================
import json

manifest = {}
for name, obj in [("glyph", res_glyph), ("revision", res_rev), ("more", res_more)]:
    for k, f in (obj.items() if isinstance(obj, dict) else [("", obj)]):
        if len(f):
            manifest[f"{name}_{k}"] = len(f)
if len(res_448):
    manifest["res448"] = len(res_448)

(OUT_ROOT / "REVISION_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
print(f"\nDownload the whole {OUT_ROOT} folder from the Output tab.")
