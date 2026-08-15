# Running the Step-4 gate on Kaggle

End-to-end, from this machine to a gate verdict. Budget ~2 h of your attention
(most of it upload) plus ~1–2 h of GPU that runs unattended.

Local paths below assume:

```bash
cd <repo>
PY=<repo>/.venv/bin/python
```

---

## Part 1 — Local (CPU), ~10 minutes

### 1.1 Confirm Stages A and B are current

Already done for all four datasets, but re-check after any pipeline change:

```bash
$PY -m asa package --dataset fetal_planes --size 224 --workers 10
```

The staging script in step 1.3 **refuses to upload** if masks are newer than the
pack, so you cannot ship shards that disagree with the masks. If it stops you,
re-run `asa package` for the named dataset.

### 1.2 (Recommended, not blocking) Verify masks

The gate will run without this, but no number from it should be reported before
it is done:

```bash
$PY -m asa verify-export --dataset fetal_planes --fraction 0.05
# two people score work/fetal_planes/verification/verification_sheet.csv
$PY -m asa verify-score  --dataset fetal_planes
```

### 1.3 Stage the two Kaggle datasets

```bash
$PY scripts/prepare_kaggle_upload.py --user <your-kaggle-username>
```

Two datasets, on purpose — a Kaggle dataset version re-uploads everything it
contains, and the code changes constantly while 4.5 GB of shards do not:

| Folder | Contents | Size |
|---|---|---|
| `kaggle_upload/asa-code` | `src/`, `configs/`, `tests/` | ~200 KB |
| `kaggle_upload/asa-packs` | `pack_224/` for all four datasets | ~4.5 GB |

For the gate you only need two datasets, so you can cut the upload to ~3.5 GB:

```bash
$PY scripts/prepare_kaggle_upload.py --user <you> --datasets fetal_planes bus_bra
```

### 1.4 Upload

Install the CLI **into the project venv**, not system Python. Debian marks the
system interpreter externally managed (PEP 668), so a bare `pip install` fails
with `externally-managed-environment`, and `--break-system-packages` is exactly
what its name says:

```bash
$PY -m pip install kaggle
```

That puts a `kaggle` shim in `.venv/bin/`. Either call it by full path, or add
the venv to your PATH for the session:

```bash
export PATH=<repo>/.venv/bin:$PATH
```

Get `kaggle.json` from **kaggle.com → your avatar → Settings → API → Create New
Token**, then:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

Check it authenticates before starting a multi-GB upload:

```bash
$PY -m kaggle datasets list --mine
```

Then run the two `kaggle datasets create` commands the script printed. They
create **private** datasets. The packs upload will take a while — start it and
leave it.

Later changes use `kaggle datasets version -p <folder> -m "message"` instead of
`create`, and Kaggle notebooks pin dataset versions, so after pushing a new
version you must **re-add the input** in the notebook to pick it up.

---

## Part 2 — Kaggle notebook setup, ~5 minutes

### 2.1 Create the notebook

kaggle.com → **Create → New Notebook**.

### 2.2 Add both datasets as inputs

Right panel → **Add Input** → **Your Datasets** → add:
- `annotation-shortcut-audit-code`
- `asa-packs-224`

They mount read-only under `/kaggle/input/<slug>/`.

### 2.3 Notebook settings — the three that matter

Right panel → **Settings**:

| Setting | Value | Why |
|---|---|---|
| **Accelerator** | **GPU T4 x2** (or P100) | Without it the run is unusably slow. Only one GPU is used; T4 x2 is just what Kaggle offers. |
| **Internet** | **ON** | torchvision downloads ImageNet weights *at model-build time*. With internet off the run dies minutes in, not at the start. See 2.4 if you can't enable it. |
| **Persistence** | Files only (optional) | Lets `/kaggle/working` survive between interactive sessions, so a resumed run keeps its results. |

Enabling internet requires a phone-verified Kaggle account.

### 2.4 If you cannot turn internet on

Two options:

- **Preferred:** create a third small Kaggle dataset containing
  `resnet50-11ad3fa6.pth` (torchvision's ResNet-50 `IMAGENET1K_V2` weights),
  add it as an input. The notebook automatically copies any `.pth` under
  `/kaggle/input` into `TORCH_HOME` and reports what it staged.
- **Last resort:** set `model.pretrained: none` in
  `configs/experiments/step4_gate.yaml`. This changes the experiment — a
  randomly initialised backbone is a different question — so only do it as a
  smoke test, never for a reported result.

---

## Part 3 — Run the gate

### 3.1 Paste the notebook

Open `notebooks/step4_gate.py` locally. It is a jupytext-style script: the
`# %%` markers are cell boundaries. Either paste each block into its own Kaggle
cell, or convert first:

```bash
pip install jupytext && jupytext --to ipynb notebooks/step4_gate.py
```

and upload the resulting `.ipynb` via **File → Import Notebook**.

No path editing is needed. The notebook detects Kaggle, finds `src/asa` and the
packages under `/kaggle/input`, and writes to `/kaggle/working`.

### 3.2 Run the first three cells interactively

You are checking three things before spending GPU time:

1. **Environment cell** — prints the repo path and lists the discovered
   packages. If a package is missing, the input was not added.
2. **Weights cell** — prints `internet: ON`, or how many checkpoints it staged.
3. **Sanity cell** — asserts that `A-removed` changed pixels only inside the
   annotation mask and `A-only` preserved them. If either assertion fires,
   **stop**: the packaging is wrong and every downstream number would be
   meaningless.

Expect roughly `mask=0.8% of pixels` for FETAL_PLANES. A warning that mask
coverage has collapsed means the resolution is too low to measure anything.

### 3.3 Commit the real run

**Save Version → Save & Run All (Commit)**.

Do not run the training cell interactively and walk away — interactive sessions
die on idle. A committed run gets the full session.

The gate is 2 datasets × 3 folds × 2 seeds = 12 trainings, roughly 1–2 h on a
T4. Results are written after **every** trained model, and completed cells are
skipped on re-entry, so if a session is cut short just commit again and it
continues.

### 3.4 Collect the output

When the version finishes: **Output** tab → download
`runs/step4_gate/`. You want:

- `gate_report.md` — the verdict table and the decision
- `gate_decision.json` — the same, machine-readable
- `<dataset>/matrix_results.csv` — per-run metrics with bootstrap intervals

---

## Part 4 — Reading the verdict

`gate_report.md` says **PROCEED** or **STOP**. The rule was fixed in
`src/asa/train/gate.py` before any number existed:

- **A-only** must clear empirical chance by a pre-registered margin, judged on
  the bootstrap **lower bound**, not the point estimate;
- on **at least two datasets** — a shortcut seen on one annotation convention is
  a quirk, not a finding;
- and the **equal-area control** must *not* drop as much as annotation removal.
  If it does, the effect is pixel removal rather than annotations, and the
  dataset fails automatically. This is the pixel-removal objection and it refutes the
  hypothesis regardless of what A-only shows.

Three outcomes:

| Verdict | What it means | What to do |
|---|---|---|
| **PROCEED** | The shortcut exists. | Pre-register (proposal Step 5) *before* the full matrix, so H1–H4 and the analysis plan are fixed before the numbers are known. |
| **STOP**, A-only at chance | Hypothesis refuted for ~1 GPU-hour. | This is a publishable null — it retires a concern BUSClean stated and never tested. Or redirect to the alternative study. |
| **STOP**, control drops too | The effect is pixel removal. | Same as above. Do not tune the threshold to rescue it. |

What the gate does **not** establish: how much of the headline A-full number is
annotation-derived. That is the full matrix. Do not report a gate result as if
it were the audit.

---

## Part 5 — Feeding it back into the paper

```bash
# copy the downloaded results into runs/, then
$PY scripts/analyse_matrix.py runs/step4_gate/fetal_planes
$PY scripts/make_paper_assets.py          # auto-detects ../Manuscript
cd ../Manuscript && make && make check
```

`analyse_matrix.py` writes `matrix_cells.csv`, `hypothesis_tests.json` (Holm–
Bonferroni across the declared family) and `report.md`. Section 6.3 of the paper
is written to be filled from that report; `make check` lists the `\todo` markers
still outstanding.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: could not find src/asa` | Code dataset not added as an input | Add Input → your code dataset |
| `assert len(PACKS) >= 2` | Packs dataset missing, or only one dataset staged | Check inputs; re-stage with both datasets |
| `URLError` several minutes in | Internet off, no staged weights | Turn Internet ON, or add a weights dataset (2.4) |
| `NO GPU` in the torch cell | Accelerator not set | Settings → Accelerator → GPU |
| Session ends mid-run | 12-hour cap or idle timeout | Commit again; it resumes from `matrix_results.csv` |
| Sanity assertion fires | Packaging/mask mismatch | Re-run `asa package` locally, re-upload |
| `pack_224 is OLDER than the annotation features` | Masks re-run without repacking | `python -m asa package --dataset <name> --size 224` |
| `error: externally-managed-environment` | Bare `pip` hits Debian's system Python (PEP 668) | Use `$PY -m pip install ...`; never `--break-system-packages` |
| `kaggle: command not found` | The shim lives in `.venv/bin` | `export PATH=<repo>/../.venv/bin:$PATH`, or use `$PY -m kaggle` |
| `Authentication required to call the Kaggle API` | No `~/.kaggle/kaggle.json` | Create an API token and install it as in 1.4 |
| Results look identical across seeds | Resume skipped everything | Delete `runs/step4_gate/<dataset>/matrix_results.csv` to force a rerun |
