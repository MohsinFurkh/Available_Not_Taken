# Obtaining the datasets

None of the four datasets are redistributed here. Each carries its own licence
and access conditions, and each must be obtained from its authors. Once
downloaded, set the `root:` field in the matching `configs/datasets/*.yaml`.

| Dataset | Task | Images / groups | Source |
|---|---|---|---|
| **FETAL_PLANES_DB** | 6-class fetal standard plane | 12,400 / 1,792 patients | Burgos-Artizzu et al., *Sci Rep* 2020 — Zenodo record 3904280 |
| **HC18** | fetal head-circumference regression | 999 labelled | van den Heuvel et al., *PLOS ONE* 2018 — grand-challenge.org |
| **BUS-BRA** | breast benign/malignant | 1,875 / 1,064 women | Gómez-Flores et al., *Med Phys* 2024 |
| **MMOTU-2D** | 8-class ovarian tumour subtype | 1,469 | Zhao et al., 2022 — github.com/cv516Buaa/MMOTU_DS2Net |

## Expected layout

The manifest readers in `src/asa/data/manifest.py` expect each dataset in the
form its authors distribute it. No reorganisation is needed; point the config at
the extracted directory.

```
FETAL_PLANES_DB_data/   FETAL_PLANES_DB_data.csv, Images/
HC18 Dataset/           training_set_pixel_size_and_HC.csv, training_set/
BUS-BRA/                bus_data.csv, Images/, Masks/
MMOTU/                  OTU_2d/{images,annotations,train_cls.txt,val_cls.txt}
```

## Annotation masks

The detected masks (~490 MB across the four datasets) are **not** included here:
they are derived from licensed images and are larger than a code repository
should carry. They regenerate in roughly 15 minutes on a multi-core CPU:

```bash
python -m asa manifest --dataset <name>
python -m asa masks    --dataset <name> --workers 10
```

Detection is deterministic given the config in `configs/`, so regenerated masks
are identical to those used in the paper.

## A note on grouping

HC18 and MMOTU-2D publish no subject identifier for their released 2D images, so
splitting for those two is at file level. This is stated rather than papered
over, and MMOTU-2D is treated as supporting evidence rather than a primary arm.
