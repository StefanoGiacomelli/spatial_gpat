# Data Setup

Automated incremental setup for the datasets used in this project.
Each component is downloaded / extracted only if not already present — re-running is safe.

| Directory | Dataset | Source |
|---|---|---|
| `audioset-qiuqiangkong/` | AudioSet (Strong labels, qiuqiangkong split) | SharePoint (manual download) |
| `STARSS23/` | STARSS23 (Sony-TAU Realistic Spatial Soundscapes) | Zenodo 7880637 |
| `TAU-NIGENS2021/` | TAU-NIGENS Spatial Sound Events 2021 | Zenodo 5476980 |
| `TAU2019/` | TAU Spatial Sound Events 2019 (DCASE2019 Task 3) | Zenodo 2599196 + 3377088 |

## Requirements

- **Free disk space:** ~100 GB for all three SELD datasets (FOA only); ~200 GB if AudioSet is included.
- **Tools:** `curl`, `zip`, `unzip` — standard on macOS and most Linux distros.  
  The script warns if any are missing.

## Step 1 — Download AudioSet manually

AudioSet cannot be downloaded automatically. Download `audioset_bundle.tar` from SharePoint and place it in this directory (`data/`) before running the setup script:

[https://univaq-my.sharepoint.com/:u:/g/personal/stefano_giacomelli_graduate_univaq_it/IQASIKNipYfbTofBl6lLXEQwAVS9xgAoBjf6smTUXvXMWmY?e=bRGUD4](https://univaq-my.sharepoint.com/:u:/g/personal/stefano_giacomelli_graduate_univaq_it/IQASIKNipYfbTofBl6lLXEQwAVS9xgAoBjf6smTUXvXMWmY?e=bRGUD4)

The file must be named exactly `audioset_bundle.tar` and placed inside `data/`.  
If the file is absent the script skips AudioSet with a warning.

## Step 2 — Run the setup script

```bash
# All datasets
bash data/setup_data.sh

# Single dataset only
bash data/setup_data.sh --dataset STARSS23
bash data/setup_data.sh --dataset TAU-NIGENS2021
bash data/setup_data.sh --dataset TAU2019
bash data/setup_data.sh --dataset audioset
```

The script checks each component before acting:

- If a directory already contains the expected audio / CSV files → **skip**.
- If a downloaded archive already exists locally → **skip download, go straight to extract**.
- Downloaded archives (`.zip`, `.z01`, `.z02`) are **kept** after extraction and act as local
  caches — re-running will not re-download them.

## Expected result (FOA only)

```
data/
├── audioset-qiuqiangkong/
│   ├── balanced_train_segments/
│   ├── eval_segments/
│   ├── metadata/
│   └── unbalanced_strong/
├── STARSS23/
│   ├── foa_dev/
│   │   ├── dev-train-sony/
│   │   ├── dev-train-tau/
│   │   ├── dev-test-sony/
│   │   └── dev-test-tau/
│   ├── foa_eval/
│   ├── metadata_dev/
│   │   ├── dev-train-sony/
│   │   ├── dev-train-tau/
│   │   ├── dev-test-sony/
│   │   └── dev-test-tau/
│   ├── LICENSE
│   └── README.md
├── TAU-NIGENS2021/
│   ├── foa_dev/
│   │   ├── dev-train/
│   │   ├── dev-val/
│   │   └── dev-test/
│   ├── foa_eval/
│   │   └── eval-test/
│   ├── metadata_dev/
│   │   ├── dev-train/
│   │   ├── dev-val/
│   │   └── dev-test/
│   ├── metadata_eval/
│   └── README.md
└── TAU2019/
    ├── dev/
    │   ├── foa_dev/             (split1–4, ~400 clips)
    │   ├── metadata_dev/        (split1–4 CSV files)
    │   ├── LICENSE
    │   └── README.md
    └── eval/
        ├── proj/asignal/DCASE2019/dataset/foa_eval/   (100 clips, split0)
        ├── metadata_eval/                              (split0 CSV files)
        ├── short2longname.txt
        └── README.md
```

> **Note — TAU2019 eval path:** the `foa_eval.zip` from Zenodo 3377088 unpacks into the
> nested path `proj/asignal/DCASE2019/dataset/foa_eval/`.  This is intentional and matches
> the path expected by `TAU2019Dataset._collect_eval()` in `datasets/tau2019.py`.
