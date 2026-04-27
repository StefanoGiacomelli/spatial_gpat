# Data Setup

Automated setup for the three datasets used in this project:

| Directory | Dataset | Source |
|---|---|---|
| `audioset-qiuqiangkong/` | AudioSet (Strong labels, qiuqiangkong split) | SharePoint (manual download) |
| `STARSS23/` | STARSS23 (Sony-TAU Realistic Spatial Soundscapes) | Zenodo 7880637 |
| `TAU-NIGENS2021/` | TAU-NIGENS Spatial Sound Events 2021 | Zenodo 5476980 |

## Requirements

- **Free disk space:** at least **200 GB** (peak usage during extraction). The script checks this upfront and aborts if insufficient.
- **Tools:** `curl`, `tar`, `unzip` (all standard on macOS and most Linux distros).

## Step 1 — Download AudioSet manually

AudioSet cannot be downloaded automatically. Download `audioset_bundle.tar` from SharePoint and place it in this directory (`data/`) before running the setup script:

[https://univaq-my.sharepoint.com/:u:/g/personal/stefano_giacomelli_graduate_univaq_it/IQASIKNipYfbTofBl6lLXEQwAVS9xgAoBjf6smTUXvXMWmY?e=bRGUD4](https://univaq-my.sharepoint.com/:u:/g/personal/stefano_giacomelli_graduate_univaq_it/IQASIKNipYfbTofBl6lLXEQwAVS9xgAoBjf6smTUXvXMWmY?e=bRGUD4)

The file must be named exactly `audioset_bundle.tar` and placed inside `data/`.

## Step 2 — Run the setup script

```bash
bash data/setup_data.sh
```

The script will:

1. Verify that at least 200 GB of free space is available.
2. Extract `audioset_bundle.tar` into `audioset-qiuqiangkong/`.
3. Download and extract STARSS23 from Zenodo.
4. Download and extract TAU-NIGENS2021 from Zenodo.
5. Remove all downloaded archives (`.tar`, `.zip`) after successful extraction.

## Expected result

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
│   ├── mic_dev/
│   │   └── (same splits as foa_dev)
│   ├── metadata_dev/
│   │   └── (same splits as foa_dev)
│   ├── foa_eval/
│   └── mic_eval/
└── TAU-NIGENS2021/
    ├── foa_dev/
    │   ├── dev-train/
    │   ├── dev-val/
    │   └── dev-test/
    ├── mic_dev/
    │   └── (same splits as foa_dev)
    ├── metadata_dev/
    │   └── (same splits as foa_dev)
    ├── foa_eval/
    │   └── eval-test/
    ├── mic_eval/
    │   └── eval-test/
    └── metadata_eval/
```
