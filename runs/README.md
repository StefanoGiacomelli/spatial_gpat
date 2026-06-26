# Training Artifacts Setup

Training checkpoints, TensorBoard event files, metrics, and experiment-state
snapshots are stored outside the Git history to keep the repository lightweight.
The `runs/` directory itself is tracked so that the expected location is always
present after cloning.

The full generated training state can be downloaded from the public OneDrive
archive and decompressed directly inside this directory:

<https://univaq-my.sharepoint.com/:u:/g/personal/stefano_giacomelli_graduate_univaq_it/IQCTGsPS9Z96ToXa2FjZcx4aAQE0tP41cphDyOrClWQh-Mg?e=Stcfg2>

| Directory | Contents | Used by |
| --- | --- | --- |
| `at2seld_nas/` | AT2SELD checkpoints, metrics, state files, and TensorBoard logs. | `main_nas-seld.py`, diagnostic scripts, cross-dataset analyses. |
| `standalone_seld/` | Standalone Conformer/ResNet SELD checkpoints, metrics, state files, and TensorBoard logs. | `main_standalone_seld.py`, threshold and DOA diagnostics. |

## Requirements

- **Free disk space:** depends on the archive version; around 30GiB.
- **Tools:** a browser or command-line downloader, an archive extractor
  compatible with `.zip` files.
- **Repository state:** clone the repository first, then extract the archive
  into `runs/`.

## Restore the local training state

Download the archive from the OneDrive link above, then extract its contents
inside this directory:

```bash
# from the repository root
cd runs
unzip /path/to/downloaded_archive.zip
```

## Expected result

After extraction, the repository should contain the same training-artifact
layout used by the analysis scripts:

```text
runs/
├── README.md
├── at2seld_nas/
│   ├── checkpoints/
│   ├── metrics/
│   └── ...
└── standalone_seld/
    ├── checkpoints/
    ├── metrics/
    └── ...
```

The exact subdirectories structure refers to all framework experiments.
Existing analysis presets and diagnostic scripts expect checkpoint
paths under these two top-level folders.
