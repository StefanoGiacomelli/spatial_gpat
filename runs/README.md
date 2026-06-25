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
| `at2seld_nas/` | NAS-SELD / AT2SELD checkpoints, metrics, state files, and TensorBoard logs. | `main_nas-seld.py`, diagnostic scripts, cross-dataset analyses. |
| `standalone_seld/` | Standalone Conformer/ResNet SELD checkpoints, metrics, state files, and TensorBoard logs. | `main_standalone_seld.py`, threshold and DOA diagnostics. |

## Requirements

- **Free disk space:** depends on the archive version; keep enough space for
  checkpoints and TensorBoard logs.
- **Tools:** a browser or command-line downloader, plus an archive extractor
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

If the archive already contains a top-level `runs/` folder, extract it from the
repository root instead:

```bash
# from the repository root
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

The exact subdirectories depend on which experiments were included in the
archive. Existing analysis presets and diagnostic scripts expect checkpoint
paths under these two top-level folders.

## Git policy

Only this `README.md` is tracked. The generated contents of `runs/` remain
ignored by Git:

- checkpoints and TensorBoard event files can be large;
- experiment state is reproducible from the archive or by rerunning training;
- analysis code treats `runs/` as a read-only input archive.

Do not commit decompressed checkpoint or log files. Update this README if the
external archive URL or expected layout changes.
