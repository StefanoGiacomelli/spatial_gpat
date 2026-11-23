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

## Integrity and provenance

* `docs/checksums.sha256` lists the SHA-256 of every checkpoint, `hparams.yaml`, metrics JSON and experiment-state file of the
  archive; verify a restored copy from the repository root with `python analysis/maintenance/checksums.py --verify`.
* The configuration that was executed by each run is its stored `hparams.yaml`; `results/experiments/run_configs.csv` exports it for all
  69 runs (`python analysis/experiments/export_run_configs.py --check` compares the ledger with the files restored here).
* The archive may contain `last.ckpt` / `last-v1.ckpt` files of interrupted runs; for six runs they are byte-identical to the best-epoch
  checkpoint (see the comments at the end of `docs/checksums.sha256`) and can be deleted without loss.
