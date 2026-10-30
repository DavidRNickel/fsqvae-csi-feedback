# Generative precoding

This repository holds the code to reproduce the paper's sum-rate figures. It compares
learned feedback quantizers (FSQ-VAE and VQ-VAE) against classical channel-quantization
baselines for downlink multi-user MIMO precoding. All schemes are evaluated on two channel
models:

- **DeepMIMO** (`asu_campus_3p5`): real, ray-traced channels.
- **GMM**: a synthetic K-component Gaussian mixture.

| Label in figures | Method                                          | DeepMIMO script            | GMM script                     |
|------------------|-------------------------------------------------|----------------------------|--------------------------------|
| FSQ / FSQM       | FSQ-VAE, DFT / learned pilot matrix             | `run_fsq.sh`               | `gmm/run_fsq.sh`               |
| VQ / VQM         | VQ-VAE, DFT / learned pilot matrix              | `run_vq.sh`                | `gmm/run_vq.sh`                |
| CVQ              | K-means clustered vector quantization           | `baselines/run_cvq.sh`     | `gmm/run_cvq.sh`               |
| DVQ              | rotated DFT codebook                            | `baselines/run_dft.sh`     | `gmm/run_dft.sh`               |
| KVQ              | KLT + reverse water-filling transform coding    | `baselines/run_transform_vq.sh` | `gmm/run_transform_vq.sh` |

Every script is under `precoder/` and sweeps the full set of configurations used in the
figures. Each one `cd`s to its own directory, so it can be launched from anywhere.

## Layout

```
precoder/
├── main.py, main_vq.py          # FSQ-VAE / VQ-VAE training on DeepMIMO channels
├── fsqvae.py, vqvae.py, enc_dec_modules.py, common_setup.py
├── make_channels.py             # downloads the DeepMIMO scenario and writes channel_data/
├── experiment_grid.sh           # shared (N, J, Td) grid + result-directory checks for the run scripts
├── run_fsq.sh, run_vq.sh
├── baselines/                   # CVQ / DVQ / KVQ on DeepMIMO channels
├── gmm/                         # every method on GMM channels
│   └── channel_stats/           # fixed GMM component covariances (keep these: they define the channel model)
└── analysis/
    ├── aggregate_results.py     # scans all result directories -> aggregated_results.csv
    ├── aggregated_results.csv   # the table behind the paper figures
    └── make_paper_figures.py    # aggregated_results.csv -> analysis/photos/*.png
```

## Requirements

Tested with Python 3.12. Install the dependencies with:

```
pip install -r requirements.txt
```

The learned methods (`main.py`, `main_vq.py`) are slow on a CPU, so use a CUDA GPU for those.
The baselines run on the CPU.

## Reproducing the figures

### Figures from the shipped table

`precoder/analysis/aggregated_results.csv` already contains every number plotted. To redraw
the figures from it:

```
cd precoder/analysis
python3 make_paper_figures.py --skip-latent   # sum-rate figures -> photos/
```

The FSQ-VAE latent-space figure (`fsqvae_latent_space_dm_Tu12.png`) needs a trained
checkpoint. Running `run_fsq.sh` produces it in `results_fsq_eval/`. With the checkpoint in
place, run `make_paper_figures.py` without `--skip-latent`, or with `--latent-only` to build
just that figure.

### Full pipeline from scratch

1. **Generate the DeepMIMO channels.** Run this once for each antenna count:

   ```
   cd precoder
   for N in 4 8 32; do python3 make_channels.py $N; done
   ```

   This writes `channel_data/asu_campus_3p5/{N}x1/` with the channel, correlation and
   correlation square-root arrays. The GMM experiments need no data generation: they sample
   channels from `gmm/channel_stats/`.

2. **Run the experiments.** Run any of the scripts in the table above, for example:

   ```
   ./precoder/run_fsq.sh
   ./precoder/baselines/run_cvq.sh
   ./precoder/gmm/run_cvq.sh
   ```

   Each script lists its parameter grid at the top. Output goes to:
   - `results_fsq_<run>/` and `results_vq_<run>/` for the learned methods. Runs `_1`, `_2`
     and `_3` are independent trainings that are averaged, and the error bars show their
     standard deviation.
   - `wmmse_results/` for the baselines.

   CVQ caches its K-means codebooks under `wmmse_results/.../centroids/`, so later runs
   reuse them.

3. **Aggregate and plot.**

   ```
   cd precoder/analysis
   python3 aggregate_results.py     # rewrites aggregated_results.csv
   python3 make_paper_figures.py
   ```

   `aggregate_results.py` skips any method that has no results yet, so a subset of the
   experiments still produces a valid (partial) table.

Training seeds are drawn at random unless you pass `--seed`, so rerun numbers will differ
slightly from the shipped table. They should stay within the plotted error bars.
