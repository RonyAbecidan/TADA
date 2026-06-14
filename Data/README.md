# Data Files

The training command expects the HDF5 files below. They are **gitignored** and must be downloaded from Zenodo:

**[TADA toy sharpen experiment — HDF5 training data](https://doi.org/10.5281/zenodo.20688394)**  
DOI: [10.5281/zenodo.20688394](https://doi.org/10.5281/zenodo.20688394)

Both files were **built from the same fixed pool of 2,000 RAW images randomly sampled from [ALASKA#2](https://utt.hal.science/hal-02950094)** (Cogranne, Giboulot, Bas, WIFS 2020). They are not the ALASKA#2 benchmark itself. Please cite the [Zenodo dataset](../README.md#zenodo-dataset-hdf5-files) and [ALASKA#2](../README.md#alaska2-upstream-benchmark) when using these data.

```text
TADA/
  color_raws_512.hdf5
  targets/
    sharpen_full_stego.hdf5
```

## `color_raws_512.hdf5`

Expected key:

```text
train
```

This dataset contains **2,000** color TIF crops (`512x512`) from that **2,000-RAW ALASKA#2 subset** (demosaicked with `amaze`), chosen to be as spatially uniform as possible. The paper experiment shows that **500 source images are sufficient**: at training time, the code randomly samples `training.n_samples` TIFs from this pool (default: `500` in `sharpen100_bpnzac.yaml`), then extracts random **`256x256`** crops from each image (`im_size_source`) for memory reasons. TADA losses are computed on **`8x16` KB residual patches** sampled from those crops, so the crop size does not define the learning granularity.

These crops are intentionally **relatively uniform**, but not perfectly flat. Fully uniform regions would produce residuals with little discriminative power, so the selection keeps enough structure to estimate the pipeline while limiting the influence of overly textured areas during TADA training.

Once TADA has learned the pipeline, the paper builds the final steganalysis training source from the **same 2,000-RAW ALASKA#2 subset**, but using **smart crops** (ALASKA procedure) instead of uniform crops, so the developed images carry richer textures and better match operational target content.

## `targets/sharpen_full_stego.hdf5`

Expected keys:

```text
operational
pmap_ope
eval
pmap_eval
```

This file corresponds to the toy sharpen target from the paper. It is built from the **same 2,000-RAW ALASKA#2 subset** (developed into `512x512` crops, converted to grayscale, sharpened with the paper kernel below, JPEG-compressed at quality factor 100, and fully embedded with UERD at 1 bpnzac, `full_stego` scenario).

Sharpen kernel used in the paper (Table 1):

```text
  0.00  -0.25   0.00
 -0.25   2.00  -0.25
  0.00  -0.25   0.00
```

`operational` contains **1,000** unlabeled target images used to train TADA (the paper toy protocol mentions 500; this repository keeps 1,000 operational / 1,000 eval). The remaining **1,000** images are stored under `eval` (with `pmap_eval`) and are kept disjoint from the operational set to avoid optimistic performance estimates due to overlap between training and evaluation. `pmap_ope` contains the corresponding UERD embedding probability maps used to reduce the influence of strongly modified patches during residual-statistics estimation. At training time, the code uses at most `operational.n_samples` images from the operational pool (default: `1000` in `sharpen100_bpnzac.yaml`).

## Flickr / YFCC100M List

`flickr_yyc100m_images.txt` is a placeholder for Flickr image identifiers from YFCC100M used in the paper's operational experiments. Replace it with the provided list if you want to reconstruct the same subset.
