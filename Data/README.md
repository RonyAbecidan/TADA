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

## Operational camera bases (CANON, NIKON, SONY)

The paper's operational experiments include three camera-specific target databases — **CANON**, **NIKON**, and **SONY** extracted from [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) ([Thomee et al., 2016](https://doi.org/10.1145/2812802); see [upstream citation](../README.md#yfcc100m-upstream-metadata)). This folder provides the metadata needed to **reconstruct the CANON and NIKON subsets** from Flickr.

### Selection criteria

Each base was built with care by identifying, within [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/), a **single Flickr user** who consistently uses **one camera model**.

### `user_canon.csv`

Metadata export for the **CANON** operational base (~33k images).

| Field | Role |
|-------|------|
| `username` | Flickr display name (`Andy E. Nystrom`) |
| `model` | EXIF camera model (mainly **Canon PowerShot SX30 IS**) |
| `url` | Flickr photo page |
| `picture` | Direct image URL (download entry point) |

Flickr user: [24917258@N05](https://www.flickr.com/photos/24917258@N05/)

### `user_nikon.csv`

Metadata export for the **NIKON** operational base (~21k images).

| Field | Role |
|-------|------|
| `username` | Flickr display name (`NR Acampamentos`) |
| `model` | EXIF camera model (mainly **Nikon D40**) |
| `url` | Flickr photo page |
| `picture` | Direct image URL |

Flickr user: [28004289@N03](https://www.flickr.com/photos/28004289@N03/)

### SONY base (metadata not available)

The original **SONY** image list was lost and is **not** included in this repository. The images used in the paper nevertheless come from the Flickr photostream of **Tom** ([tomstravelscom](https://www.flickr.com/photos/tomstravelscom/)).

It is possible to **rebuild an equivalent SONY base** from the same author by applying the same selection protocol (single user, single camera model) on more recent Sony bodies present in that photostream.

### Rebuilding HDF5 targets from the CSV files

1. Download JPEGs from the `picture` URLs (respect Flickr / Creative Commons licenses listed in each row).
2. Keep only images from the paper's target camera model.

The CSV column layout follows the [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) metadata format; unnamed numeric columns are kept as in the original export. Please cite [YFCC100M](../README.md#yfcc100m-upstream-metadata) when using these lists.
