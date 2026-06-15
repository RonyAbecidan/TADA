# Data

This folder documents the data used in the TADA paper. The README is organized in **two independent parts**, depending on what you want to reproduce.

| Part | Goal | Where to get it |
|------|------|-----------------|
| **1 — Toy sharpen experiment** | Run TADA on the controlled sharpening target from the paper | HDF5 files on [Zenodo](https://doi.org/10.5281/zenodo.20688394) |
| **2 — Operational experiments** | Rebuild the real-world Flickr camera bases (CANON, NIKON, SONY) | CSV metadata files in this folder |

If you only want to **train TADA with the toy protocol** (`python pipeline_learning_config.py sharpen100_bpnzac.yaml cuda`), start with Part 1. If you want to **reproduce the operational steganalysis experiments** on Flickr images, continue to Part 2 after downloading the camera lists below.

---

## Part 1 — Toy sharpen experiment (Zenodo)

The toy experiment is a simplified setting where the target pipeline is known in advance: images are sharpened with a fixed 3×3 kernel, JPEG-compressed at quality factor 100, and fully embedded with UERD at 1 bpnzac. TADA must learn to emulate this sharpening from a generic RAW source.

The preprocessed HDF5 files are **not in git** (too large). Download them from Zenodo:

**[TADA toy sharpen experiment — HDF5 training data](https://doi.org/10.5281/zenodo.20688394)**  
DOI: [10.5281/zenodo.20688394](https://doi.org/10.5281/zenodo.20688394)

Place the files as follows:

```text
TADA/
  color_raws_512.hdf5
  targets/
    sharpen_full_stego.hdf5
```

Both HDF5 archives were built from the **same fixed pool of 2,000 RAW images** randomly sampled from [ALASKA#2](https://utt.hal.science/hal-02950094) (Cogranne, Giboulot, Bas, WIFS 2020). They are derivatives of ALASKA#2, not the benchmark itself. Please cite the [Zenodo dataset](../README.md#zenodo-dataset-hdf5-files) and [ALASKA#2](../README.md#alaska2-upstream-benchmark) when using these data.

### `color_raws_512.hdf5` — source images

Expected key:

```text
train
```

Contains **2,000** color TIF crops (`512×512`) from the ALASKA#2 subset above (demosaicked with `amaze`), chosen to be as spatially uniform as possible.

At training time, the code randomly samples `training.n_samples` TIFs from this pool (default: **500** in `sharpen100_bpnzac.yaml`), then extracts random **`256×256`** crops (`im_size_source`) to limit GPU memory. Losses are computed on **`8×16` KB residual patches** sampled inside those crops — the crop size does not define the learning granularity.

These crops are intentionally **relatively uniform**, but not perfectly flat: fully flat regions would carry little residual information. The selection keeps enough structure for TADA to estimate the pipeline without being dominated by highly textured content.

After TADA training, the paper builds the final steganalysis source from the **same 2,000 RAWs**, but with **smart crops** (ALASKA procedure) instead of uniform ones, so developed images better match operational target textures.

### `targets/sharpen_full_stego.hdf5` — sharpened target

Expected keys:

```text
operational
pmap_ope
eval
pmap_eval
```

Built from the **same 2,000-RAW ALASKA#2 subset**: developed into `512×512` crops, converted to grayscale, sharpened with the kernel below, JPEG QF 100, fully embedded with UERD at 1 bpnzac (`full_stego` scenario).

Sharpen kernel (Table 1 in the paper):

```text
  0.00  -0.25   0.00
 -0.25   2.00  -0.25
  0.00  -0.25   0.00
```

| Split | Size | Role |
|-------|------|------|
| `operational` + `pmap_ope` | 1,000 images | Unlabeled targets for TADA training (paper mentions 500; this repo keeps 1,000) |
| `eval` + `pmap_eval` | 1,000 images | Held-out set, disjoint from `operational` |

`pmap_ope` / `pmap_eval` are UERD embedding probability maps: they down-weight strongly modified patches when estimating residual statistics. By default, `operational.n_samples: 1000` uses the full operational pool.

---

## Part 2 — Operational camera bases (CANON, NIKON, SONY)

The operational experiments use **real Flickr images** from three camera-specific databases. These are the hardest targets for TADA in the paper: each base reflects a distinct in-camera pipeline and JPEG compression habit, unlike the controlled ALASKA#2 toy setting.

This folder ships **metadata lists** so you can download the same images and rebuild the operational HDF5 targets on your side. No pre-built HDF5 is provided for these bases.

| Base | File | Source | Flickr user |
|------|------|--------|-------------|
| CANON | `user_canon.csv` | [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) | [Andy E. Nystrom](https://www.flickr.com/photos/24917258@N05/) |
| NIKON | `user_nikon.csv` | [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) | [NR Acampamentos](https://www.flickr.com/photos/28004289@N03/) |
| SONY | `user_sony.csv` | Image links only (metadata lost) | [Tom](https://www.flickr.com/photos/tomstravelscom/) |

CANON and NIKON lists were curated from [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) ([Thomee et al., 2016](https://doi.org/10.1145/2812802)) by selecting a **single Flickr user** who consistently shoots with **one camera model**. Please cite [YFCC100M](../README.md#yfcc100m-upstream-metadata) when using these lists.

### `user_canon.csv` (~33k images)

| Field | Role |
|-------|------|
| `username` | Flickr display name (`Andy E. Nystrom`) |
| `model` | EXIF camera model (mainly **Canon PowerShot SX30 IS**) |
| `url` | Flickr photo page |
| `picture` | Direct image URL |

### `user_nikon.csv` (~21k images)

| Field | Role |
|-------|------|
| `username` | Flickr display name (`NR Acampamentos`) |
| `model` | EXIF camera model (mainly **Nikon D40**) |
| `url` | Flickr photo page |
| `picture` | Direct image URL |

### `user_sony.csv` (1,000 images)

| Field | Role |
|-------|------|
| `id` | Flickr photo identifier |
| `picture` | Direct image URL |
| `url` | Flickr photo page for metadata (`https://www.flickr.com/photo.gne?id=<id>`) |

Sony pictures are from the user "toms_travel" [Tom's photostream](https://www.flickr.com/photos/tomstravelscom/). Use the `url` field of the .csv to fetch EXIF, license, and other metadata (e.g. [photo 10560566714](https://www.flickr.com/photo.gne?id=10560566714)).

### Getting started with the CSV files

1. Download JPEGs from the `picture` column.
2. Keep only images from the target camera model (for CANON and NIKON, filter on the `model` column).
3. Make your own HDF5 to launch TADA.

The CANON and NIKON CSV column layout follows the [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/) metadata format; unnamed numeric columns are kept as in the original export.
