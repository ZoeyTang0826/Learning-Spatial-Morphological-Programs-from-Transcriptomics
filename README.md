# Learning Spatial Morphological Programs from Transcriptomics

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
  - [Patch-seq](#patch-seq)
  - [Xenium Human Breast Cancer](#xenium-human-breast-cancer)
- [Workflow](#workflow)
- [Models](#models)
  - [Unimodal Models](#unimodal-models)
  - [Multimodal Models](#multimodal-models)
- [Key Results](#key-results)
- [Main Findings](#main-findings)
- [Authors](#authors)

---

## Introduction

Cell morphology contains structural information about cell identity that may not be fully captured by gene expression alone. This project investigates whether morphology can serve as a complementary modality for representation learning and improve the biological structure of learned single-cell representations.

We compare unimodal and multimodal approaches for integrating **gene expression** and **cell morphology**, including dimensionality-reduction baselines, β-VAEs, latent fusion, multimodal VAEs, and shared-private latent models.

---

## Datasets

### Patch-seq

Paired single-cell measurements of:

- Gene expression
- Neuronal morphology
- RNA cell-type annotations

After preprocessing and multimodal alignment:

- **644 cells**
- **3,000 highly variable genes**
- **23 curated morphology features**

Morphology captures properties including laminar depth, dendritic extent, branching complexity, and arbor polarity.

### Xenium Human Breast Cancer

Spatial transcriptomics dataset containing approximately:

- **167,000 cells**
- **313 genes**
- Cell and nucleus segmentation masks
- Spatial coordinates

After quality control and morphology-variance filtering, approximately **33,000 cells** were retained. Cell and nuclear morphology features were extracted from segmentation masks using `scikit-image`. 

---

## Workflow

```text
                     Raw Data
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Gene Expression         Morphology
             │                     │
       Preprocessing          Preprocessing
             │                     │
             ▼                     ▼
         Gene VAE           Morphology VAE
             │                     │
             └──────────┬──────────┘
                        ▼
              Multimodal Integration
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
 Weighted Latent      MMVAE       Shared-Private
     Fusion                           VAE
        │               │               │
        └───────────────┴───────────────┘
                        ▼
              Representation Evaluation
                        │
             ARI / NMI / KNN / R²
```
---

## Models

### Unimodal Models

- Gene PCA / UMAP
- Morphology PCA / UMAP
- Gene β-VAE
- PCA-preprocessed Gene VAE
- Morphology β-VAE

### Multimodal Models

- **Weighted Latent Fusion**
- **Mixture-of-Experts Multimodal VAE (MMVAE)**
- **Shared-Private Multimodal VAE**

The shared-private model separates each modality into shared and modality-specific latent components, with explicit alignment between shared representations.

---

## Key Results

Multimodal integration improved cell-type representation compared with unimodal baselines.

| Representation | Fine ARI | Fine NMI | Coarse ARI |
|---|---:|---:|---:|
| Gene PCA VAE | 0.46–0.52 | 0.68–0.73 | 0.48–0.57 |
| **Weighted Latent Fusion** | **0.637** | **0.747** | 0.606 |
| MMVAE | 0.556 | 0.694 | 0.526 |
| Shared-Private VAE | 0.630 | 0.742 | 0.466 |

Weighted latent fusion achieved the strongest overall clustering performance, while the shared-private VAE incorporated stronger morphology-specific information into the latent representation.

---

## Main Findings

- Morphology contains biologically meaningful cell-type information beyond gene expression.
- Multimodal integration improves representation quality over unimodal models.
- Weighted latent fusion achieved the strongest overall clustering performance.
- Shared-private modeling captured a larger contribution from morphology than MMVAE.
- Spatial and non-spatial morphology contribute differently to fine- and coarse-grained cell-type organization.

---

## Authors

**Keyi Tang, Hunter Qin**  
Vanderbilt University  
Mentor: **Prof. Hirak Sarkar**
