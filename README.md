# [ICLR'26] A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models

## 🚀 News 
- **January 31, 2026**: Evaluation codes for A-TPT are released.
- **January 26, 2026**: Paper accepted at ICLR 2026! 🎉
- **October 29, 2025**: Paper is available on arXiv! 📄

For more details, please feel free to check out our:

[![Paper](https://img.shields.io/badge/Paper-Published-blue)](https://arxiv.org/pdf/2510.26441.pdf) [![arXiv](https://img.shields.io/badge/arXiv-2503.12096-b31b1b.svg)](https://arxiv.org/abs/2510.26441) [![Project Page](https://img.shields.io/badge/%F0%9F%94%97-Project%20Page-blue)](https://mb-shihab-aaqil-ahamed.github.io/A-TPT/)

This repository provides the official PyTorch implementation of our ICLR 2026 paper:
> A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models
> Authors: [Shihab Aaqil Ahamed](https://mb-shihab-aaqil-ahamed.github.io/), [Udaya S.K.P. Miriya Thanthrige](https://scholar.google.com/citations?user=vuOC9RYAAAAJ&hl=en), [Ranga Rodrigo](https://ranga.staff.uom.lk/), [Muhammad Haris Khan](https://m-haris-khan.com/)

Our major contributions are summarized as follows:

*   We introduce a numerical optimization method, called `A-TPT`, for better calibration of test-time prompt tuning for VLMs. This resolves the suboptimal performance of existing leading calibration techniques for test-time prompt tuning.
*   We introduce novel angular diversity that effectively promotes the diversity among textual features, thereby improving the calibration capabilities of VLMs when $N > |D|$ and $N < |D|$. This is accomplished by maximizing the minimum pairwise angular distance between normalized textual features.
*   We conduct extensive experiments to validate the generalizability of our approach on different datasets, including medical datasets, across various baselines. The results show that `A-TPT` surpasses state-of-the-art methods in calibration performance. We also provide thorough analyses, including theoretical aspects. Moreover, our approach provides superior calibration compared to the zero-shot CLIP model, which reveals improved calibration.

## Angular Diversity (AD) vs. Expected Calibration Error (ECE)

| <img src="figures/hard_prompt.png" width="100%" /> | <img src="figures/tuned_prompt.png" width="100%" /> |
|:---:|:---:|
| Hard Prompt | Tuned Prompt |

## Comparison with C-TPT and O-TPT

| <img src="figures/radar_plot.png" width="100%" /> | <img src="figures/sketch.png" width="100%" /> |
|:---:|:---:|

## Installation
```bash
# Clone this repo
git clone https://github.com/MB-Shihab-Aaqil-Ahamed/A-TPT.git
cd A-TPT

# Create a conda enviroment
conda env create -f environment.yml
conda activate atpt
```

## Datasets
We evaluate our method (A-TPT) on fine-grained and natural distribution shift datasets:

- For fine-grained classification, we consider 11 datasets:
    * ImageNet
    * Flower102
    * OxfordPets
    * SUN397
    * DTD
    * Food101
    * StanfordCars
    * Aircraft
    * UCF101
    * EuroSAT
    * Caltech101

- For natural distribution shift, we consider 4 datasets:
    * ImageNet-V2
    * ImageNet-A
    * ImageNet-R
    * ImageNet-Sketch

Prepare the datasets based on the following GitHub repository [TPT](https://github.com/azshue/TPT).

## Experiments

In each of the bash script .sh files, change the {data_root} accordingly. And, you can change the CLIP pretrained backbone by modifying the {arch} parameter to either ‘RN50’ or ‘ViT-B/16’. Also, you can change baselines by modifying the {run_type} to either ‘tpt’ or ‘tpt_ts’ or ‘tpt_atpt’.

1. Baseline (CLIP)
```bash
bash scripts/test_baseline.sh {dataset}
```

2. Test-Time Prompt Tuning (TPT)
```bash
# for Fine-grained classification
bash scripts/test_tpt_fg.sh {dataset}

# for natural distribution shift
bash scripts/test_tpt_ds.sh {dataset}

# for temperature scaling experiments, change the run_type to tpt_ts in the .sh file.
```

3. Ours (A-TPT) 
```bash
# for Fine-grained classification
bash scripts/test_tpt_atpt_fg.sh {dataset}

# for natural distribution shift
bash scripts/test_tpt_atpt_ds.sh {dataset}
```

The command line argument {dataset} can be specified as follows: for fine-grained classification datasets, ‘I’, ‘DTD’, ‘Flower102’, ‘Food101’, ‘Cars’, ‘SUN397’, ‘Aircraft’, ‘Pets’, ‘Caltech101’, ‘UCF101’, or ‘eurosat’, and for datasets with natural distribution shifts, ‘V2’, ‘A’, ‘R’, or ‘K’.

## Results

### Comparison on Fine-grained Datasets

**CLIP ViT-B/16 (512-d)**

| Method | Metric | ImageNet | DTD | Flowers102 | Food101 | SUN397 | Aircrafts | OxfordPets | Caltech101 | UCF101 | EuroSAT | Stanford Cars | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | Acc. | 66.70 | 44.30 | 67.30 | 83.60 | 62.50 | 23.90 | 88.00 | 92.90 | 65.00 | 41.30 | 65.30 | 63.70 |
| | ECE | 2.12 | 8.50 | 3.00 | 2.39 | 2.53 | 5.11 | 4.37 | 5.50 | 3.59 | 13.89 | 4.25 | 4.43 |
| TPT | Acc. | 69.00 | 46.70 | 69.00 | 84.70 | 64.50 | 23.40 | 87.10 | 93.80 | 67.30 | 42.40 | 66.30 | 65.00 |
| | ECE | 10.60 | 21.20 | 13.50 | 3.98 | 11.30 | 16.80 | 5.77 | 4.51 | 2.54 | 13.20 | 5.16 | 11.60 |
| C-TPT | Acc. | 68.50 | 46.00 | 69.80 | 83.70 | 64.80 | 24.85 | 88.20 | 93.63 | 65.70 | 43.20 | 65.80 | 64.57 |
| | ECE | 3.15 | 11.90 | 5.04 | 3.43 | 5.04 | 4.36 | 1.90 | 4.24 | 2.54 | 13.20 | 1.59 | 5.13 |
| O-TPT | Acc. | 67.33 | 45.68 | 70.07 | 84.13 | 64.23 | 23.64 | 87.95 | 93.95 | 64.16 | 42.84 | 64.53 | 64.41 |
| | ECE | 1.96 | 7.88 | 3.87 | 1.46 | 4.93 | 3.68 | 1.90 | 3.80 | 2.34 | 12.98 | 1.78 | 4.23 |
| **A-TPT** | **Acc.** | **67.70** | **45.51** | **69.22** | **83.64** | **66.04** | **23.76** | **88.33** | **93.87** | **66.16** | **44.06** | **65.78** | **64.92** |
| **(Ours)** | **ECE** | **1.45** | **4.76** | **3.61** | **1.37** | **3.28** | **3.14** | **1.17** | **2.76** | **2.12** | **3.92** | **1.09** | **2.61** |

**CLIP RN50 (1024-d)**

| Method | Metric | ImageNet | DTD | Flowers102 | Food101 | SUN397 | Aircrafts | OxfordPets | Caltech101 | UCF101 | EuroSAT | Stanford Cars | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | Acc. | 58.10 | 40.00 | 61.00 | 74.00 | 58.60 | 15.60 | 83.80 | 85.80 | 58.40 | 23.70 | 55.70 | 55.90 |
| | ECE | 2.09 | 9.91 | 3.19 | 3.11 | 3.54 | 6.45 | 5.91 | 4.33 | 3.05 | 15.40 | 4.70 | 5.61 |
| TPT | Acc. | 60.70 | 41.50 | 62.50 | 74.90 | 61.10 | 17.00 | 84.50 | 87.00 | 59.50 | 28.30 | 58.00 | 57.70 |
| | ECE | 11.40 | 25.70 | 13.40 | 5.25 | 9.24 | 16.10 | 3.65 | 5.04 | 12.40 | 22.50 | 3.76 | 11.70 |
| C-TPT | Acc. | 60.20 | 42.20 | 65.20 | 74.70 | 61.00 | 17.00 | 84.10 | 86.90 | 59.70 | 27.80 | 56.50 | 57.75 |
| | ECE | 3.01 | 19.80 | 4.14 | 1.86 | 2.93 | 10.70 | 2.77 | 2.07 | 3.83 | 15.10 | 1.94 | 6.19 |
| O-TPT | Acc. | 58.97 | 41.90 | 65.61 | 74.22 | 60.85 | 16.77 | 83.40 | 86.86 | 58.84 | 28.35 | 56.44 | 57.47 |
| | ECE | 3.10 | 16.53 | 2.50 | 1.20 | 3.20 | 8.18 | 3.50 | 2.75 | 2.60 | 14.71 | 1.69 | 5.45 |
| **A-TPT** | **Acc.** | **58.44** | **40.90** | **64.89** | **74.10** | **60.46** | **14.58** | **83.48** | **86.57** | **60.24** | **32.14** | **57.08** | **57.53** |
| **(Ours)** | **ECE** | **2.49** | **6.41** | **2.39** | **1.11** | **2.90** | **6.14** | **2.47** | **1.98** | **2.34** | **2.51** | **1.38** | **2.92** |


### Comparison on Natural Distribution Shift Datasets

**CLIP ViT-B/16 (512-d)**

| Method | Metric | ImageNet-A | ImageNet-V2 | ImageNet-R | ImageNet-S | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | Acc. | 47.80 | 60.80 | 74.00 | 46.10 | 57.20 |
| | ECE | 8.61 | 3.01 | 3.58 | 4.95 | 5.04 |
| TPT | Acc. | 52.60 | 63.00 | 76.70 | 47.50 | 59.90 |
| | ECE | 16.40 | 11.10 | 4.36 | 16.10 | 12.00 |
| C-TPT | Acc. | 51.60 | 62.70 | 76.00 | 47.90 | 59.60 |
| | ECE | 8.16 | 6.23 | 1.54 | 7.35 | 5.82 |
| O-TPT | Acc. | 49.87 | 61.65 | 72.55 | 47.12 | 57.80 |
| | ECE | 7.22 | 3.97 | 1.46 | 6.87 | 4.88 |
| **A-TPT** | **Acc.** | **50.39** | **60.90** | **74.87** | **46.09** | **58.06** |
| **(Ours)** | **ECE** | **6.45** | **2.96** | **1.39** | **4.87** | **3.92** |

**CLIP RN50 (1024-d)**

| Method | Metric | ImageNet-A | ImageNet-V2 | ImageNet-R | ImageNet-S | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | Acc. | 21.70 | 51.40 | 56.00 | 33.30 | 40.60 |
| | ECE | 21.30 | 3.33 | 2.07 | 3.15 | 7.46 |
| TPT | Acc. | 25.20 | 54.60 | 58.90 | 35.10 | 43.50 |
| | ECE | 31.00 | 13.10 | 9.18 | 13.70 | 16.70 |
| C-TPT | Acc. | 23.40 | 54.70 | 58.00 | 35.10 | 42.80 |
| | ECE | 25.40 | 8.58 | 4.57 | 9.70 | 12.10 |
| O-TPT | Acc. | 23.07 | 53.11 | 54.47 | 33.98 | 41.16 |
| | ECE | 24.56 | 3.87 | 4.47 | 5.85 | 9.69 |
| **A-TPT** | **Acc.** | **21.66** | **51.48** | **55.78** | **33.37** | **40.57** |
| **(Ours)** | **ECE** | **21.14** | **3.10** | **3.96** | **3.09** | **7.82** |

## Qualitative Results - Reliability Diagrams

### CLIP ViT-B/16

| Method | Food | DTD | FLW | Air | UCF | Cars | SUN |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C-TPT** | <img src="figures/C-TPT-Food.png" width="100%"> | <img src="figures/C-TPT-DTD.png" width="100%"> | <img src="figures/C-TPT-Flower.png" width="100%"> | <img src="figures/C-TPT-Aircraft.png" width="100%"> | <img src="figures/C-TPT-UCF.png" width="100%"> | <img src="figures/C-TPT-Cars.png" width="100%"> | <img src="figures/C-TPT-SUN.png" width="100%"> |
| **O-TPT** | <img src="figures/O-TPT-Food.png" width="100%"> | <img src="figures/O-TPT-DTD.png" width="100%"> | <img src="figures/O-TPT-Flower.png" width="100%"> | <img src="figures/O-TPT-Aircraft.png" width="100%"> | <img src="figures/O-TPT-UCF.png" width="100%"> | <img src="figures/O-TPT-Cars.png" width="100%"> | <img src="figures/O-TPT-SUN.png" width="100%"> |
| **A-TPT** | <img src="figures/A-TPT-Food.png" width="100%"> | <img src="figures/A-TPT-DTD.png" width="100%"> | <img src="figures/A-TPT-Flower.png" width="100%"> | <img src="figures/A-TPT-Aircraft.png" width="100%"> | <img src="figures/A-TPT-UCF.png" width="100%"> | <img src="figures/A-TPT-Cars.png" width="100%"> | <img src="figures/A-TPT-SUN.png" width="100%"> |

### CLIP RN50 Backbone

| Method | Air | UCF | Cars | SUN |
|:---:|:---:|:---:|:---:|:---:|
| **C-TPT** | <img src="figures/C-TPT-RN-Aircraft.png" width="100%"> | <img src="figures/C-TPT-RN-UCF.png" width="100%"> | <img src="figures/C-TPT-RN-Cars.png" width="100%"> | <img src="figures/C-TPT-RN-SUN.png" width="100%"> |
| **O-TPT** | <img src="figures/O-TPT-RN-Aircraft.png" width="100%"> | <img src="figures/O-TPT-RN-UCF.png" width="100%"> | <img src="figures/C-TPT-RN-Cars.png" width="100%"> | <img src="figures/O-TPT-RN-SUN.png" width="100%"> |
| **A-TPT** | <img src="figures/A-TPT-RN-Aircraft.png" width="100%"> | <img src="figures/A-TPT-RN-UCF.png" width="100%"> | <img src="figures/A-TPT-RN-Cars.png" width="100%"> | <img src="figures/A-TPT-RN-SUN.png" width="100%"> |

## Acknowledgement

The computational resources for this research were supported by the Accelerating Higher Education Expansion and Development (AHEAD) Operation Grant No. 6026-LK/8743-LK from the Ministry of Higher Education, Sri Lanka, funded by the World Bank and the National Research Council of Sri Lanka Grant No. 19-080.

Also we would like thank the authors of the [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp), [TPT](https://github.com/azshue/TPT) and [C-TPT](https://github.com/hee-suk-yoon/C-TPT) for releasing their code open-source and their instructions for data preparation.

## Citation

If you find our work, this repository useful in your research, please consider giving a star ⭐ and citation.

```bibtex
@article{ahamed2025tpt,
    author  = {Ahamed, Shihab Aaqil and Thanthrige, Udaya SKP and Rodrigo, Ranga and Khan, Muhammad Haris},
    title   = {A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models},
    journal = {arXiv preprint arXiv:2510.26441},
    year    = {2025}
}
```

## Contact
If you have any questions, please feel free to reach out at shihabaaqilahamed@gmail.com