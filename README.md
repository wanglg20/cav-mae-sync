# CAV-MAE Sync: Improving Contrastive Audio-Visual Mask Autoencoders via Fine-Grained Alignment

by [Edson Araujo](https://edsonroteia.github.io/), [Andrew Rouditchenko](https://people.csail.mit.edu/roudi/), [Yuan Gong](https://yuangongnd.github.io/), [Saurabhchand Bhati](https://scholar.google.com/citations?user=eVc2TGkAAAAJ&hl=en), [Samuel Thomas](https://research.ibm.com/people/samuel-thomas), [Brian Kingsbury](https://research.ibm.com/people/brian-kingsbury), [Leonid Karlinsky](https://scholar.google.com/citations?user=WbO7tjYAAAAJ&hl=en), [Rogerio Feris](https://research.ibm.com/people/rogerio-feris), [James R. Glass](https://www.csail.mit.edu/person/jim-glass), [Hilde Kuehne](https://hildekuehne.github.io/).

📚 [arXiv preprint](https://arxiv.org/abs/2505.01237) | 🖥️ [Project webpage](https://edsonroteia.github.io/cav-mae-sync)

- 📰 Our work was featured [on MIT News](https://news.mit.edu/2025/ai-learns-how-vision-and-sound-are-connected-without-human-intervention-0522)! 
- ✨ The paper has been accepted at CVPR 2025!
- 🚀 Code and pretrained models for retrieval on VGGSound are here! 
- 🚧 Classification/Localization codes and models coming soon!


## 🛠️ Installation

Before running the code, you need to install the required Python libraries. You can do this using either a virtual environment with `pip` or `conda`.

### Using `venv` and `pip`

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### Using `conda`

```bash
# Create a conda environment 
conda create -n cav-mae-sync python=3.7

# Activate the conda environment
conda activate cav-mae-sync

# Install the required packages
pip install -r requirements.txt
```


## 📦 Data Preparation

For preparing your data (audio, frames, and label files), **follow the original CAV-MAE instructions here:**  
https://github.com/YuanGongND/cav-mae#data-preparation

This ensures your data is in the correct format for CAV-MAE Sync. No changes are needed, just use the same process as the original repo.


## 🚀 Retrieval

To perform retrieval tasks, you first need to download the pretrained models and generate the necessary data files.

### 1. Download Pretrained Models

Navigate to the `pretrained_models` directory and execute the script:

```bash
cd pretrained_models
sh get_pretrained_model.sh
```

This will download the `cav_mae_sync.pth` file.

### 2. Generate Data Files

Navigate to the `datafiles` directory and run the script:

```bash
cd ../datafiles # Assuming you are in pretrained_models, otherwise adjust path
sh generate_datafiles.sh
```
This script will prompt you to enter the path to your VGGSound dataset. It will then use this path to generate the final data JSON files from the templates.

### 3. Run Retrieval

To run retrieval (example on VGGSound subset):

```bash
python src/retrieval.py --nums_samples 1600 --directions audio video --strategy diagonal_mean
```

Where:
- `--nums_samples` is the number of samples to evaluate.
- `--directions` is the direction of the retrieval task. `{audio, video}`
- `--strategy` is the strategy to use for retrieval. `{diagonal_mean, diagonal_max, mean, max}`


> **Dataset Size Note:** The VGGSound subset used here contains about 1520 samples. If you set `--nums_samples` greater than this, the script will just use the entire dataset (i.e., all available samples are evaluated).

> **Note:** Computing the similarity matrix in this (non-parallelized) code can take up to 40 minutes for the full retrieval result. For quick tests, reduce `--nums_samples` to a smaller value (e.g., 100 or 500). Alternatively, it is possible to optimize this large matrix multiplication. (Gladly accepting PRs!)


## 📌 Citation

If you use CAV-MAE Sync, please cite:

```
@inproceedings{araujo2025cavmaesync,
  title     = {CAV-MAE Sync: Improving Contrastive Audio-Visual Mask Autoencoders via Fine-Grained Alignment},
  author    = {Araujo, Edson and Rouditchenko, Andrew and Gong, Yuan and Bhati, Saurabhchand and Thomas, Samuel and Kingsbury, Brian and Karlinsky, Leonid and Feris, Rogerio and Glass, James R.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}

```

