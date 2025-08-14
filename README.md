# Human3DSEG

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2507.18655)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://your-project-page.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Human3DSEG: Part Segmentation of Human Meshes via Multi-View Human Parsing**  
> [James Dickens](https://github.com/JamesMcCullochDickens), Kamyar Hamad  
> 2025

## Abstract
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua

<!--## Key Features

- Feature 1
- Feature 2  
- Feature 3-->

## Repository Structure

This repository contains components for automatic 3D human point cloud annotation and segmentation, organized into two main parts:

```
Human3DSEG/
├── Data_Processing/      # Data preparation and preprocessing scripts
├── Model/                # Model training, evaluation, and inference code
├── data/                 
├── data_fps.py           
├── utils/                # Utility functions and helper modules
├── README.md             # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/Human3DSeg.git
cd Human3DSEG

# Create conda environment
conda create -n human3dseg python=3.9
conda activate human3dseg

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation (`Data_Preprocessing/` folder)
The data preparation component provides tools and scripts for automatic 3D human point cloud annotation. This component:
- Operates independently from the training pipeline
- Includes comprehensive data preparation utilities
- Handles automatic annotation of 3D human point clouds
- Can be used as a standalone module for data preprocessing

```
Data_Preprocessing/
├── src/                   # Data loading utilities and scripts
│   ├── PointTransformerV3
│   ├── PyOpenGL
│   ├── Sapiens
│   ├── m2fp
│   └── Yolov8
├── data_processing/       # Evaluation scripts and metrics  
├── __init__.py 
```
To use m2fp as 2d segmentor, clone the official implementation into the src/ directory.

**Usage for data preparation:**
```bash
cd Data_Preprocessing
python Data_Processing/data_processing.py 
```

---

### Training Pipeline (`Model/` folder)
The training pipeline processes the prepared data from the data preparation stage. This component:
- Works with output from the data preparation module
- Implements the main segmentation model training
- Includes evaluation and inference capabilities

```
Model/
├── Dataloaders/      # Data loading utilities and scripts
├── Evaluation/       # Evaluation scripts and metrics
├── Training/         # Training scripts and model definitions
├── __init__.py    
├── run.py            # Main entry point for running training/evaluation
```

To use Point Transformer v1 as a backbone, clone the official implementation into the `Model/` directory.

You can do this by running:
```bash
cd Model
git clone https://github.com/POSTECH-CVLab/point-transformer.git .
```

**Training:**
```bash
python Model/run.py pointtransformer_cihp_seg.yml human_seg_3d_cihp_PT.yml --gpu_override '[0, 1]'
```
<!--**Evaluation:**

### Evaluation
```bash
python 
```


## Dataset

TODO: Information about the dataset used, how to obtain it, and preprocessing steps.

## Results

| Method | Metric 1 | Metric 2 | Metric 3 |
|--------|----------|----------|----------|
| Ours   | **X.XX** | **X.XX** | **X.XX** |
| Method A | X.XX   | X.XX     | X.XX     |
| Method B | X.XX   | X.XX     | X.XX     |-->

## Citation

```bibtex
@article{dickens2025human3dseg,
  title={Part Segmentation of Human Meshes via Multi-View Human Parsing},
  author={Dickens, James and Hamad, Kamyar},
  journal={arXiv preprint arXiv:2507.18655},
  year={2025},
  month={07},
  doi={10.48550/arXiv.2507.18655},
  url={https://arxiv.org/abs/2507.18655}
}
```

## License
[MIT](LICENSE) © 2025 James Dickens, Kamyar O. Hamad

## Acknowledgments

We thank the [THUman2.0 dataset](https://github.com/ytrock/THuman2.0-Dataset) for providing the human scans used in this work; please cite the dataset if you use it.


