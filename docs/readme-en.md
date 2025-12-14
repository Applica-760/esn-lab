# esn-lab: A Python Package for Echo State Network Experiments

[日本語](./readme.md) | [English](./readme-en.md)

## Overview

`esn-lab` is a Python package developed for efficiently conducting research and experiments using Echo State Networks (ESN).

It was created for my university research and is designed to allow flexible experiment management based on configuration files.<br>
While it includes some general implementations, it also has implementations tailored to specific domains.<br>
It also serves as a portfolio showcasing my development skills.<br>
<br>
Currently, it has implemented the basic functionalities required for research, but I plan to continue its development to grow it into a more versatile tool.


## ✨ Features

  * **Flexible Experiment Management via Configuration Files**: Manage model parameters and training data using `YAML` format configuration files.
  * **Ensuring Reproducibility**: Automatically saves the runtime configuration to enhance experiment reproducibility.
  * **Multiple Execution Modes**: Aims to support a wide range of experimental scenarios, from training/prediction on a single data point to batch processing of multiple data, and hyperparameter search using 10-fold cross-validation.
  * **Easy Operation via CLI**: Intuitively execute training, prediction, and evaluation processes through the command-line interface (CLI).

## 📦 Installation

```bash
pip install esn-lab
```

## 🚀 Usage

### 1\.Initialize Configuration Files

First, run the following command to generate a `configs` directory in the current directory.

```
esnlab init
```


This will copy the templates for various configuration files.

### 2\. Edit Configuration Files

Describe the basic settings common to the entire project in configs/base.yaml.

```
project: "esn-research"
seeds: [2024, 706, 4410, 5385, 1029, 1219, 8380, 8931, 5963, 19800]
num_of_classes: 3

data:
    type: "complement"

model:
    name: "esn"
    Nu: 256
    Nx: 100
    Ny: 3
    density: 0.5
    input_scale: 0.01
    rho: 0.9
    optimizer: "tikhonov"

```

Next, edit the respective configuration file, such as `configs/train/single.yaml`, according to the mode you want to run.


```
id: "sample_001"
path: "/path/to/your/data/sample_001.jpg"
class_id: 0
```

### 3\. Run Training

To train on a single piece of data, run the following command.

```
esnlab train single
```

When training is complete, the results (weight files and logs) will be saved in the `outputs/runs/{execution_datetime}_{mode}-{variant}/` directory.

## 🛠️ Command Line Interface

`esn-lab` supports the following modes and variants.

| Mode       | Variant    | Description                                                                 |
| :---------- | :---------- | :-------------------------------------------------------------------------- |
| `train`     | `single`    | Train on a single dataset.                                                  |
|              | `batch`     | Train on multiple datasets at once.                                        |
|              | `tenfold`   | Perform 10-fold cross-validation to explore hyperparameters.               |
| `predict`   | `single`    | Make predictions using a single dataset.                                   |
|              | `batch`     | Make predictions using multiple datasets at once.                         |
| `evaluate`  | `run`       | Evaluate prediction results.                                               |
|              | `summary`   | Plot a confusion matrix summarizing the 10-fold cross-validation results.  |
|              | `tenfold`   | Perform inference on test data using the 10-fold cross-validation results. |


## Roadmap
This project is still under development. The following feature enhancements are planned for the future.

 - Enhanced Evaluation Functions: Generation of confusion matrices and visualization of more detailed evaluation metrics.

- Integration of Visualization Tools: Functions to visualize the learning process and the internal state of the reservoir.

## Dependencies
 - numpy
 - matplotlib
 - pandas
 - opencv-python
 - PyYAML
 - omegaconf
 - networkx


## License
This project is licensed under the MIT License.