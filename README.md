# LightGCL-improved: Dynamic Learnable Graph Augmentation

This is the official PyTorch implementation for the paper [**An improvement of LightGCL: Simple yet effective graph contrastive learning for recommendation**], accepted at the *30th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2026)*.

<img width="1242" height="505" alt="image" src="https://github.com/user-attachments/assets/fce3035e-a90e-4a47-b9dd-db41f0c6292b" />


### 1. Note on datasets and directories
Due to the size of the datasets, we have compressed them into zip files. Please unzip them before running the model. This repository supports four datasets evaluated in our paper: **Yelp, Gowalla, Amazon, and BeerAdvocate**. Keeping the current directory structure is recommended.

Before running the codes, please ensure that two directories `log/` and `saved_model/` are created under the root directory. They are used to store the training results and the saved model/optimizer states.

### 2. Running environment

We developed our codes in the following environment:

```text
Python version 3.9.12
torch==1.12.0+cu113
numpy==1.21.5
tqdm==4.64.0
