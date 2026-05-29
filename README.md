# LightGCL-improved: Dynamic Learnable Graph Augmentation

This is the official PyTorch implementation for the paper **"An improvement of LightGCL: Simple yet effective graph contrastive learning for recommendation"**, accepted at the *30th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2026)*.

<img width="1242" height="505" alt="image" src="https://github.com/user-attachments/assets/b09a3603-758a-4a53-979d-2dc8fc75c2d3" />
---

# 1. Note on Datasets and Directories

Due to the size of the datasets, we have compressed them into zip files. Please unzip them before running the model.

This repository supports four datasets evaluated in our paper:

* Yelp
* Gowalla
* Amazon
* BeerAdvocate

Keeping the current directory structure is recommended.

Before running the code, please ensure that the following directories are created under the project root:

```text
log/
saved_model/
```

These folders are used to store:

* Training logs
* Saved model checkpoints
* Optimizer states

---

# 2. Running Environment

We developed and tested the code in the following environment:

```text
Python version 3.9.12
torch==1.12.0+cu113
numpy==1.21.5
tqdm==4.64.0
```

---

# 3. How to Run the Code

Unlike the static SVD augmentation in the original LightGCL, our improved version utilizes a **dynamic learnable graph augmentor**.

To enable this mechanism, specify:

```bash
--view2 graphaug
```

along with the corresponding augmentation hyperparameters.

## Yelp

```bash
python main.py \
  --data yelp \
  --view2 graphaug \
  --twohop_per_user 10 \
  --xi 0.8 \
  --tau1 0.2 \
  --lambda1 0.2
```

## Gowalla

```bash
python main.py \
  --data gowalla \
  --view2 graphaug \
  --twohop_per_user 10 \
  --xi 0.8 \
  --tau1 0.2 \
  --lambda1 0.2 \
  --lambda2 0
```

## Amazon

```bash
python main.py \
  --data amazon \
  --view2 graphaug \
  --twohop_per_user 10 \
  --xi 0.85 \
  --tau1 0.2 \
  --gnn_layer 1 \
  --lambda2 0 \
  --temp 0.1
```

## BeerAdvocate

```bash
python main.py \
  --data beeradvocate \
  --view2 graphaug \
  --twohop_per_user 10 \
  --xi 0.8 \
  --tau1 0.2
```

---

# 4. Configurable Arguments

| Argument            | Description                                                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| `--cuda`            | Specifies which GPU to run on if multiple GPUs are available.                                              |
| `--data`            | Selects the dataset to use (`yelp`, `gowalla`, `amazon`, `beeradvocate`).                                  |
| `--view2`           | Specifies the augmentation strategy for View 2. Set to `graphaug` to use the proposed learnable augmentor. |
| `--twohop_per_user` | Bounds the candidate set size (K) for 2-hop neighbor sampling per user.                                    |
| `--xi`              | Specifies the hard threshold (\xi) used to filter weak connections and maintain graph sparsity.            |
| `--tau1`            | The temperature (\tau_1) used in the Gumbel-Sigmoid reparameterization trick.                              |
| `--lambda1`         | Regularization weight (\lambda_1) for the contrastive learning loss.                                       |
| `--lambda2`         | L2 regularization weight (\lambda_2).                                                                      |
| `--temp`            | Temperature (\tau) used in the InfoNCE contrastive learning loss.                                          |

---

# 5. Complexity Analysis of LightGCL-improved

A major advantage of the original LightGCL is its computational efficiency. However, its SVD-based augmentation requires a dense global low-rank matrix approximation during preprocessing, resulting in a complexity of:

```math
O(q(|\mathcal{U}| + |\mathcal{V}|)Ld)
```

In our model, we trade global low-rank density for adaptive and sparse higher-order exploration.

By:

* Constraining the 2-hop candidate mining with a strict upper bound (K)
* Applying the threshold (\xi)

the augmented edge set (\mathcal{E}') remains highly sparse.

The computational complexity becomes:

### Interaction Probability Computation

```math
O(K|\mathcal{U}|d_{mlp})
```

### View-2 Graph Propagation

```math
O(|\mathcal{E}'|Ld)
```

Since (|\mathcal{E}'|) is strictly controlled, the proposed dynamic framework maintains efficiency comparable to the baseline LightGCN while completely eliminating the memory overhead caused by storing dense reconstructed adjacency matrices.

---

# 6. Citation

Please kindly cite our paper if you find this research or code useful for your work.

```bibtex
@inproceedings{nguyen2026lightgclimproved,
  title={An improvement of LightGCL: Simple yet effective graph contrastive learning for recommendation},
  author={Nguyen, Ngoc-Thao and Tran, Thanh-Long and Nguyen, Duc-Phuc},
  booktitle={30th International Conference on Knowledge-Based and Intelligent Information \& Engineering Systems (KES 2026)},
  series={Procedia Computer Science},
  publisher={Elsevier},
  year={2026}
}
```
