# README

This repository provides the implementation for our paper **"CompLeak: Deep Learning Model Compression Exacerbates Privacy Leakage"**, which investigates how model compression exacerbates privacy leakage under membership inference attacks (MIA) across three different scenarios.

---

## 1. Environment Setup

The code requires **PyTorch  2.7.0 and Python  3.12**. It is recommended to install all dependencies specified in `requirements.txt` to ensure compatibility:

```bash
pip install -r requirements.txt
```

---

## 2. Overview of CompLeak

**CompLeak** presents the **first systematic study** of privacy risks introduced by three widely used model compression techniques under MIA:

* **Pruning**
* **Quantization**
* **Weight Clustering**

We evaluate these techniques across **three scenarios**, differentiated by the attacker’s access to original and/or compressed models.

### 2.1 CompLeak<sub>NR</sub>

* **Setting**: Only a *single model* is available (either the original model or one compressed model).
* **Approach**: Existing MIAs are directly applied to the available victim model (single original/compressed model).
* **Goal**: Determine whether the compressed model is more vulnerable to MIAs than original model.

---

### 2.2 CompLeak<sub>SR</sub>

* **Setting**: Both the **original model** and **one compressed model** are available.
* **Approach**: The compressed model is used as a **reference model** and paired with the original model during the attack.
* **Goal**: Provide a more fine-grained analysis of whether the **compression process itself** exacerbates privacy leakage.

---

### 2.3 CompLeak<sub>MR</sub>

* **Setting**: The attacker has access to **multiple compressed models** with different compression levels.
* **Approach**: Multiple compressed model are used as the **reference model**
* **Goal**: Investigate how privacy leakage in the **multiple compressed models** scenario.

---


## 3. Step-by-Step Procedure

The workflow consists of preparing the original and compressed models, 
then performing privacy auditing using CompLeak<sub>NR</sub>, CompLeak<sub>SR</sub>, and CompLeak<sub>MR</sub>.

---

### 3.0 Data Preparation


For **faster testing**, 
you may choose **CIFAR-10 with ResNet18**.

**CIFAR-10:**

   No manual preparation is required.
   The dataset will be automatically downloaded to:

   ```
   data/datasets/
   ```
   when running any training or attack script.
The data split is in `data/cifar10_data_index.pkl`
---

### 3.1 Prepare Original Models and Compressed Models
  * **Example (CIFAR-10 + ResNet18)**
  * Train the original models by running commands:

    ```bash
    python train.py --dataset_name cifar10 --model_name resnet18 --num_cls 10 --lr 0.1 --epochs 100 --batch_size 128
    ```
   * Compress the original models by following these steps:
     * **1. Pruning**: Use `--prune_sparsity` to set the pruning ratio (0.6, 0.7, 0.8, 0.9 correspond to 60%, 70%, 80%, 90% sparsity):
        ```bash
        python pruning/prune.py --dataset_name cifar10 --model_name resnet18 --num_cls 10 --lr 0.001 --prune_epochs 30 --prune_sparsity 0.6
        ````
        Adjust `--prune_sparsity` with 0.6, 0.7, 0.8, 0.9 to obtain four models with 60%, 70%, 80%, 90% pruning levels.
     * **2. Quantization**: Run the following command for int8 quantization:

        ```bash
        python quantization/qat.py --dataset_name cifar10 --model_name resnet18 --num_cls 10 --lr 0.001 --quantized_epochs 30 --degree int8
        ```
     * **3. Weight Clustering**:
Use `--bit` to control the number of cluster levels:
       * `2` → 4 cluster centers
       * `3` → 8 cluster centers
       * `4` → 16 cluster centers

        ```bash
        python clustering/clustering.py --dataset_name cifar10 --model_name resnet18 --num_cls 10 --lr 0.00001 --epochs 30 --bit 2
        ```

        Adjust `--bit` with 2, 3, 4 to obtain three models with different cluster levels.

    
After 3.1, you will have:

* One **original model** (full precision) in `results/`
* Multiple **compressed models** with different compression techniques and levels
  * Pruned models with 60%, 70%, 80%, and 90% sparsity in `results_pruning/`
  * INT-8 quantized model in `results_quantization/`
  * Clustered models with 4, 8, and 16 cluster centers in `results_clustering/`

---

### 3.2 CompLeak<sub>NR</sub>

**CompLeak<sub>NR</sub>** applies six existing MIAs to audit the privacy leakage of a single model (original or compressed).
  * **Two train-based attacks [1, 2]**
  * **Two metric-based attacks [3, 4]**
  * **SAMIA(also train-based) [5]**
  * **RMIA [6]**
---
#### Attack the Original Model

```bash
python CompLeakNR.py --model_name resnet18 --dataset_name cifar10 --num_cls 10 --batch_size 128 --original
```
---

#### Attack Compressed Models
* `--compress_name` specifies the compression type:

  * `pruning` (with `--compress_degree` 0.6, 0.7, 0.8, 0.9)
  * `quantization` (with `--compress_degree` int8)
  * `clustering` (with `--compress_degree` 4, 8, 16)

**Pruning (60%, 70%, 80%, 90%):**

```bash
python CompLeakNR.py --compress_degree 0.6 --dataset_name cifar10 --num_cls 10 --batch_size 128 --compress --compress_name pruning --model_name resnet18
```
```bash
python CompLeakNR.py --compress_degree 0.7 --dataset_name cifar10 --num_cls 10 --batch_size 128 --compress --compress_name pruning --model_name resnet18
```
```bash
python CompLeakNR.py --compress_degree 0.8 --dataset_name cifar10 --num_cls 10 --batch_size 128 --compress --compress_name pruning --model_name resnet18
```
```bash
python CompLeakNR.py --compress_degree 0.9 --dataset_name cifar10 --num_cls 10 --batch_size 128 --compress --compress_name pruning --model_name resnet18
```
**Quantization (int8):**

```bash
python CompLeakNR.py --compress_degree int8 --dataset_name cifar10 --num_cls 10 --batch_size 128 --compress --compress_name quantization --model_name resnet18
```

**Weight Clustering (4, 8, 16 centroids):**

```bash
python CompLeakNR.py --compress_degree 4 --dataset_name cifar10 --num_cls 10 --batch_size 128 --compress --compress_name clustering --model_name resnet18
```
```bash
  python CompLeakNR.py --compress_degree 8 --dataset_name cifar10 --num_cls 10 --batch_size 128 --compress --compress_name clustering --model_name resnet18
```
```bash
python CompLeakNR.py --compress_degree 16 --dataset_name cifar10 --num_cls 10 --batch_size 128 --compress --compress_name clustering --model_name resnet18
```
---

#### Results

Each run reports six MIAs:

* CompLeakNR_SAMIA: Reported as **CompLeak<sub>NR</sub> [74]** in the paper
* CompLeakNR_metric_based_method_1: Reported as **CompLeak<sub>NR</sub> [60]** in the paper
* CompLeakNR_metric_based_method_2: Reported as **CompLeak<sub>NR</sub> [73]** in the paper
* CompLeakNR_train_based_method_1: Reported as **CompLeak<sub>NR</sub> [63]** in the paper
* CompLeakNR_train_based_method_2: Reported as **CompLeak<sub>NR</sub> [51]** in the paper
* CompLeakNR_RMIA: Reported as **CompLeak<sub>NR</sub> (RMIA) [75]** in the paper

Reported metrics include **ACC**, **AUC**, and **TPR at 0.1% FPR**.
Only highly compressed models (e.g., 90% pruning) show reduced vulnerability.
---


## 3.3. CompLeak<sub>SR</sub>

**CompLeakSR** combines information from the original model and one compressed model.

* `--compress_name` specifies the compression type:

  * `pruning` (with `--compress_degree` 0.6, 0.7, 0.8, 0.9)
  * `quantization` (with `--compress_degree` int8)
  * `clustering` (with `--compress_degree` 4, 8, 16)

---
#### Execution

**Attack Pruning (60%, 70%, 80%, 90%):**

```bash
python CompLeakSR.py --compress_degree 0.6 --dataset_name cifar10 --num_cls 10 --batch_size 128 --original --compress --model_name resnet18 --compress_name pruning
```
```bash
python CompLeakSR.py --compress_degree 0.7 --dataset_name cifar10 --num_cls 10 --batch_size 128 --original --compress --model_name resnet18 --compress_name pruning
```
```bash
python CompLeakSR.py --compress_degree 0.8 --dataset_name cifar10 --num_cls 10 --batch_size 128 --original --compress --model_name resnet18 --compress_name pruning
```
```bash
python CompLeakSR.py --compress_degree 0.9 --dataset_name cifar10 --num_cls 10 --batch_size 128 --original --compress --model_name resnet18 --compress_name pruning
```
**Quantization:**

```bash
python CompLeakSR.py --compress_degree int8 --dataset_name cifar10 --num_cls 10 --batch_size 128 --original --compress --model_name resnet18 --compress_name quantization
```

**Weight Clustering (4, 8, 16 centroids):**

```bash
python CompLeakSR.py --compress_degree 4 --dataset_name cifar10 --num_cls 10  --batch_size 128 --original --compress --model_name resnet18 --compress_name clustering
```
```bash
python CompLeakSR.py --compress_degree 8 --dataset_name cifar10 --num_cls 10  --batch_size 128 --original --compress --model_name resnet18 --compress_name clustering
```
```bash
python CompLeakSR.py --compress_degree 16 --dataset_name cifar10 --num_cls 10  --batch_size 128 --original --compress --model_name resnet18 --compress_name clustering
```
---

#### Results

**CompLeak<sub>SR</sub>** amplifies privacy leakage compared to **CompLeak<sub>NR</sub>**.

---

### 3.3 CompLeak<sub>MR</sub>

**CompLeak<sub>MR</sub>** aggregates information from multiple compressed models.

---

#### Execution

Ensure all compressed models have been processed using **CompLeak<sub>SR</sub>**, then open:

```bash
jupyter notebook CompLeakMR.ipynb
```

Execute all cells.

---

#### Results

Results are reported in **last cell** of `CompLeakMR.ipynb`.
Aggregating multiple compressed models further amplifies privacy leakage.

---


## References

```
[1] Shokri, Reza, et al. Membership inference attacks against machine learning models. 2017 IEEE Symposium on Security and Privacy (SP). IEEE, 2017.

[2] Nasr, Milad, Reza Shokri, and Amir Houmansadr. Machine learning with membership privacy using adversarial regularization. Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security. 2018.

[3] Yeom, Samuel, et al. Privacy risk in machine learning: Analyzing the connection to overfitting. 2018 IEEE 31st Computer Security Foundations symposium (CSF). IEEE, 2018.

[4] Song, Liwei, and Prateek Mittal. Systematic evaluation of privacy risks of machine learning models. 30th USENIX security symposium (USENIX security 21). 2021.

[5] Yuan, Xiaoyong, and Lan Zhang. Membership inference attacks and defenses in neural network pruning. 31st USENIX Security Symposium (USENIX Security 22). 2022.

[6] Sajjad Zarifzadeh, Philippe Liu, and Reza Shokri. Low-cost high-power membership inference attacks. In Forty-first International Conference on Machine Learning, ICML 2024.
```



