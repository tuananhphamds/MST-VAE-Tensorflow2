# MST-VAE-Tensorflow2
### Multi-Scale Temporal Variational Autoencoder for Anomaly Detection in Multivariate Time Series

MST-VAE is an unsupervised learning approach for anomaly detection in multivariate time series. Inspired by InterFusion paper, we propose a simple yet effective multi-scale convolution kernels applied in Variational Autoencoder. 

Main techniques in this paper:
- Multi-scale module: short-scale and long-scale module
- We adopted Beta-VAE for training the model
- MCMC is applied to achieve better latent representations while detecting anomalies

## How to use the repository
### Clone the repository
```bash
git clone https://github.com/tuananhphamds/MST-VAE-Tensorflow2.git
cd MST-VAE-Tensorflow2
```

### Prepare experiment environment (GPU is needed)
1. Install Anaconda version 4.11.0
2. Create an environment with Python 3.7, Tensorflow-gpu=2.8.0 and dependencies
```
conda create -n mstvaetf2 python=3.7
conda activate mstvaetf2
conda install anaconda::cudatoolkit==11.3.1
conda install anaconda::cudnn=8.2.1
pip install -r requirements.txt
```

### Prepare data
In this study, we use five public datasets: ASD (Application Server Dataset), SMD (Server Machine Dataset), PSM (Pooled Server Metrics), SWaT (Secure Water Treatment), and WADI (Water Distribution).

ASD, SMD, PSM can be refered in ``MST-VAE-Tensorflow2/data/processed`` folder.

SWaT, WADI should be requested from https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

Each dataset must contain three files to be compatible with the code:
- <Name_of_dataset>_train.pkl: train data
- <Name_of_dataset>_test.pkl: test data
- <Name_of_dataset>_test_label.pkl: test label

For all detailed information of datasets, refer to ``MST-VAE-Tensorflow2/data_info.txt``
___
After requesting dataset SWaT & WADI, an email will be replied from iTrust website.

**For SWaT dataset**: 
download two files ``SWaT_Dataset_Attack_v0.csv`` and ``SWaT_Dataset_Normal_v0.csv`` in folder ``SWaT.A1 & A2_Dec 2015/Physical``

**For WADI dataset**:
download three files ``WADI_attackdata.csv``, ``WADI_14days.csv``, and ``table_WADI.pdf`` in folder ``WADI.A1_9 Oct 2017``.

```bash
Move all downloaded files to folder MST-VAE-Tensorflow2/swat_wadi
```
Make sure current working directory is MST-VAE-Tensorflow2/explib, run:
```bash

These following files will be generated: 
- SWaT_train.pkl
- SWaT_test.pkl
- SWaT_test_label.pkl
- WADI_train.pkl
- WADI_test.pkl
- WADI_test_label.pkl (not available): should be created using the description file ``table_WADI.pdf``

Move all the aboved files to ``MST-VAE-Tensorflow2/data/processed``

The training config can be modified at ``train_config.json``

### Run the code
Choose the dataset that you want to run and modify it on ``line 30, file run_experiment.py``
```bash
config = {
        'dataset': 'machine-2-1',
        'z_dim': 8,

Move to folder MST-VAE-Tensorflow2, run the following command:
```bash
python run_experiment.py
```

The training config can be modified at ``algorithm/train_config.json``

### Run your own data
1. Put your own data (three files) in ``data/processed``: ``<Name_of_dataset>_train.pkl, <Name_of_dataset>_test.pkl, <Name_of_dataset>_test_label.pkl``
2. Add your data dimension (number of metrics) in function ``get_data_dim on line 75, file algorithm/utils.py``
```bash
elif dataset == '<Name_of_dataset>':
    return <dim>
```
3. Add train config for running your data, you can copy and modify an example in ``train_config.json`` 
4. Do the steps in **Run the code**

### Cite this paper
If this code is useful for your research, please cite this paper: https://doi.org/10.3390/app121910078
python raw_data_converter.py
```
