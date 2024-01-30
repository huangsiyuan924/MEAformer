# MEAformer
MEAformer: An all-MLP Transformer with Temporal External Attention for Long-term Time Series Forecasting

This repo is the official Pytorch implementation of MEAformer: "MEAformer: An all-MLP Transformer with Temporal External Attention for Long-term Time Series Forecasting". 

### Main Results
Multivariate Forecasting:
![image](https://private-user-images.githubusercontent.com/47273722/300806692-294818da-7c7e-4102-bbab-60fc059ce372.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDY2MTg0MDgsIm5iZiI6MTcwNjYxODEwOCwicGF0aCI6Ii80NzI3MzcyMi8zMDA4MDY2OTItMjk0ODE4ZGEtN2M3ZS00MTAyLWJiYWItNjBmYzA1OWNlMzcyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMzAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTMwVDEyMzUwOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQ3NTBlMTBiYTQyZDkyOTQ1NjljMzI3NWQxMmU2Mzk3M2EzZDkyYTI1ZjZmMzAyZmFkMzUxMDI3ZmNhY2M0OTUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.-kfaugHGXl6eHBIb4BAfhe-DwNmzysuklxGeM7VBX0U)
MEAformer and decomposition-based MEAformer outperform other methods by a large margin.
## Detailed Description
We provide all experiment script files in `./scripts`:


This code is simply built on the code base of DLinear and Autoformer. We appreciate the following GitHub repos a lot for their valuable code base or datasets:

The implementation of DLinear is from https://github.com/cure-lab/LTSF-Linear


The implementation of Autoformer, Informer, and Transformer is from https://github.com/thuml/Autoformer

The implementation of FEDformer is from https://github.com/MAZiqing/FEDformer

The implementation of Pyraformer is from https://github.com/alipay/Pyraformer


## Getting Started
### Environment Requirements

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n MEAformer python=3.6.9
conda activate MEAformer
pip install -r requirements.txt
```

### Data Preparation

You can obtain all the nine benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. All the datasets are well-pre-processed and can be used easily.

```
mkdir dataset
```
**Please put them in the `./dataset` directory**

### Training Example
- In `scripts/ `, we provide the model implementation *MEAformer/MEAformer(D)/Dlinear/Autoformer/Informer/Transformer*
- In `FEDformer/scripts/`, we provide the *FEDformer* implementation
- In `Pyraformer/scripts/`, we provide the *Pyraformer* implementation

For example:

To train the **MEAformer** on **Traffic dataset**, you can use the script `scripts/EXP-LongForecasting/MEAformer/traffic.sh`:
```
sh scripts/EXP-LongForecasting/MEAformer/traffic.sh
```
It will start to train MEAformer by default, the results will be shown in `logs/LongForecasting`.



## Citing

If you find this repository useful for your work, please consider citing it as follows:

```bibtex

```

