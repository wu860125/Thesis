# Thesis_code_Jim Wu

This repository contains all the code implementations for the thesis **"On the Development of Graph Deviation Networks-based Equipment Health Prognostics Framework"**.

## Folder Structure
```
Code
│- CMP_Dataset                                  # Dataset of case study
│  ├─ 2016 PHM DATA CHALLENGE CMP DATA SET     # Original data
│  ├─ MyCMP                                    # Data after synchronization
│  ├─ preprocess_cmp.ipynb                     # Code for data preprocessing
│
│- Data_Plot                                   # Data visualization
│  ├─ {dataset folder}                         # Generated from preprocess_cmp.ipynb
│  ├─ data_plot.ipynb                          # Code for Figure 35 and 36 in the thesis
│
│- Full_Scenario                              # Models for full measurement scenario
│  ├─ Full_Univariate_time_series             # Univariate time series models
│  │  ├─ TTS_model.ipynb                      # ES, AR, ARIMA models (baseline)
│  │  ├─ GRU_model.ipynb                      # GRU model (baseline)
│  ├─ Full_GDN_related                        # Models based on the proposed framework
│  │  ├─ GDN_GRU.ipynb                         # Proposed framework
│  │  ├─ Random_GRU.ipynb                      # Random Features + GRU model (baseline)
│  │  ├─ Stats_GRU.ipynb                       # Statistical Features + GRU model (baseline)
│
│- Sampling_Scenario                          # Models for sampling measurement scenario
│  ├─ Sampling_Univariate_time_series        # Univariate time series models
│  │  ├─ Sampling_TTS_model.ipynb            # ES, AR, ARIMA models (baseline)
│  │  ├─ Sampling_GRU_model.ipynb            # GRU model (baseline)
│  ├─ Sampling_GDN_related                   # Models based on the proposed framework
│  │  ├─ Sampling_GDN_GRU.ipynb               # Proposed framework
│  │  ├─ Sampling_Latest_Filling.ipynb        # Latest Filling model (baseline)
│  │  ├─ Sampling_Random_GRU.ipynb            # Random Features + GRU model (baseline)
│  │  ├─ Sampling_Stats_GRU.ipynb             # Statistical Features + GRU model (baseline)
│
│- VM_Stats_features                          # Statistical Features VM
│  ├─ Stats_VM.ipynb                          
│- VM_GDN_based                               # GDN-based VM
│  ├─ GDN_VM.ipynb                            
```

## Installation

### Environment
- Windows 11
- NVIDIA GeForce RTX 3090 Ti

### Requirements
- Python >= 3.6
- CUDA == 11.8
- [PyTorch == 2.1.0](https://pytorch.org/)
- [PyG: torch-geometric == 1.7.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Required Packages
```
numpy
pandas
matplotlib
seaborn
sklearn
statsmodels
pmdarima
xgboost
```

### Quickstart
To create a virtual environment, run the following command:
```
conda env create --file environment.yaml
```

## Usage

### Data Preparation
1. The CMP dataset is sourced from [PHM Data Challenge 2016](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2016/phm-data-challenge-4/).
2. Standardize wafer processing time in the raw CMP data to the same length, following the format in `MyCMP`.
3. Run `process_cmp.ipynb` to process, standardize, and split CMP data into training and test sets.
4. This will generate the following files inside a dataset folder:
```
| - {dataset folder} A456/B456
|  ├─ list.txt   # SVID name
|  ├─ train.csv  # Training set
|  ├─ test.csv   # Testing set
```
5. Move the dataset folder into the `data` folder of GDN-related models as follows:
```
Full_GDN_related
├─ data
│  ├─ A456
│  │  ├─ list.txt
│  │  ├─ train.csv
│  │  ├─ test.csv
│  ├─ B456
│  │  ├─ list.txt
│  │  ├─ train.csv
│  │  ├─ test.csv
```
6. Univariate time series models use a separate dataset file containing only MRR data.
7. If using other datasets, follow the same format as the processed CMP dataset.

### Running the Model
Each `.ipynb` file contains model building, training, and testing steps. Execute them completely to reproduce the results.

#### Full Measurement Scenario
- **Univariate Time Series Models:**
  - Run `TTS_model.ipynb` for ES, AR, ARIMA predictions.
  - Run `GRU_model.ipynb` for the GRU model (baseline).
- **Proposed Framework Models:**
  - Run `Random_GRU.ipynb` for Random Features + GRU model (baseline).
  - Run `GDN_GRU.ipynb` for the proposed health prognostics framework.

#### Sampling Measurement Scenario
1. **Construct the Virtual Metrology Model:**
   - Statistical Feature-based models in `VM_Stats_features`
   - GDN-based models in `VM_GDN_based`
   - Execute `.ipynb` files in these folders to train the models and predict unmeasured wafer measurements.

2. **Run the Baseline Models:**
   - **Univariate Time Series Models:**
     - `Sampling_TTS_model.ipynb`: ES, AR, ARIMA models.
     - `Sampling_GRU_model.ipynb`: GRU model.
   - **Proposed Framework Models:**
     - `Sampling_Random_GRU.ipynb`: Random Features + GRU.
     - `Sampling_GDN_GRU.ipynb`: Proposed framework.
     - `Sampling_Latest_Filling.ipynb`: Latest MRR Filling model.

> **Note:** Some models involve randomness, leading to slight variations in MSE.

## References
- GDN structure is based on [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series (AAAI'21)](https://arxiv.org/pdf/2106.06947.pdf).
- GDN layer implementation is adapted from [d-ailin/GDN](https://github.com/d-ailin/GDN.git).
