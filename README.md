# Thesis_code_Jim Wu
This repo contain all the code implementation of the thesis "On the Development of Graph Deviation Networks-based Equipment Health Prognostics Framework".

Description of the folders  
```
    Code
    |-CMP_Dataset                                  # dataset of case study
    | |-2016 PHM DATA CHALLENGE CMP DATA SET           # original data
    | |-MyCMP                                          # data after synchronization
    | |-preprocess_cmp.ipynb                           # code for data preprocessing
    |-Data_Plot                                    # data visualization
    | |-{dataset folder}                               # generated from preprocess_cmp.ipynb
    | |-data_plot.ipynb                                # code for Figure 35 and 36 in the thesis
    |-Full_Scenario                                # models for full measurement scenario   
    | |-Full_Univariate_time_series                    # models based on univariate time series
    | | | ...
    | | |-TTS_model.ipynb                                  # included ES, AR, ARIMA models (baseline)
    | | |-GRU_modell.ipynb                                 # GRU model (baseline)
    | |-Full_GDN_related                               # models based on the proposed framework-based 
    | | | ...
    | | |-GDN_GRU.ipynb                                    # proposed framework
    | | |-Random_GRU.ipynb                                 # Random Features + GRU model (baseline)
    | | |-Stats_GRU.ipynb                                  # Statistical Features + GRU model (baseline)
    |-Sampling_Scenario                            # models for sampling measurement scenario  
    | |-Sampling_Univariate_time_series                # models based on univariate time series models
    | | | ...
    | | |-Sampling_TTS_model.ipynb                         # included ES, AR, ARIMA models (baseline)
    | | |-Sampling_GRU_model.ipynb                         # GRU model (baseline)
    | |-Sampling_GDN_related                           # models based on the proposed framework-based 
    | | | ...
    | | |-Sampling_GDN_GRU.ipynb                           # proposed framework
    | | |-Sampling_Latest_Filling                          # Latest Filling model (baseline)
    | | |-Sampling_Random_GRU.ipynb                        # Random Features + GRU model (baseline)
    | | |-Sampling_Stats_GRU.ipynb                         # Statistical Features + GRU model (baseline)
    | |-VM_Stats_features                          
    | | | ...                           
    | | |-Stats_VM.ipynb                               # Statistical Features VM
    | |-VM_GDN_based
    | | | ...
    | | |-GDN_VM.ipynb                                 # GDN-based VM
```


# Installation
### Environment
* WIN 11
* NVIDIA GeForce RTX 3090 Ti
  
### Requirement
* Python >= 3.6
* cuda == 11.8
* [Pytorch==2.1.0](https://pytorch.org/)
* [PyG: torch-geometric==1.7.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Packages
* numpy
* pandas
* matplotlib
* seaborn
* sklearn
* statsmodels
* pmdarima
* xgboost

### Quickstart
```
    # run the following command in the terminal to create a virtual environment.
    conda env create --file environment.yaml
```

# Usage
### Data Preparation
* The CMP dataset used in this research is sourced from [PHM Data Challenge 2016](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2016/phm-data-challenge-4/).  
* First, the wafer processing time in the raw CMP data needs to be standardized to the same length, with the format matching the files in the 'MyCMP' folder.
* Next, run the code in ```process_cmp.ipynb.``` The CMP data will be formatted, standardized, and split into training and test sets. After execution, the following folder will be generated, containing three files: ```list.txt```, ```train.csv```, and ```test.csv```.
```
    | -{dataset folder} A456/B456
    | |-list.txt   # SVID name
    | |-train.csv  # training set
    | |-test.csv   # testing set
```
* Finally, place this dataset folder into the ```data``` folder of the GDN-related models as the sample below.
```
    Full_GDN_related
    |-data
    | | |-A456
    | | | |-list.txt
    | | | |-train.csv
    | | | |-test.csv
    | | |-B456
    | | | |-list.txt
    | | | |-train.csv
    | | | |-test.csv
```
* For univariate time series models, we have separately prepared a file that contains only MRR data.
* If using other datasets, please follow the formatted structure of the processed CMP dataset.

### Run Model
* Each ```.ipynb``` file includes model building, training and testing. Please execute them completely to reproduce the experimental results.
* For full measurement scenario, there are two categories of models in ```Full_Univariate_time_series``` and ```Full_Framework_related``` folders.
    ```Full_Univariate_time_series``` 
        Run ```TTS_model.ipynb``` to obtain prediction results of ES, AR, and ARIMA models, and run ```GRU_model.ipynb``` for the result of the GRU model, they are all baseline model.

    ```Full_Framework_related```
        Run ```Random_GRU.ipynb.ipynb``` to obtain prediction results of the Random Features + GRU model, and ```GDN_GRU.ipynb.ipynb``` for the results of Statistical Features + GRU model, they are both baseline model.
        Run ```GDN_GRU.ipynb.ipynb``` to obtain prediction results of the our proprosed healht prognostic framework.

* For Sampling measurement scenario, there are also two categories of models in ```Sampling_Univariate_time_series``` and ```Sampling_Framework_related``` folders.
    First, we construct the virtual metrology model, which is divided into two approaches: Statistical Feature-based and GDN-based, stored in ```VM_Stats_features``` and ```VM_GDN_based``` folders. By executing the ```.ipynb``` file in each folder, you can train the model and predict the unmeasured wafers' measurement value. (place the dataset in the 'data' folder as well.)

    Next, according to the explanation in the thesis, we select the Statistical Features + RF VM model. Therefore, we directly include it in the execution files of the following models.
    ```Sampling_Univariate_time_series```
        Run ```Sampling_TTS_model.ipynb``` to obtain prediction results of ES, AR, and ARIMA models, and run ```Sampling_GRU_model.ipynb``` for the result of the GRU model, they are all baseline model.

    ```Sampling_Framework_related```
        Run ```Sampling_Random_GRU.ipynb.ipynb``` to obtain prediction results of the Random Features + GRU model, ```Sampling_GDN_GRU.ipynb.ipynb``` for the results of Statistical Features + GRU model, and ```Sampling_Latest_Filling.ipynb.ipynb``` for the results of Latest MRR Filling model, they are both baseline model.
        Run ```Sampling_GDN_GRU.ipynb.ipynb``` to obtain prediction results of the our proprosed healht prognostic framework.

* Some models have inherent randomness, so the MSE may vary slightly.

# Reference
* The GDN structure is referenced from [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series(AAAI'21)](https://arxiv.org/pdf/2106.06947.pdf) 
* The code for GDN layer is derived from [a-ailin/GDN](https://github.com/d-ailin/GDN.git).