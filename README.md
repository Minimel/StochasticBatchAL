# <center>Active Learning with Stochastic batches</center>

This is the code associated with the paper <b>"Active learning for medical image segmentation with stochastic batches"</b> by Melanie Gaillochet, Christian Desrosiers, and Herve Lombaert <br/> <br/>


## Setup
Install the following packages: 
`numpy matplotlib scikit-image monai h5py nibabel comet_ml flatten-dict pytorch_lightning pytorch-lightning-spells kornia torch torchvision`

Alternatively, create a virtual environments and install the packages of `requirements.txt`:
```
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Data
We assume the data folder (`data_dir`) has the following structure:

```
data
├── <dataset_name> 
│ └── raw
│   └── ...
│ └── preprocessed
│   └── train
|       └── data.hdf5
|       └── label.hdf5
|       └── preprocessing_info.json
|       └── scan_info.json
|       └── slice_info.json
│   └── test
|       └── data.hdf5
|       └── label.hdf5
|       └── preprocessing_info.json
|       └── scan_info.json
|       └── slice_info.json
```

An example of the preprocessing applied to the Hippocampus dataset is shown in the notebook `preprocessing_Hippocampus.ipynb` located in `JupyterNotebook` folder.<br/> <br/>

## Code

### Bash script
Experiments can be run with `Bash_scripts/bash_AL_hippocampus.sh <path-to-data_dir> <path-to-output_dir> <experiment_name> <sample_set_index> <train_config> <sampling_config> <seed>`<br/>
For prostate data, replace 'hippocampus' by 'prostate'. 

- `<experiment_name>` : can be any name under which we would like to save the experiment
- `<sample_set_index>` : integer from 1 to 5 to select the row with the corresponding initial indices to use (ie: 1 will select the first row of indices, 2, the second row, etc.) These indices are located in `Configs/init_indices/hippocampus_indices` or `Configs/init_indices/prostate_indices` 
- `<train_config>` : config file for training hyperparameters. Use `train_learningloss_config.yaml` if using learning Loss strategy (with `sampling_config/LearningLoss.yaml` sampling config), else `train_config.yaml`.
- `<sampling_config>` : name of config file in the form `sampling_config/<filename>`. Available config files are located in the folder `Configs/sampling_config`<br/> <br/>

Example: 
```
$ Bash_scripts/bash_AL_hippocampus.sh <DATA_DIR> <OUTPUT_DIR> entropy_stochasticbatches 1 train_config.yaml sampling_config/Entropy_SB.yaml 0
``` 

```
$ Bash_scripts/bash_AL_hippocampus.sh <DATA_DIR> <OUTPUT_DIR> learningloss 1 train_learningloss_config.yaml sampling_config/LearningLoss.yaml 0
``` 

Note: The code has been developed with comet_ml to track the experiments, but the default tensorboard logger can also be used. To use Comet ml, simply modify `logger_config`. <br/><br/>


### Main training function
The bash script runs the `main.py` file that trains the segmentation model with a subset of labeled images, for a predefined number of active learning cycles.
There are several input arguments, but most of them get filled in when using the appropriate bash script.

Input of `main.py`:
```
# These are the paths to the data and output folder
--data_dir        # path to directory where we saved our (preprocessed) data
--output_dir      # path to directory where we want to save our output model

# These are config file names located in src/Config
--data_config     
--model_config   
--train_config
--sampling_config   # Should be of the form: sampling_config/<config-file-name>
--logger_config 

# Additional training configs (seed and gpu)
--seed
--num_gpu      # number of GPU devices to use
--gpu_idx      # otherwise, gpu index, if we want to use a specific gpu

# Training hyper-parameters that we should change according to the dataset
# Note: arguments start with 'train__' if modifying train_config, 
# and with 'model__'  if modifying model_config, 
# Name is determined by hierarchical keywords to value in config, each separated by '__'
--model__out_channels             # number of output channels
--train__train_indices            # indices of labelled training data for the main segmentation task
--train__loss__normalize_fct      # softmax or sigmoid
--train__loss__n_classes          # number of output classes (depends on data)
--train__val_plot_slice_interval  # (optional) interval between 2 slices in a volume to be plotted and saved during validation (if we don't want to plot all slices)
```
