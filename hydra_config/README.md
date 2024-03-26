# Hydra Configurations

```config.yaml```
Base configuration for details of the purpose and backbone.

```
debug: False
mission: "train"    # or "test"
defaults:
    - dataset: FX
    - model: NSformer_ohlc
```

## 1. dataset
```FX.yaml```:
configurations regarding the dataset and the dataloader, need to be set by either changing directly in file or overriding in command line

## 2. dataset_model
Not used as of recent research, but is combination of configuration of dataset and model

## 3. model
```{model_name}.yaml```:
configurations regarding the model architecture, need to be set by either changing directly in file or overriding in command line

## Adding new configuration
In order to either 1) research on a new model and desire to train, or 2) train the model with a new dataset, new configuration yaml must be created with the equivalent naming convention. Then, it must be declared in ```main.py```
