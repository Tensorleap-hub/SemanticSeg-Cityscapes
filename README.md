# Semantic Segmentation: Cityscapes


The model: which is a semantic segmentation model: [DeepLabV3Plus](https://github.com/VainF/DeepLabV3Plus-Pytorch) was imported into Tensorleap along with the the dataset the model was trained on: Cityscapes dataset. 






This quick start guide will walk you through the steps to get started with this example repository project.

## Tensorleap **CLI Installation**


#### Prerequisites

Before you begin, ensure that you have the following prerequisites installed:

- **[Python](https://www.python.org/)** (version 3.7 or higher)
- **[Poetry](https://python-poetry.org/)**

<br>


## Tensorleap CLI Usage

### Tensorleap **Login**

To login to Tensorealp:

```
leap auth login [api key] [api url].

```

- API Key is your Tensorleap token (see how to generate a CLI token in the section below).
- API URL is your Tensorleap environment URL: CLIENT_NAME.tensorleap.ai


## Tensorleap **Dataset Deployment**

To deploy your local changes:

```
leap project push models/DeeplabV3.h5
```

### **Tensorleap files**

Tensorleap files in the repository include `leap_binder.py` and `leap.yaml`. The files consist of the  required configurations to make the code integrate with the Tensorleap engine:

**leap.yaml**

leap.yaml file is configured to a dataset in your Tensorleap environment and is synced to the dataset saved in the environment.

For any additional file being used we add its path under `include` parameter as in this project [.yaml file](https://github.com/Tensorleap-hub/DeepLabV3Plus-ADAS/blob/8d2fa17c7f7b4e23f8c2c798aec1d8703fcbb903/leap.yaml). 

**leap_binder.py file**

`leap_binder.py` configure all binding functions used to bind to Tensorleap engine. These are the functions used to evaluate and train the model, visualize the variables, and enrich the analysis with external metadata variables

## Testing

To test the system we can run `leap_test.py` file using poetry:

```
poetry run test
```

This file will execute several tests on leap_binder.py script to assert that the implemented binding functions: preprocess, encoders,  metadata, etc,  run smoothly.

*For further explanation please refer to the [docs](https://docs.tensorleap.ai/)*
