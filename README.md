# RC_CVAE

Tensorflow implementation of the model described in the paper Conditional Variational AutoEncoder to Predict Suitable Conditions for Hydrogenation Reactions 

## Components
* **descriptors_generation.py** - functions for data preparation (x, y generation)
* **cvae_models.py** - functions for RC CVAE modeling
* **Run_VAEs.ipynb** - tutorial 
* **data/** - Dataset directory containing some data used in this study
* **data/Reaxys_IDs/** - Reaxys database identifiers for the test sets used in models evaluation.
* **data/example_data/example_hydrogenation_USPTO.rdf** - example dataset
* **data/example_data/acids_bases_poisons_list.txt** - list of additives: acids, bases, catalytic poisons
* **data/example_data/catalysts_list.txt** - list of catalysts

## Data
Note: Due to the commercial nature of the Reaxys database, we are unable to publicly release the complete datasets used in this study. Instead, we provide Reaxys IDs for the test sets (data/Reaxys_IDs/)  and an example dataset of hydrogenation reactions from the USPTO database ("data/example_data/example_hydrogenation_USPTO.rdf") to enable code testing and validation. These hydrogenation reactions have been preliminarily standardized.

## Usage
Example of usage can be found in Run_VAEs.ipynb

## Dependencies

Only python 3.6

* numpy == 1.18.1
* tensorflow-gpu == 2.1.0 ; python_version == '3.6'
* tensorflow == 2.1.0
* tensorflow_probability == 0.9.0
* keras == 2.2.4
* h5py == 2.10.0
* git+https://github.com/cimm-kzn/s-vae-tf.git
* cgrtools == 3.1.9
* CIMtools==3.1.0
* [Fragmentor](https://github.com/cimm-kzn/CIMtools/tree/master/Fragmentor)

## Installation

1. Install dependencies from requirements.py
2. Download Fragmentor 2017 from https://github.com/cimm-kzn/CIMtools/tree/master/Fragmentor based on your OS and add it to bin/ folder of your virtual environment
3. Rename Fragmentor file to fragmentor-2017.x

## Citation
@Article{Mazitov2025,
  title={Conditional variational autoEncoder to predict suitable conditions for hydrogenation reactions},
  author={Mazitov, Daniyar A. and Poyezzhayeva, Assima and Afonina, Valentina A. and Gimadiev, Timur R. and Madzhidov, Timur I.},
}exa
