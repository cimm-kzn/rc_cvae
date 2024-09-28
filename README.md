# RC_CVAE

Tensorflow implementation of the model described in the paper Conditional Variational AutoEncoder to Predict Suitable Conditions for Hydrogenation Reactions 

## Components
* **descriptors_generation.py** - functions for data preparation (x, y generation)
* **cvae_models.py** - functions for RC CVAE modeling
* **Run_VAEs.ipynb** - tutorial 
* **example_data/example_hydrogenation_USPTO.rdf** - example dataset
* **example_data/acids_bases_poisons_list.txt** - list of additives: acids, bases, catalytic poisons
* **example_data/catalysts_list.txt** - list of catalysts

## Data
Note: Because the Reaxys database is commercially available, we do not have permission to release the datasets used in this paper to the public. Instead, we provide an example dataset ("example_data/example_hydrogenation_USPTO.rdf") so that anyone can test the code. It contains some hydrogenation chemical reaction records from USPTO database, preliminary standardized.

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
@Article{Mazitov2024,
  title={Conditional variational autoEncoder to predict suitable conditions for hydrogenation reactions},
  author={Mazitov, Daniyar A. and Poyezzhayeva, Assima and Afonina, Valentina A. and Gimadiev, Timur R. and Madzhidov, Timur I.},
}exa
