# Protein Function Prediction from Amino Acid Sequences

This project provides a series of Jupyter notebooks to train a deep learning model to predict functional properties of proteins (via their *Go terms*) based solely on their amino acid sequences.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Data Preparation and Training Pipeline](#data-preparation-and-training-pipeline)
4. [Installation](#installation)
5. [Notebook Usage](#notebook-usage)
6. [Performance](#performance)

---

## Project Overview

The goal of this project is to develop a deep learning model capable of predicting protein functional properties represented by *Go terms* (Gene Ontology). Using high-quality data from the Swiss-Prot corpus, we transform amino acid sequences into a tensor format for training a model based on a *Transformer Decoder Only* architecture. The model is further refined to account for class imbalances, maximizing the prediction accuracy for each *Go term*.

## Data

The dataset used comes from **Swiss-Prot**, a manually annotated and well-curated database containing around 550,000 unique proteins. This corpus includes high-quality annotations for each protein, making it a reliable source for training robust models. The data is provided in JSON format.

This dataset can be accessed here: [UniProtKB](https://www.uniprot.org/uniprotkb?query=*&facets=reviewed%3Atrue)

## Data Preparation and Training Pipeline

The data preparation and training pipeline consists of several stages, each implemented in a distinct Jupyter notebook for clarity and modularity.

1. **Data Filtering**
    - **filter_data_1.ipynb**: Extraction of essential information and filtering proteins by length to limit the model’s context size.
    - **filter_data_2.ipynb**: Filtering classes by occurrence count to avoid overly rare classes.
    - **filter_data_3.ipynb**: Excluding proteins with rare amino acids (X, U, Z…) to limit the size of the input vocabulary.

2. **Data Preparation**
    - **prepare_data.ipynb**: Transforming data into a PyTorch-compatible tensor, ready for model training.

3. **Model Training**
    - **train_1.ipynb**: Training a model based on *Transformer Decoder Only* architecture using Binary Cross Entropy to minimize loss.
    - **train_2.ipynb**: Calculating optimal thresholds for each class by maximizing the F1 score on the training data to address class imbalances.

4. **Performance Evaluation**
    - **test.ipynb**: Evaluating model performance on the test set by calculating precision, recall, and accuracy to assess prediction effectiveness.

## Notebook Usage

Each notebook represents a step in the data processing, training, and evaluation pipeline. To reproduce similar results:

1. Follow the notebooks in the order indicated in the [Data Preparation and Training Pipeline](#data-preparation-and-training-pipeline) section.
2. Adjust training or filtering parameters as needed to adapt the model to specific data requirements.
3. Review metrics in `test.ipynb` to analyze the model's final performance.

**Note**: Model training may take significant time, with approximately 2 hours per iteration on an AMD 6800XT GPU.

## Performance

The model was evaluated using the following metrics:
- **Precision**: Measures the model's ability to avoid false positives.
- **Recall**: Measures the model's ability to correctly identify *Go terms*.
- **Accuracy**: Overall rate of correct predictions.