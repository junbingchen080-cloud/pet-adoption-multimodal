# pet-adoption-multimodal
Multimodal pet adoption speed prediction using image, text and tabular data.
# Multimodal Prediction of Pet Adoption Speed

This project predicts pet adoption speed using multimodal information from the PetFinder.my Adoption Prediction dataset. Three data modalities are used:

- Pet images
- Pet descriptions
- Tabular listing information

The project compares unimodal models with Early Fusion and Late Fusion approaches.

## Dataset

The project uses the **PetFinder.my Adoption Prediction** dataset from Kaggle.

Dataset page:  
https://www.kaggle.com/c/petfinder-adoption-prediction

The training dataset contains:

- 14,993 pet listings
- 58,311 training images
- 24 original tabular variables
- AdoptionSpeed labels from 0 to 4

After requiring both an image and a non-empty description, 14,640 complete multimodal samples were retained.

The original dataset is **not included in this repository**. It must be downloaded from Kaggle.

## Data Quality Assessment

The dataset was checked before modelling.

Main findings:

- No duplicated rows or duplicated PetIDs were found.
- 13 pet descriptions were missing.
- 1,265 pet names were missing.
- 14,652 pets had at least one image.
- 14,640 pets had both image and text information.
- No corrupted training images were detected.
- Image dimensions and aspect ratios varied substantially.
- AdoptionSpeed was imbalanced, with Class 0 representing only a small proportion of the dataset.

Because of the class imbalance, Macro F1, Weighted F1, Quadratic Weighted Kappa and confusion matrices were used for evaluation.

## Feature Extraction

### Image Features

A pretrained **ResNet18** model was used as a frozen image feature extractor.

The final classification layer was removed, producing a:

`512-dimensional feature vector`

for each image.

When a pet had multiple images, the image embeddings were mean-pooled to produce one 512-dimensional representation per pet.

### Text Features

Pet descriptions were encoded using:

`bert-base-multilingual-cased`

Descriptions were tokenised with a maximum sequence length of 256 tokens.

The final BERT hidden states were combined using mask-aware mean pooling, producing a:

`768-dimensional text representation`

for each pet.

### Tabular Features

The tabular data contained categorical and numerical variables.

Categorical features were processed using one-hot encoding and numerical variables were standardised.

The final processed tabular representation contained:

`347 features`

## Data Split

The data was divided into training, validation and test sets.

The final split used in the experiment contained:

- Training: 10,938 samples
- Validation: 1,754 samples
- Test: 1,948 samples

The same split indices were applied to image, text, tabular and target data so that all modalities remained aligned.

The test set was reserved for final model evaluation.

## Models

### Unimodal Baselines

Three unimodal XGBoost classifiers were trained:

1. Image-only model
2. Text-only model
3. Tabular-only model

These models were used as baselines for evaluating multimodal fusion.

### Early Fusion

Image, text and tabular features were concatenated:

- Image: 512 dimensions
- Text: 768 dimensions
- Tabular: 347 dimensions

This produced a combined:

`1627-dimensional feature vector`

which was passed to a single XGBoost classifier.

### Late Fusion

Separate XGBoost models were trained for each modality.

Their predicted class probabilities were combined using weighted averaging.

The fusion weights were selected using the validation set.

The selected weights were:

- Image: 0.2
- Text: 0.0
- Tabular: 0.8

## Final Results

| Model | Validation Macro F1 | Test Macro F1 | Test Weighted F1 | Test QWK |
|---|---:|---:|---:|---:|
| Image-only | 0.2602 | 0.2531 | 0.3246 | 0.2538 |
| Text-only | 0.2228 | 0.2106 | 0.2749 | 0.0920 |
| Tabular-only | 0.2861 | 0.2780 | 0.3557 | 0.3002 |
| Early Fusion | **0.2974** | 0.2725 | 0.3517 | **0.3289** |
| Late Fusion | 0.2889 | **0.2809** | **0.3611** | 0.3139 |

Late Fusion achieved the highest Test Macro F1 and Weighted F1.

Early Fusion achieved the highest Test Quadratic Weighted Kappa.

Among the unimodal models, the tabular model performed best.

## Error Analysis

The confusion matrices showed substantial differences between AdoptionSpeed classes.

Class 0 was particularly difficult to predict because it contained very few samples compared with the other classes.

Neither Early Fusion nor Late Fusion correctly classified a Class 0 sample in the test set.

Class 4 achieved the strongest recall among the five classes.

The difference between training and validation performance also showed substantial overfitting, particularly for the image, text and Early Fusion models.

## How to Run

The notebook was developed using Google Colab with GPU acceleration.

### 1. Clone or download this repository

Open:

`Pet_Adoption_Final_Project.ipynb`

in Google Colab.

### 2. Enable GPU

In Colab:

`Runtime → Change runtime type → GPU`

### 3. Obtain Kaggle API credentials

A Kaggle account and API credentials are required to download the PetFinder dataset.

Do not upload your Kaggle API token to this repository.

### 4. Run the notebook

Execute the notebook cells in order.

The notebook performs:

1. Dataset download and extraction
2. Data quality assessment
3. Image preprocessing and ResNet18 feature extraction
4. Text preprocessing and multilingual BERT feature extraction
5. Tabular preprocessing
6. Train/validation/test splitting
7. Unimodal baseline training
8. Early Fusion
9. Late Fusion
10. Model evaluation and confusion matrix analysis

## Main Python Libraries

The main libraries used are:

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- PyTorch
- torchvision
- transformers
- XGBoost
- Pillow
- OpenCV
- tqdm

## Privacy and Ethics

The dataset may contain information associated with pet owners or rescuers in descriptions or images.

The dataset is not redistributed in this repository.

Potential sources of bias include:

- Class imbalance
- Differences between rescuers
- Different image quality
- Different languages and writing styles

The model should be treated as a decision-support tool rather than a system for reducing the priority of particular animals.

## Repository Structure

```text
pet-adoption-multimodal/
│
├── Pet_Adoption_Final_Project.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Authors

Junbing Chen

## Acknowledgement

Dataset:

PetFinder.my Adoption Prediction, Kaggle.
