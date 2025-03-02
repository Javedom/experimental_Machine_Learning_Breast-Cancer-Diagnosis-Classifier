# Breast Cancer Diagnosis Classifier

## Overview
This project provides a machine learning model to predict breast cancer diagnosis (malignant or benign) based on features computed from digitized images of fine needle aspirates (FNA) of breast masses. The implementation uses a Random Forest classifier with optimized hyperparameters through both Grid Search and Bayesian Optimization.

## Features
- Supports both CSV and Excel file formats for input data
- Automatic encoding detection for CSV files
- Hyperparameter optimization using Grid Search and Optuna (Bayesian optimization)
- Comprehensive model evaluation with multiple metrics
- Feature importance analysis
- Color-coded Excel output for easy interpretation of results
- Visualization of model performance
- **Automatic model reuse** - tries to use a saved model first before training a new one
- Robust error handling with graceful fallbacks
- Contains data from https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Installation

### Prerequisites
- Python 3.7+

### Setup
1. Clone this repository:
   ```
   git clone https://github.com/Javedom/experimental_Machine_Learning_Breast-Cancer-Diagnosis-Classifier
   cd breast-cancer-classifier
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation
The classifier works with two types of datasets:

1. **Training data**: Must include a "diagnosis" column with values 'M' (malignant) or 'B' (benign)
2. **Prediction data**: Should have the same feature columns as the training data, but does not need a diagnosis column

### Running the Classifier

#### Smart Model Reuse (Default Behavior)
By default, the classifier first checks for a saved model and uses it if available. If no model is found, it automatically trains a new one:

#### Command Line Arguments

The classifier supports the following command line arguments:

```
--train TRAIN_FILE    Base name of training data file (without extension)
--predict PREDICT_FILE Base name of prediction data file (without extension)
--model MODEL_FILE    Path to saved model file
--force-train         Force training a new model even if a saved model exists
```

#### Using Default File Names
Place your files in the project directory with these names:
- `training_data.csv` or `training_data.xlsx` (for training data)
- `prediction_data.csv` or `prediction_data.xlsx` (for data to be classified)

Then run:
```
python breast_cancer_classifier.py
```

#### Using Custom File Names (Legacy Method)
```
python breast_cancer_classifier.py my_training_data new_samples
```
Note: Don't include file extensions - the program will check for both .csv and .xlsx versions.

#### Examples

Use a saved model with default filenames:
```
python breast_cancer_classifier.py
```

Force training a new model:
```
python breast_cancer_classifier.py --force-train
```

Use a saved model with custom prediction data:
```
python breast_cancer_classifier.py --predict new_samples
```

Use custom files and model path:
```
python breast_cancer_classifier.py --train wdbc_training --predict wdbc_new_cases --model custom_model.pkl
```

## Output Files

The classifier generates the following outputs:

1. **breast_cancer_predictions.xlsx**: 
   - Contains predictions for each sample in the prediction dataset
   - Includes color-coding (red for malignant, green for benign)
   - Shows probability of malignancy for each prediction

2. **model_evaluation_results.xlsx**:
   - Performance metrics (accuracy, precision, recall, F1 score, ROC AUC)
   - Formatted for easy interpretation

3. **breast_cancer_model.pkl**:
   - Saved model file with all components needed for prediction
   - Automatically used for future predictions unless --force-train is specified

4. **Visualization files**:
   - `confusion_matrix.png`: Visual representation of classification performance
   - `roc_curve.png`: ROC curve showing model discrimination
   - `feature_importance.png`: Bar chart of feature importance
   - `correlation_matrix.png`: Heatmap of feature correlations
   - `feature_distributions.png`: Box plots of key features by diagnosis

## Dataset Information

This implementation is designed for the Breast Cancer Wisconsin (Diagnostic) Dataset, but can be adapted for similar classification tasks.

### Features
The dataset includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass:

- radius (mean of distances from center to points on the perimeter)
- texture (standard deviation of gray-scale values)
- perimeter
- area
- smoothness (local variation in radius lengths)
- compactness (perimeter² / area - 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry
- fractal dimension ("coastline approximation" - 1)

For each feature, three values are provided:
- mean
- standard error (se)
- "worst" or largest (mean of the three largest values)

### Source
This dataset is available through:
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
- UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/

## Model Details

### Random Forest Classifier
The implementation uses a Random Forest classifier with optimized hyperparameters.

### Hyperparameter Tuning
The model performs hyperparameter optimization through:
1. **Grid Search**: Systematic exploration of predefined parameter values
2. **Bayesian Optimization (Optuna)**: Efficient search that learns from previous trials

The best model from these two approaches is selected for the final prediction.

### Model Reuse Workflow
The program follows this workflow:
1. Check if a saved model exists
2. If found and not overridden by --force-train, try to load it
3. If loaded successfully, use it for predictions
4. If no model exists or loading fails, train a new model
5. Save the newly trained model for future use

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

## Error Handling
The program includes robust error handling to manage common issues:
- Missing files
- Incompatible data formats
- Missing features
- Excel formatting issues (falls back to CSV)

## License
CC0

## Acknowledgments
About Dataset

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:
ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant
