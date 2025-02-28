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
- Contains demo data from https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Installation

### Prerequisites
- Python 3.7+

### Setup
1. Clone this repository:
   ```
   git clone <repository-url>
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

#### Using Default File Names
Place your files in the project directory with these names:
- `training_data.csv` or `training_data.xlsx` (for training data)
- `prediction_data.csv` or `prediction_data.xlsx` (for data to be classified)

Then run:
```
python breast_cancer_classifier.py
```

#### Using Custom File Names
```
python breast_cancer_classifier.py my_training_data new_samples
```
Note: Don't include file extensions - the program will check for both .csv and .xlsx versions.

### Example
```
python breast_cancer_classifier.py wdbc_training wdbc_new_cases
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
   - Can be loaded for future use without retraining

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

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

## License
[Include appropriate license information]

## Acknowledgments
- Original Wisconsin Breast Cancer Dataset by Dr. William H. Wolberg, University of Wisconsin Hospitals
- UCI Machine Learning Repository for hosting the dataset
