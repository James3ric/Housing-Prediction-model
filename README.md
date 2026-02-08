# Housing Price Prediction Model

A machine learning project to predict median house prices using the California Housing Dataset. This project implements a complete ML pipeline including data exploration, preprocessing, model selection, and hyperparameter optimization.

## Overview

This project uses regression models to predict median house values based on features like location (latitude/longitude), property characteristics (age, rooms, bedrooms), and demographic data (population, income). The dataset contains 20,640 housing records from California with 9 features and 1 target variable.

## Dataset

- **Source**: California Housing Dataset
- **Samples**: 20,640 house records
- **Features**: 9 (8 numerical + 1 categorical)
- **Target**: `median_house_value` (USD)

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `longitude` | float | Geographic longitude |
| `latitude` | float | Geographic latitude |
| `housing_median_age` | float | Age of the housing block |
| `total_rooms` | float | Total number of rooms |
| `total_bedrooms` | float | Total number of bedrooms |
| `population` | float | Population in the area |
| `households` | float | Number of households |
| `median_income` | float | Median household income |
| `ocean_proximity` | categorical | Location relative to ocean (5 categories) |

## Project Structure

```
housing-prediction-model/
├── housing.csv              # Dataset
├── prediction.ipynb         # Main notebook with full analysis
└── README.md               # This file
```

## Key Findings from Exploratory Data Analysis

- **Missing Values**: Only `total_bedrooms` has 207 missing values (1% of data)
- **No Duplicates**: Dataset is clean with no duplicate records
- **Target Distribution**: Right-skewed, with a ceiling at $500,001
- **Strong Predictor**: `median_income` has the highest correlation with house prices (r = 0.688)
- **High Multicollinearity**: Room and population features are highly correlated
- **Outliers**: Several features show outliers and skewed distributions

## Methodology

### 1. Data Preprocessing
- **Numerical Features**: Median imputation for missing values + StandardScaler normalization
- **Categorical Features**: Most frequent imputation + One-Hot Encoding
- **Train-Test Split**: 80-20 split with random state for reproducibility

### 2. Model Selection
Evaluated multiple regression models using 5-fold cross-validation:
- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- HistGradientBoosting Regressor

### 3. Evaluation Metrics
- **Primary**: Root Mean Squared Error (RMSE)
- **Secondary**: Mean Absolute Error (MAE), R² Score

## Results

### Baseline Model (Linear Regression)

| Set | RMSE | MAE | R² |
|-----|------|-----|-----|
| Training | 68,433.94 | 49,594.84 | 0.650 |
| Test | 70,059.19 | 50,670.49 | 0.625 |

The baseline model explains approximately 62.5% of the variance in house prices on the test set.

## Installation

### Requirements
- Python 3.8+
- Jupyter Notebook

### Dependencies
```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

Or install from requirements (if available):
```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Housing-Prediction-model.git
cd Housing-Prediction-model
```

2. **Open the notebook**:
```bash
jupyter notebook prediction.ipynb
```

3. **Run all cells** to execute the complete pipeline:
   - Data loading and exploration
   - Preprocessing
   - Model training and evaluation
   - Cross-validation results

## Model Pipeline

The project uses scikit-learn pipelines to ensure proper data handling and avoid data leakage:

```
Pipeline
├── Preprocessing
│   ├── Numerical: Imputation → Scaling
│   └── Categorical: Imputation → One-Hot Encoding
└── Model
    ├── Linear Regression
    ├── Ridge/Lasso
    ├── Random Forest
    └── HistGradientBoosting
```

## Future Improvements

- [ ] Feature engineering (e.g., rooms per household, income per capita)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Ensemble methods combining multiple models
- [ ] Address data patterns (right-skewed target, ceiling effect)
- [ ] Implement model deployment (Flask/FastAPI)
- [ ] Add cross-validation visualization
- [ ] Feature importance analysis
- [ ] Handling outliers and data imbalance

## Contributing

Contributions are welcome! Feel free to:
- Report issues or bugs
- Suggest improvements
- Submit pull requests with enhancements

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Dataset: [California Housing Dataset](https://www.kaggle.com/datasets/codebreaker/california-housing-prices)
- Built with: scikit-learn, pandas, numpy, matplotlib, seaborn
