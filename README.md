# heart_stroke_detection using R 

## Overview
This project aims to predict the likelihood of heart stroke using statistical models in **R**. The analysis is based on a dataset containing health and medical data, which helps in building predictive models to assess stroke risk factors.

### Key Components:
- **Code**: R script (`heart_stroke.R`) for data cleaning, feature engineering, model training, and evaluation.
- **Dataset**: The `healthcare-dataset-stroke-data.csv` file contains patient information and stroke indicators.

## How to Use

### R Setup:
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/your-username/heart-stroke-prediction.git
    ```

2. Load the dataset in R using:
    ```R
    data <- read.csv("healthcare-dataset-stroke-data.csv")
    ```

3. Source the R script to run the analysis:
    ```R
    source("heart_stroke.R")
    ```

4. The script will:
   - Perform data preprocessing such as handling missing values and encoding categorical variables.
   - Build machine learning models like logistic regression or decision trees.
   - Evaluate model performance using various metrics.

### Key Features:
- Data preprocessing: Cleaning and transforming the dataset for optimal model performance.
- Model building: Using different algorithms to predict the probability of stroke.
- Evaluation: Assessing model accuracy and other key metrics for performance.


## Usage
The dataset and code are structured to analyze key health risk factors contributing to strokes. The models predict the likelihood of stroke based on patient demographics and health conditions.
