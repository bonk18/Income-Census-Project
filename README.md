
# Income Census Prediction Project 
## Overview
This project leverages the Adult Income Census Dataset, also known as the “Adult” dataset, to predict whether a person’s annual income exceeds $50,000. The project employs various machine learning techniques, along with data manipulation and visualization tools, to build, train, and evaluate predictive models.

## Objective
To create a machine learning model that can classify individuals based on whether their income exceeds $50,000, using features such as age, education, occupation, and hours worked per week.

## Dataset
The dataset, sourced from the UCI Machine Learning Repository, contains 48,842 records with both categorical and numerical attributes. The features include:
- Age  
- Workclass  
- Education  
- Marital Status  
- Occupation  
- Race  
- Gender  
- Hours-per-week  
- Native Country  
- Income (target)
## Libraries Used
- Scikit-learn: For machine learning algorithms and model evaluation.
- Pandas: For data manipulation and cleaning.  
- NumPy: For numerical operations.  
- Seaborn: For data visualization through advanced statistical plots.
- Matplotlib: For basic plotting and visual representation of the dataset.
## Workflow
### Data Preprocessing:

Handling missing values.
Encoding categorical variables.
Scaling numerical data for model input.  
### Exploratory Data Analysis (EDA):

Visualizing feature distributions using Seaborn and Matplotlib.
Analyzing correlations and patterns in the dataset.
Addressing class imbalance if necessary.  

### Modeling:

Applying machine learning algorithms such as:
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines
- Hyperparameter tuning to optimize model performance.

### Model Evaluation:

Using accuracy, precision, recall, and F1-score to assess performance.  
Analyzing confusion matrices for classification results.  
Cross-validation to ensure the robustness of the model.   
## Results
The final model was selected based on its performance metrics, particularly its ability to balance precision and recall while achieving high accuracy. Insights from the project can help identify the most significant factors influencing income level predictions.

## Future Improvements
Experimenting with other machine learning algorithms (e.g., XGBoost, Neural Networks).
Implementing feature engineering for more robust predictions.
Addressing bias and fairness in income predictions.
## How to Run the Project
Clone the repository:
git clone https://github.com/your-repo/income-census-project.git

Install the required libraries:
pip install -r requirements.txt

Run the Jupyter Notebook or Python script:
jupyter notebook
or
python main.py