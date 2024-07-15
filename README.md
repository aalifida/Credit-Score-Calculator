# Credit Scoring using Logistic Regression

This repository contains a Python script for training a logistic regression model to predict credit scores based on a dataset (`a_Dataset_CreditScoring.xlsx`). The model is evaluated using standard evaluation metrics and predictions are saved to a CSV file for further analysis.

## Requirements

To run the script, ensure you have Python 3.x installed along with the following libraries:

- pandas
- numpy
- scikit-learn

You can install these dependencies using pip:

```
pip install pandas numpy scikit-learn
```

## Usage

1. **Clone the Repository:**

   ```
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install Dependencies:**

   ```
   pip install -r requirements.txt
   ```

3. **Run the Script:**

   ```
   python credit_scoring_logistic_regression.py
   ```

4. **Output:**

   - The script will print the confusion matrix and accuracy score of the model on the test set.
   - Predictions along with actual outcomes and predicted probabilities will be saved to `c1_Model_Prediction.csv`.

## Script Explanation

### Loading and Preprocessing Data

- The dataset (`a_Dataset_CreditScoring.xlsx`) is loaded using pandas (`pd.read_excel()`).
- Missing values are filled with the mean of each column (`dataset.fillna(dataset.mean())`).
- The 'ID' column is dropped assuming it is not relevant to the modeling (`dataset.drop('ID', axis=1)`).

### Training and Evaluation

- Features (`X`) and target variable (`y`) are separated from the dataset.
- The dataset is split into training and test sets using `train_test_split()` with 80% for training and 20% for testing.
- Features are standardized using `StandardScaler()` to ensure all features have mean 0 and variance 1.
- Logistic Regression model (`LogisticRegression()`) is initialized and trained on the training set (`classifier.fit(X_train, y_train)`).

### Prediction and Metrics

- Predictions (`y_pred`) are made on the test set using `classifier.predict(X_test)`.
- Evaluation metrics such as confusion matrix (`confusion_matrix(y_test, y_pred)`) and accuracy score (`accuracy_score(y_test, y_pred)`) are computed and printed.

### Saving Predictions

- Predicted probabilities for each class (`predictions`) are generated using `classifier.predict_proba(X_test)`.
- DataFrames (`df_prediction_prob`, `df_prediction_target`, `df_test_dataset`) are created for predictions, actual outcomes, and predicted targets respectively.
- These DataFrames are concatenated and saved to `c1_Model_Prediction.csv` using `dfx.to_csv()`.

## Future Enhancements

- **Hyperparameter Tuning**: Explore grid search or other techniques to optimize model performance.
- **Deployment**: Consider deploying the trained model in a production environment using frameworks like Flask or FastAPI.


