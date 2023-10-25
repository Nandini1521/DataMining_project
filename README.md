# DataMining_project


## Building an ensemble classifier:

This code performs data preprocessing and classification using cross validation and ensemble techniques. The goal is to create an ensemble classifier that combines the strengths of multiple machine learning classifiers to achieve better predictive performance.

## Code Structure

- `main` function: The main entry point for the code. Instructs the entire workflow, including data preprocessing, classifier selection, model evaluation, and generating the predictions.

- `preprocess_data` function: Performs data preprocessing, including imputing missing values for numerical and categorical data and normalizing numerical features.

- `classifiers` function: Evaluates different classifiers, such as Decision Tree, Random Forest, K-Nearest Neighbors, and Gaussian Naive Bayes. It selects the best hyperparameters for each classifier and returns the best classifiers along with their F1 scores.

- `preprocess_combos` function: Applies various combinations of preprocessing techniques to the input dataset and returns a list of preprocessed datasets.

- Hyperparameter Grids: Specifies hyperparameter grids for each classifier to perform a grid search for hyperparameter tuning.

## Usage

1. Ensure you have Python and the required libraries (e.g., pandas, numpy, scikit-learn) installed on your system.

2. Prepare the training and test data in CSV format. Make sure the training data includes both the main dataset and additional data.

3. Include the data file paths in the `main` function for `main_train` and `main_test` dataframes.

4. Run the code by executing the script. The `main` function will execute the entire workflow, including preprocessing, classifier selection, and evaluation.

5. The final ensemble classifier will be used to generate predictions on the test dataset.

6. The predictions will be saved to a CSV file named `s4593940.csv`.
   

## Model Evaluation

The code uses k-fold cross-validation to evaluate the performance of the ensemble classifier and individual classifiers. It computes F1 scores and accuracy scores.

## Output

The final predictions are stored in a CSV file named `s4593940.csv`. The file contains the predictions for the test dataset. The average F1 score and average accuracy score of the cross validated training data, fit to the ensemble classifier, were also included manually into this csv file.

