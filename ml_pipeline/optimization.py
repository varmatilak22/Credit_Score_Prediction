from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from data_preprocessing import load_data, split_data, preprocess
from feature_selection import mutual_information
from dimensionality_reduction import find_optimal_components, principal_ca
import numpy as np

def random_optimization(X_train, y_train, param_grid):
    """
    Perform hyperparameter optimization for an XGBoost model using Randomized Search.

    Parameters:
    X_train (pd.DataFrame): The training feature data.
    y_train (pd.Series): The training target variable.
    param_grid (dict): Dictionary with hyperparameter options to be searched.
    
    The function:
    - Initializes an XGBoost classifier.
    - Sets up RandomizedSearchCV to find the best hyperparameters.
    - Fits the model on the training data.
    - Prints the best hyperparameters and their corresponding accuracy score.
    """
    # Initialize the XGBoost classifier
    xgb_model = XGBClassifier()

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_model, 
        param_distributions=param_grid, 
        scoring='accuracy', 
        cv=3, 
        verbose=1, 
        n_jobs=-1,
        random_state=42  # For reproducibility
    )

    # Fit the model
    random_search.fit(X_train, y_train)

    # Get the best hyperparameters and their corresponding accuracy score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("Best Hyperparameters:", best_params)
    print("Best Accuracy Score:", best_score)


if __name__ == '__main__':
    # Load the dataset
    train_data, test_data = load_data()

    # Preprocess the data
    train_data_pre, test_data_pre, y_encode = preprocess(train_data, test_data)
    
    # Compute mutual information for feature selection
    mi_info = mutual_information(train_data_pre, y_encode)
    
    # Identify features with mutual information less than the threshold
    unselected_features = mi_info[mi_info['Mutual_Information'] < 0.11]['Features']
    print("Features with low mutual information:", unselected_features)
    
    # Drop features with low mutual information from the training and testing datasets
    train_data_pre = train_data_pre.drop(unselected_features, axis=1)
    test_data_pre = test_data_pre.drop(unselected_features, axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(train_data_pre, y_encode)

    # Apply dimensionality reduction using PCA
    X_train_pca, X_test_pca = principal_ca(X_train, X_test)

    # Define the parameter grid for Randomized Search
    param_grid = {
        'n_estimators': np.arange(100, 1001, 100),                # Number of boosting rounds
        'learning_rate': np.linspace(0.01, 0.3, 10),             # Step size shrinkage
        'max_depth': np.arange(3, 11, 1),                         # Maximum depth of each tree
        'min_child_weight': np.arange(1, 11, 1),                  # Minimum sum of instance weight needed in a child
        'subsample': np.linspace(0.5, 1, 6),                     # Fraction of samples used for fitting trees
        'colsample_bytree': np.linspace(0.5, 1, 6),              # Fraction of features used for each tree
        'gamma': np.linspace(0, 5, 10),                          # Minimum loss reduction required to make a further partition
        'reg_alpha': np.linspace(0, 1, 10),                      # L1 regularization term on weights
        'reg_lambda': np.linspace(0, 1, 10)                      # L2 regularization term on weights
    }

    # Call the hyperparameter optimization function
    random_optimization(X_train_pca, y_train, param_grid)
