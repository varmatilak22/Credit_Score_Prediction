from data_preprocessing import split_data, load_data, preprocess
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import classification_report
from feature_selection import mutual_information
from dimensionality_reduction import find_optimal_components, principal_ca
import os

def training(X_train, y_train):
    """
    Train an XGBoost model with specified hyperparameters and save the trained model to a file.

    Parameters:
    X_train (pd.DataFrame): The training feature data.
    y_train (pd.Series): The training target variable.

    Saves:
    - The trained XGBoost model to a file named 'xgboost.pkl' in the 'model' directory.
    """
    # Initialize the XGBoost classifier with specified hyperparameters
    model = XGBClassifier(
        n_estimators=500,             # Number of boosting rounds (trees) to build
        max_depth=7,                  # Maximum depth of each tree
        learning_rate=0.3,            # Step size shrinkage used in update to prevent overfitting
        subsample=1.0,                # Fraction of samples used for fitting individual trees
        colsample_bytree=0.9,         # Fraction of features used for each tree
        min_child_weight=1,           # Minimum sum of instance weight (hessian) needed in a child
        gamma=0,                      # Minimum loss reduction required to make a further partition
        reg_alpha=0,                  # L1 regularization term on weights
        reg_lambda=1                  # L2 regularization term on weights
    )

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(model, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'xgboost.pkl'))
    
    print("!!!Model Saved Successfully!!!")

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
    
    # Train the model using the training data
    training(X_train, y_train)

    # Load the trained model
    model = joblib.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'xgboost.pkl'))

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Shapes of training and test data:", X_train.shape, X_test.shape)
