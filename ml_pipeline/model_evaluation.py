from data_preprocessing import split_data, preprocess, load_data
import joblib 
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import seaborn as sns 
import matplotlib.pyplot as plt
from feature_selection import mutual_information
import pandas as pd
import os 

def evaluation(X_test, y_test):
    """
    Evaluate the performance of the model using test data.

    Parameters:
    X_test (pd.DataFrame): The test feature data.
    y_test (pd.Series): The test target variable.

    Returns:
    tuple: A tuple containing:
        - report_df (pd.DataFrame): A DataFrame with the classification report metrics.
        - cm (ndarray): The confusion matrix.
    """
    # Load the trained model from a file
    model = joblib.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'xgboost.pkl'))

    # Make predictions using the test data
    y_pred = model.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose() 

    # Print classification report
    print('Classification Report - Credit Score Classes')
    print(report_df)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix')
    print(cm)

    return report_df, cm

if __name__ == '__main__':
    # Load the data
    train_data, test_data = load_data()

    # Preprocess the data
    train_data_pre, test_data_pre, y_encode = preprocess(train_data, test_data)

    # Calculate mutual information for feature selection
    mi_info = mutual_information(train_data_pre, y_encode)

    # Identify features with mutual information less than the threshold
    unselected_features = mi_info[mi_info['Mutual_Information'] < 0.11]['Features']
    print(unselected_features)

    # Drop features with low mutual information from the training and testing datasets
    train_data_pre = train_data_pre.drop(unselected_features, axis=1)
    test_data_pre = test_data_pre.drop(unselected_features, axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(train_data_pre, y_encode)

    # Evaluate the model's performance on the test set
    evaluation(X_test, y_test)
    print(X_test.shape, X_train.shape)
