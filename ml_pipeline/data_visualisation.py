# Import necessary modules and functions
from data_preprocessing import split_data, preprocess, load_data
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from feature_selection import mutual_information
import pandas as pd
from model_evaluation import evaluation
import os

# Define a function to visualize the classification report
def visualise_classification_report(report_df):
    # Visualize the classification report as a heatmap
    
    # Metrics to include in the heatmap
    metrics = ['precision', 'recall', 'f1-score']
    
    # Extract classes (excluding the last three rows for accuracy, etc.)
    classes = report_df.index[:-3]
    
    # Filter the DataFrame to include only relevant metrics
    report_df_filtered = report_df.loc[classes, metrics]

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df_filtered, annot=True, cmap='Blues', fmt='.2f')
    plt.title("Classification Report \n Accuracy: {:.2f}%".format(report_df.loc['accuracy', 'precision'] * 100))
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    
    # Save the heatmap as an image file
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'classification_report.png'))

# Define a function to visualize the confusion matrix
def visualise_confusion_matrix(cm):
    # Compute and plot the confusion matrix
    
    # Create a ConfusionMatrixDisplay object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels={0: 'Poor', 1: 'Standard', 2: 'Good'})
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix - Credit Classes')
    
    # Save the confusion matrix as an image file
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'confusion_matrix.png'))

# Define a function to plot ROC AUC curves for multi-class classification
def roc_auc_curve(X_test, y_test):
    # Load the trained model from a file
    model = joblib.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'xgboost.pkl'))
    
    # Get prediction probabilities from the model
    y_pred = model.predict_proba(X_test)

    # Binarize the output for ROC curve calculation
    y_binarized = label_binarize(y_test, classes=[0, 1, 2])

    # Initialize dictionaries to store ROC curve data
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_binarized.shape[1]

    # Compute ROC curve and ROC AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red']  # Colors for each class
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    # Plot diagonal line representing random performance
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Multi-Class')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Save the ROC curve as an image file
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'roc_auc_curve.png'))

# Main execution block
if __name__ == '__main__':
    # Load and preprocess the data
    train_data, test_data = load_data()
    train_data_pre, test_data_pre, y_encode = preprocess(train_data, test_data)

    # Perform feature selection using mutual information
    mi_info = mutual_information(train_data_pre, y_encode)
    unselected_features = mi_info[mi_info['Mutual_Information'] < 0.11]['Features']
    print(unselected_features)

    # Drop unselected features from the datasets
    train_data_pre = train_data_pre.drop(unselected_features, axis=1)
    test_data_pre = test_data_pre.drop(unselected_features, axis=1)

    # Split the data into training and testing sets and apply SMOTE
    X_train, X_test, y_train, y_test = split_data(train_data_pre, y_encode)

    # Evaluate the model and get classification report and confusion matrix
    report_df, cm = evaluation(X_test, y_test)

    # Visualize the classification report, confusion matrix, and ROC curve
    visualise_classification_report(report_df)
    visualise_confusion_matrix(cm)
    roc_auc_curve(X_test, y_test)
