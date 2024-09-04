from data_preprocessing import load_data, split_data, preprocess
from model_training import training
from model_evaluation import evaluation
from feature_selection import mutual_information
from data_visualisation import visualise_classification_report, visualise_confusion_matrix, roc_auc_curve

def run_pipeline():
    """
    Executes the complete machine learning pipeline including data processing, feature selection,
    model training, evaluation, and visualization.

    The pipeline performs the following steps:
    1. **Data Collection/Extraction**: Loads the dataset.
    2. **Data Preprocessing**: Cleans and preprocesses the data for training.
    3. **Feature Selection**: Applies feature selection techniques to identify and remove irrelevant features.
    4. **Model Training**: Trains the machine learning model using the processed data.
    5. **Model Evaluation**: Evaluates the trained model using metrics such as classification report and confusion matrix.
    6. **Data Visualization**: Generates visualizations for model evaluation, including classification report, confusion matrix, and ROC-AUC curve.

    Steps:
    1. **Load Data**:
        - Uses `load_data()` to extract the training and test datasets.
    2. **Preprocess Data**:
        - Applies `preprocess()` to clean and prepare the data.
    3. **Feature Selection**:
        - Utilizes `mutual_information()` to compute mutual information scores for features.
        - Identifies and removes features with low mutual information.
    4. **Split Data**:
        - Uses `split_data()` to create training and testing sets.
    5. **Train Model**:
        - Calls `training()` to train the model on the training data.
    6. **Evaluate Model**:
        - Uses `evaluation()` to generate a classification report and confusion matrix.
    7. **Visualize Results**:
        - `visualise_classification_report()` displays the classification report as a heatmap.
        - `visualise_confusion_matrix()` displays the confusion matrix as a plot.
        - `roc_auc_curve()` plots the ROC-AUC curve to assess model performance across different classes.

    This function serves as the main entry point for executing the entire machine learning workflow.
    """
    # Data Collection/Extraction
    train_data, test_data = load_data()

    # Data Preprocessing 
    train_data_pre, test_data_pre, y_encode = preprocess(train_data, test_data)

    # Apply Feature Selection like Mutual Information
    mi_info = mutual_information(train_data_pre, y_encode)

    # Remove irrelevant columns
    unselected_features = mi_info[mi_info['Mutual_Information'] < 0.11]['Features']
    print("Features with low mutual information:", unselected_features)

    # Drop those columns
    train_data_pre = train_data_pre.drop(unselected_features, axis=1)
    test_data_pre = test_data_pre.drop(unselected_features, axis=1)

    # Make Training and Test sets 
    X_train, X_test, y_train, y_test = split_data(train_data_pre, y_encode)
    print("Training features:", X_train.columns)

    # Model Training
    training(X_train, y_train)

    # Model Evaluation
    report_df, cm = evaluation(X_test, y_test)

    # Data Visualization 
    # Classification report 
    visualise_classification_report(report_df)
    
    # Confusion Matrix 
    visualise_confusion_matrix(cm)
    
    # ROC-AUC Curve
    roc_auc_curve(X_test, y_test)


if __name__ == '__main__':
    run_pipeline()
