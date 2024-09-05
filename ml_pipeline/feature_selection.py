from sklearn.feature_selection import mutual_info_classif
from data_preprocessing import split_data, preprocess, load_data
import pandas as pd

def mutual_information(data, y):
    """
    Calculate the mutual information between each feature and the target variable.

    Parameters:
    data (pd.DataFrame): The feature data.
    y (pd.Series): The target variable.

    Returns:
    pd.DataFrame: A DataFrame with features and their corresponding mutual information scores, sorted in descending order.
    """
    # Calculate mutual information for each feature
    mi = mutual_info_classif(data, y)

    # Create a DataFrame to store features and their mutual information scores
    mi_df = pd.DataFrame({'Features': data.columns, 'Mutual_Information': mi})

    # Sort the DataFrame by mutual information scores in descending order
    mi_df = mi_df.sort_values(by='Mutual_Information', ascending=False)

    # Print the DataFrame for reference
    print(mi_df)
    
    return mi_df

if __name__ == '__main__':
    # Load the data
    train_data, test_data = load_data()

    # Preprocess the data
    train_data_pre, test_data_pre, y_encode = preprocess(train_data, test_data)

    # Calculate mutual information scores for the features
    mi_info = mutual_information(train_data_pre, y_encode)

    # Identify features with mutual information less than the threshold
    unselected_feature = mi_info[mi_info['Mutual_Information'] <0.11]['Features']
    print(unselected_feature)

    # Drop features with mutual information below the threshold from the training and testing datasets
    train_data_pre = train_data_pre.drop(unselected_feature, axis=1)
    test_data_pre = test_data_pre.drop(unselected_feature, axis=1)
    
    # Print remaining columns and their count
    #print(train_data_pre.columns)
    #print(len(train_data_pre))
