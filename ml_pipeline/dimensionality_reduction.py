# Import necessary modules and functions
from data_preprocessing import load_data, split_data, preprocess
from feature_selection import mutual_information
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

def find_optimal_components(X_train, X_test):
    """
    Standardize the data, perform PCA, and find the optimal number of principal components 
    that explain at least 95% of the variance.
    
    Parameters:
    X_train (pd.DataFrame or np.ndarray): The training feature data.
    X_test (pd.DataFrame or np.ndarray): The testing feature data.
    
    Returns:
    tuple: A tuple containing:
        - optimal_n_components (int): The number of principal components that explain at least 95% of the variance.
        - X_train_scaled (np.ndarray): Standardized training data.
        - X_test_scaled (np.ndarray): Standardized testing data.
    """

    # Standardize the feature data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler model for future use
    joblib.dump(scaler, '../model/scaler.pkl')

    # Initialize PCA
    pca = PCA()

    # Fit PCA on the standardized training data
    pca.fit(X_train_scaled)

    # Calculate explained variance ratios and cumulative variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    # Determine the optimal number of components to explain at least 95% variance
    variance_threshold = 0.95
    optimal_n_components = next(i for i, variance in enumerate(cumulative_explained_variance, start=1) if variance >= variance_threshold)

    # Plot cumulative explained variance
    plt.figure(figsize=(10, 10))
    plt.plot(cumulative_explained_variance, marker='o', label='Cumulative Explained Variance')
    plt.axvline(x=optimal_n_components - 1, color='r', linestyle='--', label='Optimal Number of PC')
    plt.xlabel('Number of Principal Components')
    plt.ylabel("Cumulative Explained Variance")
    plt.title('Cumulative Explained Variance Vs. Number of Principal Components')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_n_components, X_train_scaled, X_test_scaled

def principal_ca(X_train, X_test):
    """
    Perform PCA on the training and testing data with the optimal number of principal components.
    
    Parameters:
    X_train (pd.DataFrame or np.ndarray): The training feature data.
    X_test (pd.DataFrame or np.ndarray): The testing feature data.
    
    Returns:
    tuple: A tuple containing:
        - X_train_pca (np.ndarray): The training data transformed by PCA.
        - X_test_pca (np.ndarray): The testing data transformed by PCA.
    """

    # Find the optimal number of principal components
    optimal_n_components, X_train_scaled, X_test_scaled = find_optimal_components(X_train, X_test)
    
    # Initialize PCA with the optimal number of components
    pca = PCA(n_components=optimal_n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return X_train_pca, X_test_pca

if __name__ == '__main__':
    # Load the data
    train_data, test_data = load_data()
    
    # Preprocess the data
    train_data_pre, test_data_pre, y_encode = preprocess(train_data, test_data)

    # Perform feature selection using mutual information
    mi_info = mutual_information(train_data_pre, y_encode)
    
    # Identify features with mutual information less than the threshold
    unselected_features = mi_info[mi_info['Mutual_Information'] < 0.11]['Features']
    print(unselected_features)

    # Drop the unselected features from the datasets
    train_data_pre = train_data_pre.drop(unselected_features, axis=1)
    test_data_pre = test_data_pre.drop(unselected_features, axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(train_data_pre, y_encode)

    # Find the optimal number of principal components and apply PCA
    X_train_pca, X_test_pca = principal_ca(X_train, X_test)
    print(X_train.shape, X_train_pca.shape)
