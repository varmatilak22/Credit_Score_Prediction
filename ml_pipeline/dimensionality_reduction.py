from data_preprocessing import load_data,split_data,preprocess
from feature_selection import mutual_information
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import joblib

def find_optimal_components(X_train,X_test):

    #Standardization
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    joblib.dump(scaler,'../model/scaler.pkl')

    #Apply dimensionality reduction
    pca=PCA()

    pca.fit(X_train_scaled)

    #Calculate explained variance ratios and cumulative variance
    explained_variance_ratio=pca.explained_variance_ratio_
    cumulative_explained_variance=explained_variance_ratio.cumsum()

    #Visualise the to find the optimal number of components 
    variance_threshold=0.95

    optimal_n_components=next(i for i,variance in enumerate(cumulative_explained_variance,start=1) if variance>=variance_threshold)

    #Plot cumulative explained variance 
    plt.figure(figsize=(10,10))
    plt.plot(cumulative_explained_variance,marker='o',label='Cumulative Explained Variance')
    plt.axvline(x=optimal_n_components-1,color='r',linestyle='--',label='Optimal Number of PC')
    plt.xlabel('Number of Principal Components')
    plt.ylabel("Cumulative Explained Variance")
    plt.title('Cumulative Explained Variance Vs. Number of Principal Components')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_n_components,X_train_scaled,X_test_scaled

def principal_ca(X_train,X_test):

    optimal_n_components,X_train_scaled,X_test_scaled=find_optimal_components(X_train,X_test)
    
    pca=PCA(n_components=optimal_n_components)
    X_train_pca=pca.fit_transform(X_train_scaled)
    X_test_pca=pca.transform(X_test_scaled)
    return X_train_pca,X_test_pca



if __name__=='__main__':
    train_data,test_data=load_data()
    
    train_data_pre,test_data_pre,y_encode=preprocess(train_data,test_data)

    mi_info=mutual_information(train_data_pre,y_encode)
    
    unselected_features=mi_info[mi_info['Mutual_Information']<0.11]['Features']

    train_data_pre=train_data_pre.drop(unselected_features,axis=1)
    test_data_pre=test_data_pre.drop(unselected_features,axis=1)

    X_train,X_test,y_train,y_test=split_data(train_data_pre,y_encode)

    #Find the optimal number of PCS
    X_train_pca,X_test_pca=principal_ca(X_train,X_test)
    print(X_train.shape,X_train_pca.shape)




