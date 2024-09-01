from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from data_preprocessing import load_data,split_data,preprocess
from feature_selection import mutual_information
from dimensionality_reduction import find_optimal_components,principal_ca

def random_optimization(X_train, y_train, param_grid):
    # Initialize the RandomForestClassifier
    xgb_model = XGBClassifier()

    # Setup GridSearchCV
    random_search = RandomizedSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

    # Fit the model
    random_search.fit(X_train, y_train)

    # Get the best hyperparameters and their corresponding accuracy score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("Best Hyperparameters:", best_params)
    print("Best Accuracy Score:", best_score)


if __name__=='__main__':
    train_data,test_data=load_data()

    train_data_pre,test_data_pre,y_encode=preprocess(train_data,test_data)
    
    mi_info=mutual_information(train_data_pre,y_encode)
    
    unselected_features=mi_info[mi_info['Mutual_Information']<0.11]['Features']
    print(unselected_features)
    
    train_data_pre=train_data_pre.drop(unselected_features,axis=1)
    test_data_pre=test_data_pre.drop(unselected_features,axis=1)

    X_train,X_test,y_train,y_test=split_data(train_data_pre,y_encode)

    #Apply dimensionality reduction
    X_train_pca,X_test_pca=principal_ca(X_train,X_test)

    # Parameter grid for RandomForestClassifier
    param_grid = {
    'n_estimators': np.arange(100, 1001, 100),
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'max_depth': np.arange(3, 11, 1),
    'min_child_weight': np.arange(1, 11, 1),
    'subsample': np.linspace(0.5, 1, 6),
    'colsample_bytree': np.linspace(0.5, 1, 6),
    'gamma': np.linspace(0, 5, 10),
    'reg_alpha': np.linspace(0, 1, 10),
    'reg_lambda': np.linspace(0, 1, 10)
    }

    # Call the grid optimization function
    random_optimization(X_train_pca, y_train, param_grid)