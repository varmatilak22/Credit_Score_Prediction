from data_preprocessing import split_data,load_data,preprocess
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import classification_report
from feature_selection import mutual_information
from dimensionality_reduction import find_optimal_components,principal_ca
import os

#print()

def training(X_train,y_train):
    
    model=XGBClassifier(n_estimators=500,
    max_depth=7,
    learning_rate=0.3,
    subsample=1.0,
    colsample_bytree=0.9,
    min_child_weight=1,
    gemma=0,
    reg_alpha=0,
    reg_lambda=1)

    model.fit(X_train,y_train)

    joblib.dump(model,os.path.join(os.path.dirname(__file__), 'model', 'xgboost.pkl'))
    
    print("!!!Model Saved Successfully!!!")


if __name__=='__main__':
    train_data,test_data=load_data()

    train_data_pre,test_data_pre,y_encode=preprocess(train_data,test_data)

    mi_info=mutual_information(train_data_pre,y_encode)

    unselected_features=mi_info[mi_info['Mutual_Information']<0.11]['Features']
    print(unselected_features)

    train_data_pre=train_data_pre.drop(unselected_features,axis=1)
    test_data_pre=test_data_pre.drop(unselected_features,axis=1)


    X_train,X_test,y_train,y_test=split_data(train_data_pre,y_encode)
    
    #Train the model
    training(X_train,y_train)

    #Load the model
    model=joblib.load(os.path.join(os.path.dirname(__file__), 'model', 'xgboost.pkl'))

    y_pred=model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test,y_pred))
    print(X_train.shape,X_test.shape)
