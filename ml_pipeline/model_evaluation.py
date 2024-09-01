from data_preprocessing import split_data,preprocess,load_data
import joblib 
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc,ConfusionMatrixDisplay,RocCurveDisplay
from sklearn.preprocessing import label_binarize
import seaborn as sns 
import matplotlib.pyplot as plt
from feature_selection import mutual_information
import pandas as pd
import os 
def evaluation(X_test,y_test):
    #Load the model
    model=joblib.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'xgboost.pkl'))

    #Make a predictions 
    y_pred=model.predict(X_test)

    #Generate Classification Report 
    report=classification_report(y_test,y_pred,output_dict=True)
    report_df=pd.DataFrame(report).transpose() 

    print('CLassification Report - Credit Score Classes')
    print(report_df)

    #Confusion matrix 
    cm=confusion_matrix(y_test,y_pred)
    print('Confusion Matrix')
    print(cm)

    return report_df,cm

if __name__=='__main__':
    train_data,test_data=load_data()

    train_data_pre,test_data_pre,y_encode=preprocess(train_data,test_data)

    mi_info=mutual_information(train_data_pre,y_encode)

    unselected_features=mi_info[mi_info['Mutual_Information']<0.11]['Features']
    print(unselected_features)

    train_data_pre=train_data_pre.drop(unselected_features,axis=1)
    test_data_pre=test_data_pre.drop(unselected_features,axis=1)


    X_train,X_test,y_train,y_test=split_data(train_data_pre,y_encode)

    evaluation(X_test,y_test)
    print(X_test.shape,X_train.shape)

    
