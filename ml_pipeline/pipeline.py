from data_preprocessing import load_data,split_data,preprocess 
from model_training import training 
from model_evaluation import evaluation 
from feature_selection import mutual_information
from data_visualisation import visualise_classification_report,visualise_confusion_matrix,roc_auc_curve

def run_pipeline():
    #Data Collection/Extraction
    train_data,test_data=load_data()

    #Data Preprocessing 
    train_data_pre,test_data_pre,y_encode=preprocess(train_data,test_data)

    #Apply Feature Selection like Mutual Information
    mi_info=mutual_information(train_data_pre,y_encode)

    #Remove irrelevant columns
    unselected_features=mi_info[mi_info['Mutual_Information']<0.11]['Features']
    print(unselected_features)

    #Drop those columns
    train_data_pre=train_data_pre.drop(unselected_features,axis=1)
    test_data_pre=test_data_pre.drop(unselected_features,axis=1)

    #Make Training and Test sets 
    X_train,X_test,y_train,y_test=split_data(train_data_pre,y_encode)
    print(X_train.columns)

    #Model training 
    training(X_train,y_train)

    #Model Evaluation
    report_df,cm=evaluation(X_test,y_test)

    #Data Visualisation 
    #Classification report 
    visualise_classification_report(report_df)
    
    #Confusion Matrix 
    visualise_confusion_matrix(cm)
    
    #ROC-AUC Curve
    roc_auc_curve(X_test,y_test)


if __name__=='__main__':
    run_pipeline()
    