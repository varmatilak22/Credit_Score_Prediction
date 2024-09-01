from data_preprocessing import split_data,preprocess,load_data
import joblib 
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc,ConfusionMatrixDisplay,RocCurveDisplay
from sklearn.preprocessing import label_binarize
import seaborn as sns 
import matplotlib.pyplot as plt
from feature_selection import mutual_information
import pandas as pd
from model_evaluation import evaluation 
import os


def visualise_classification_report(report_df):

    #Visualise the classification report 

    #Remove unwanted columns and rows 
    metrics=['precision','recall','f1-score']
    
    classes=report_df.index[:-3]
    
    report_df_filtered=report_df.loc[classes,metrics]

    plt.figure(figsize=(10,6))
    sns.heatmap(report_df_filtered,annot=True,cmap='Blues',fmt='.2f')
    plt.title("Classification Report \n Accuracy:{:.2f}%".format(report_df.loc['accuracy','precision']*100))
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'classification_report.png'))

def visualise_confusion_matrix(cm):
    #Compute and plot the confusion matrix 
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels={0:'Poor',1:'Standard',2:'Good'})
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix - Credit Classes')
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'confusion_matrix.png'))

def roc_auc_curve(X_test,y_test):

    #Load the model using joblin
    model=joblib.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'xgboost.pkl'))
    
    #prediction probabilities
    y_pred=model.predict_proba(X_test)

    # Binarize the output (needed for ROC curve)
    y_binarized = label_binarize(y_test, classes=[0,1,2])
    #print(y_binarized.shape[1])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_binarized.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red']  # Adjust the number of colors according to the number of classes
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Multi-Class')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'roc_auc_curve.png'))


if __name__=='__main__':
    train_data,test_data=load_data()

    train_data_pre,test_data_pre,y_encode=preprocess(train_data,test_data)

    mi_info=mutual_information(train_data_pre,y_encode)

    unselected_features=mi_info[mi_info['Mutual_Information']<0.11]['Features']
    print(unselected_features)

    train_data_pre=train_data_pre.drop(unselected_features,axis=1)
    test_data_pre=test_data_pre.drop(unselected_features,axis=1)


    X_train,X_test,y_train,y_test=split_data(train_data_pre,y_encode)

    report_df,cm=evaluation(X_test,y_test)

    visualise_classification_report(report_df)
    visualise_confusion_matrix(cm)
    roc_auc_curve(X_test,y_test)
