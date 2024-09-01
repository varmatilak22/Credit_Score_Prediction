from sklearn.feature_selection import mutual_info_classif
from data_preprocessing import split_data,preprocess, load_data
import pandas as pd

def mutual_information(data,y):

    mi=mutual_info_classif(data,y)

    mi_df=pd.DataFrame({'Features':data.columns,'Mutual_Information':mi})

    mi_df=mi_df.sort_values(by='Mutual_Information',ascending=False)

    print(mi_df)
    return mi_df

if __name__=='__main__':
    train_data,test_data=load_data()

    train_data_pre,test_data_pre,y_encode=preprocess(train_data,test_data)

    mi_info=mutual_information(train_data_pre,y_encode)

    unselected_feature=mi_info[mi_info['Mutual_Information']<0.11]['Features']
    print(len(unselected_feature))

    #Drop those columns from the data
    train_data_pre=train_data_pre.drop(unselected_feature,axis=1)
    test_data_pre=test_data_pre.drop(unselected_feature,axis=1)
    print(train_data_pre.columns)
    print(len(train_data_pre))