import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import re
import sqlite3
from sklearn.preprocessing import MinMaxScaler
import os 

def load_data():
    connection=sqlite3.connect(os.path.join(os.path.join(os.path.dirname(os.getcwd()),'data'),'sample.db'))
    
    #Using pandas to read SQL query, which includes column names 
    train_data=pd.read_sql_query('Select * from train_data;',connection)
    test_data=pd.read_sql_query('select * from test_data;',connection)

    #Commit the connection after reading the data
    connection.close()

    return train_data,test_data
def cleaned_loan_types(loan_types):

    #Remove 'and' from the loan types 
    cleaned_loan_types=[loan.replace('and ','').strip() for loan in loan_types]
    return set(cleaned_loan_types)

def extract_unique_loans_types(descriptions):
    #Create a set of elements
    unique_loans=set()

    for desc in descriptions:
        if pd.isna(desc) or 'Not Specified' in desc:
            #Handle nan and not specified values
            unique_loans.add('Not Specified')
        else:
            #Split description into indviduals loan types and clean up 
            loan_types=[loan.strip() for loan in desc.split(',')]
            unique_loans.update(loan_types)
    cleaned_loan=cleaned_loan_types(list(unique_loans))
    return cleaned_loan

def preprocess(train_data,test_data):
    train=pd.DataFrame(train_data)
    test=pd.DataFrame(test_data)

    #print(train['Monthly_Inhand_Salary'])
    #print(test.shape)

    drop_columns=['ID','Customer_ID','Name',"SSN"]
    train_data=train_data.drop(drop_columns,axis=1)
    test_data=test_data.drop(drop_columns,axis=1)

    #print(train_data.shape)
    #print(test_data.shape)

    numerical_cols=[col for col in train_data.columns if train_data[col].dtype!='object']
    categorical_cols=[col for col in train_data.columns if train_data[col].dtype=='object']
    #print(categorical_cols)
    #print(numerical_cols)
    #print(set(train_data.columns)-set(test_data.columns))

    #Underscore in numerical data
    #print(train_data['Annual_Income'].value_counts())
    train_data['Annual_Income']=train_data['Annual_Income'].apply(lambda x:x.replace("_","") if "_" in x else x)
    test_data['Annual_Income']=test_data['Annual_Income'].apply(lambda x:x.replace("_","") if "_" in x else x)
    
    #Convert the object into float64
    train_data['Annual_Income']=train_data['Annual_Income'].astype(float)
    test_data['Annual_Income']=test_data['Annual_Income'].astype(float)
    
    #We have convert those columns into numeircal which has numerical columns but idenitfy as categorical columns
    train_data['Num_of_Loan']=train_data['Num_of_Loan'].apply(lambda x:x.replace("_","") if "_" in x else x)
    test_data['Num_of_Loan']=test_data['Num_of_Loan'].apply(lambda x:x.replace("_","") if "_" in x else x)

    train_data['Num_of_Loan']=train_data['Num_of_Loan'].astype(int)
    test_data['Num_of_Loan']=test_data['Num_of_Loan'].astype(int)
    #print(train_data['Num_of_Loan'].dtype)

    #Remove numerical cols from categorical cols 
    categorical_cols.remove('Num_of_Loan')
    categorical_cols.remove('Annual_Income')
    #Adding to numrical cols
    numerical_cols.append('Num_of_Loan')
    numerical_cols.append('Annual_Income')

    #print(categorical_cols)

    #Drop the null values
    train_data=train_data.dropna()
    test_data=test_data.dropna()
    
    #print(train_data.shape)
    #print(test_data.shape)

    train_data['Num_of_Delayed_Payment']=train_data['Num_of_Delayed_Payment'].apply(lambda x:x.replace("_","") if "_" in x else x)
    test_data['Num_of_Delayed_Payment']=test_data['Num_of_Delayed_Payment'].apply(lambda x:x.replace("_","") if "_" in x else x)
    
    train_data['Num_of_Delayed_Payment']=train_data['Num_of_Delayed_Payment'].astype(int)
    test_data['Num_of_Delayed_Payment']=test_data['Num_of_Delayed_Payment'].astype(int)
    
    categorical_cols.remove('Num_of_Delayed_Payment')
    numerical_cols.append("Num_of_Delayed_Payment")

    #print(train_data['Num_of_Delayed_Payment'].unique())

    train_data['Changed_Credit_Limit']=train_data['Changed_Credit_Limit'].apply(lambda x:x.replace("_","") if "_" in x else x)
    test_data['Changed_Credit_Limit']=test_data['Changed_Credit_Limit'].apply(lambda x:x.replace("_","") if "_" in x else x)
    
    train_data['Changed_Credit_Limit']=train_data['Changed_Credit_Limit'][train_data['Changed_Credit_Limit']!='']
    test_data['Changed_Credit_Limit']=test_data['Changed_Credit_Limit'][test_data['Changed_Credit_Limit']!='']
    
    train_data['Change_Credit_Limit']=train_data['Changed_Credit_Limit'].astype(float)
    test_data['Change_Credit_Limit']=test_data['Changed_Credit_Limit'].astype(float)

    categorical_cols.remove('Changed_Credit_Limit')
    numerical_cols.append("Change_Credit_Limit")

    train_data['Outstanding_Debt']=train_data['Outstanding_Debt'].apply(lambda x:x.replace('_',"") if "_" in x else x)
    test_data['Outstanding_Debt']=test_data['Outstanding_Debt'].apply(lambda x:x.replace('_',"") if "_" in x else x)
    
    categorical_cols.remove('Outstanding_Debt')
    numerical_cols.append('Outstanding_Debt')
    
    train_data['Age']=train_data['Age'].apply(lambda x:x.replace("_","" if "_" in x else x))
    test_data['Age']=test_data['Age'].apply(lambda x:x.replace("_","" if "_" in x else x))
    
    train_data['Age']=train_data['Age'].astype(int)
    test_data['Age']=test_data['Age'].astype(int)
    
    categorical_cols.remove('Age')
    numerical_cols.append('Age')
    
    
    train_data['Monthly_Balance'] = train_data['Monthly_Balance'].astype(str)
    test_data['Monthly_Balance'] = test_data['Monthly_Balance'].astype(str)

    # Replace "_" with empty string and then convert to numeric
    train_data['Monthly_Balance'] = train_data['Monthly_Balance'].apply(lambda x: x.replace("_", "") if isinstance(x, str) else x)
    test_data['Monthly_Balance'] = test_data['Monthly_Balance'].apply(lambda x: x.replace("_", "") if isinstance(x, str) else x)

    # Convert cleaned 'Monthly_Balance' to float, handling any errors gracefully
    train_data['Monthly_Balance'] = pd.to_numeric(train_data['Monthly_Balance'], errors='coerce')
    test_data['Monthly_Balance'] = pd.to_numeric(test_data['Monthly_Balance'], errors='coerce')

    # Round the 'Monthly_Balance' values to 2 decimal places
    train_data['Monthly_Balance'] = train_data['Monthly_Balance'].round(2)
    test_data['Monthly_Balance'] = test_data['Monthly_Balance'].round(2)

    
    
    categorical_cols.remove('Monthly_Balance')
    numerical_cols.append('Monthly_Balance')
    
    train_data['Amount_invested_monthly']=train_data['Amount_invested_monthly'].apply(lambda x:x.replace("_","" if "_" in x else x))
    test_data['Amount_invested_monthly']=test_data['Amount_invested_monthly'].apply(lambda x:x.replace("_","" if "_" in x else x))
    
    train_data['Amount_invested_monthly']=train_data['Amount_invested_monthly'].round(3)
    test_data['Amount_invested_monthly']=test_data['Amount_invested_monthly'].round(3)


    train_data['Amount_invested_monthly']=train_data['Amount_invested_monthly'].astype(float)
    test_data['Amount_invested_monthly']=test_data['Amount_invested_monthly'].astype(float)
    
    
    categorical_cols.remove('Amount_invested_monthly')
    numerical_cols.append('Amount_invested_monthly')
    
    train_data=train_data.drop('Changed_Credit_Limit',axis=1)
    test_data=test_data.drop('Changed_Credit_Limit',axis=1)
    
    #print(train_data[categorical_cols])
    
    #Predicing Variables
    y=train_data['Credit_Score']
    
    #Drop the credit score
    train_data=train_data.drop(['Credit_Score'],axis=1)
    categorical_cols.remove('Credit_Score')

    #Custom labeling 
    label_to_num={'Good':2,
    'Standard':1,
    'Poor':0}

    y_encode=y.map(label_to_num)
    
    #Define the month mapping
    month_mapping={'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
    'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}

    #Applying mapping to encode the months
    train_data['Month']=train_data['Month'].map(month_mapping)
    test_data['Month']=test_data['Month'].map(month_mapping)
    
    train_data['Occupation']=train_data['Occupation'].apply(lambda x:x.replace("_______","None") if x=='_______' else x)
    
    #Define the profession encoding
    profession_mapping={'None':1,'Lawyer':2,'Mechanic':3,'Docter':4,'Architect':5,'Entreprenuer':6,
    'Teacher':7,'Media_Manager':8,'Scientist':9,'Engineer':10,'Accountant':11,'Developer':12,'Writer':13,
    'Journalist':14,'Manager':15,'Musician':16}

    #Mapping to encode the profession mapping
    train_data['Occupation']=train_data['Occupation'].map(profession_mapping)
    test_data['Occupation']=test_data['Occupation'].map(profession_mapping)
    
    #Convert the type of loans into numerical values

    #Extract unique loan types
    loan_types=train_data['Type_of_Loan'].unique()
    unique_loan_types=extract_unique_loans_types(loan_types)
    
    #print(unique_loan_types)

    #Create a mapping dictionary
    loan_type_mapping={loan_type:index for index,loan_type in enumerate(unique_loan_types)}
    #print(loan_type_mapping)

    #Transform row into integers based on the mapping
    def transform_row_to_integer(desc,loan_mapping):
        if pd.isna(desc) or "Not Specified" in desc:
            return list(set([loan_mapping['Not Specified']]))
        else:
            loan_types=[loan.strip() for loan in desc.split(',')]
            return list(set([loan_mapping[loan_type] for loan_type in loan_types if loan_type in loan_mapping]))
    
    #Apply mapping to train and test data 
    train_data['Type_of_Loan']=train_data['Type_of_Loan'].apply(lambda x:transform_row_to_integer(x,loan_type_mapping))
    test_data['Type_of_Loan']=test_data['Type_of_Loan'].apply(lambda x:transform_row_to_integer(x,loan_type_mapping))
    
    #Change in cat and num cols
    categorical_cols=list(set(categorical_cols)-set(['Type_of_Loan','Month','Occupation']))
    numerical_cols.extend(['Type_of_Loan','Month','Occupation'])
    
    #Mapping of values of credit mix 
    mix_mapping={'Standard':1,'Good':2,'Bad':0,'_':'None'}

    #Applying map method for there values in mix-mapping
    train_data['Credit_Mix']=train_data['Credit_Mix'].map(mix_mapping)
    test_data['Credit_Mix']=test_data['Credit_Mix'].map(mix_mapping)
    
    categorical_cols.remove('Credit_Mix')
    numerical_cols.append('Credit_Mix')

    #Convert the Payment Behaviour into numerical cols
    #print(train_data['Payment_Behaviour'].value_counts())

    #mapping payment behaviour
    payment_behaviour_mapping={'Low_spent_Small_value_payments':1,
    'High_spent_Medium_value_payments':2,
    'Low_spent_Medium_value_payments':3,
    'High_spent_Large_value_payments':4,
    'High_spent_Small_value_payments':5,
    'Low_spent_Large_value_payments':6,
    '!@9#%8':0}

    #Applying the mapping on payment behaviour 
    train_data['Payment_Behaviour']=train_data['Payment_Behaviour'].map(payment_behaviour_mapping)
    test_data['Payment_Behaviour']=test_data['Payment_Behaviour'].map(payment_behaviour_mapping)
    
    categorical_cols.remove('Payment_Behaviour')
    numerical_cols.append('Payment_Behaviour')

    #print(train_data['Payment_of_Min_Amount'].value_counts())

    #mapping function
    payment_mapping={'Yes':1,'No':0,'NM':-1}

    train_data['Payment_of_Min_Amount']=train_data['Payment_of_Min_Amount'].map(payment_mapping)
    test_data['Payment_of_Min_Amount']=test_data['Payment_of_Min_Amount'].map(payment_mapping)
    
    categorical_cols.remove('Payment_of_Min_Amount')
    numerical_cols.append('Payment_of_Min_Amount')
    
    #Converting the year and months into month
    def convert_to_months(age_str):
        match=re.match(r"(\d+) Years and (\d+) Months",age_str)
        if match:
            years=int(match.group(1))
            months=int(match.group(2))
            total_months=years*12 +months
            return total_months
        return None
    
    train_data['Credit_History_Age']=train_data['Credit_History_Age'].map(convert_to_months)
    test_data['Credit_History_Age']=test_data['Credit_History_Age'].map(convert_to_months)
    
    categorical_cols.remove('Credit_History_Age')
    numerical_cols.append('Credit_History_Age')

    #Data preprocessing of numerical cols
    #print(numerical_cols)
    
    train_data['Occupation']=train_data['Occupation'].fillna(0)    
    test_data['Occupation']=test_data['Occupation'].fillna(0)

    train_data['Change_Credit_Limit']=train_data['Change_Credit_Limit'].fillna(0)    
    test_data['Change_Credit_Limit']=test_data['Change_Credit_Limit'].fillna(0)
    
    #print(train_data.isnull().sum())
    #print(y_encode.value_counts())

    
    train_data['Credit_Mix']=train_data['Credit_Mix'].replace('None',-1)
    test_data['Credit_Mix']=test_data['Credit_Mix'].replace('None',-1)
    
    #print(train_data['Type_of_Loan'].value_counts())

    
    train_type_loan=train_data['Type_of_Loan']
    test_type_loan=test_data['Type_of_Loan']

    train_data=train_data.drop(['Type_of_Loan'],axis=1)
    test_data=test_data.drop(['Type_of_Loan'],axis=1)
    
    train_data['Outstanding_Debt']=train_data['Outstanding_Debt'].astype(float)
    test_data['Outstanding_Debt']=test_data['Outstanding_Debt'].astype(float)
    
    
    #print(y_encode.value_counts())

    return train_data,test_data,y_encode

def split_data(data,y):
    X_train,X_test,y_train,y_test=train_test_split(data,y,test_size=0.2,random_state=1)
        
    #Apply the oversampling technique on the data
    smote=SMOTE(random_state=1)
    X_train_resampled,y_train_resampled=smote.fit_resample(X_train,y_train)
    #print(X_train.shape,y_train.shape)
    #print(X_train_resampled.shape,y_train_resampled.shape)
    return X_train_resampled,X_test,y_train_resampled,y_test


if __name__=='__main__':
    train_data,test_data=load_data()
    train_data_pre,test_data_pre,y_encode=preprocess(train_data,test_data)
    
    #Splitting the data 
    X_train,X_test,y_train,y_test=split_data(train_data_pre,y_encode)
    print(X_train.shape,X_test.shape)