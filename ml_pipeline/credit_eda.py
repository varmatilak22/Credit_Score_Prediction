import pandas as pd
from ydata_profiling import ProfileReport
from data_preprocessing import load_data

def data_profile_report(train_data):
    # Select the first 50,000 samples
    data_subset = train_data[:5000]

    # Generate the profiling report
    profile = ProfileReport(
        data_subset,
        title="Univariate Analysis",
        minimal=True,
        explorative=True
    )
    
    # Save the report to an HTML file 
    profile.to_file("report.html")

def profile_preprocess(train_data, test_data):
    """
    Preprocesses the training and testing data by cleaning and transforming columns.
    
    Parameters:
    - train_data (DataFrame): The training data.
    - test_data (DataFrame): The testing data.
    
    Returns:
    - DataFrame, DataFrame: The preprocessed training and testing data.
    """
    # Convert raw input data into pandas DataFrames
    train = pd.DataFrame(train_data)
    test = pd.DataFrame(test_data)

    main_col=['Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate',
       'Delay_from_due_date', 'Num_Credit_Inquiries', 'Credit_Mix',
       'Outstanding_Debt', 'Payment_of_Min_Amount', 'Total_EMI_per_month']
    
    # Drop features with mutual information below the threshold from the training and testing datasets
    train_data= train_data[main_col]
    test_data= test_data[main_col]

    
    # Clean 'Annual_Income' by removing underscores and converting to float
    train_data['Annual_Income'] = train_data['Annual_Income'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    test_data['Annual_Income'] = test_data['Annual_Income'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    train_data['Annual_Income'] = train_data['Annual_Income'].astype(float)
    test_data['Annual_Income'] = test_data['Annual_Income'].astype(float)
    
    # Drop rows with null values from both training and testing datasets
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    # Clean 'Outstanding_Debt' by removing underscores
    train_data['Outstanding_Debt'] = train_data['Outstanding_Debt'].apply(lambda x: x.replace('_', "") if "_" in x else x)
    test_data['Outstanding_Debt'] = test_data['Outstanding_Debt'].apply(lambda x: x.replace('_', "") if "_" in x else x)
    train_data['Outstanding_Debt'] = train_data['Outstanding_Debt'].astype(float)
    test_data['Outstanding_Debt'] = test_data['Outstanding_Debt'].astype(float)

    
    return train_data,test_data

if __name__=='__main__':
    train_data,test_data=load_data()
    train_data_pre,test_data_pre=profile_preprocess(train_data,test_data)

    data_profile_report(train_data_pre)

