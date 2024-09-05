import pandas as pd
from ydata_profiling import ProfileReport
from data_preprocessing import load_data

def data_profile_report(train_data):
    # Select the first 50,000 samples
    data_subset = train_data[:50000]

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

    # Columns to drop from both datasets
    drop_columns = ['ID', 'Customer_ID', 'Name', "SSN"]
    train_data = train_data.drop(drop_columns, axis=1)
    test_data = test_data.drop(drop_columns, axis=1)

    # Identify numerical and categorical columns
    numerical_cols = [col for col in train_data.columns if train_data[col].dtype != 'object']
    categorical_cols = [col for col in train_data.columns if train_data[col].dtype == 'object']

    # Clean 'Annual_Income' by removing underscores and converting to float
    train_data['Annual_Income'] = train_data['Annual_Income'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    test_data['Annual_Income'] = test_data['Annual_Income'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    train_data['Annual_Income'] = train_data['Annual_Income'].astype(float)
    test_data['Annual_Income'] = test_data['Annual_Income'].astype(float)
    
    # Clean 'Num_of_Loan' by removing underscores and converting to integer
    train_data['Num_of_Loan'] = train_data['Num_of_Loan'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    test_data['Num_of_Loan'] = test_data['Num_of_Loan'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    train_data['Num_of_Loan'] = train_data['Num_of_Loan'].astype(int)
    test_data['Num_of_Loan'] = test_data['Num_of_Loan'].astype(int)
    
    # Update column lists to reflect changes
    categorical_cols.remove('Num_of_Loan')
    categorical_cols.remove('Annual_Income')
    numerical_cols.append('Num_of_Loan')
    numerical_cols.append('Annual_Income')

    # Drop rows with null values from both training and testing datasets
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    # Clean 'Num_of_Delayed_Payment' by removing underscores and convert to integer
    train_data['Num_of_Delayed_Payment'] = train_data['Num_of_Delayed_Payment'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    test_data['Num_of_Delayed_Payment'] = test_data['Num_of_Delayed_Payment'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    train_data['Num_of_Delayed_Payment'] = train_data['Num_of_Delayed_Payment'].astype(int)
    test_data['Num_of_Delayed_Payment'] = test_data['Num_of_Delayed_Payment'].astype(int)

    # Update column lists
    categorical_cols.remove('Num_of_Delayed_Payment')
    numerical_cols.append("Num_of_Delayed_Payment")

    # Clean 'Changed_Credit_Limit' by removing underscores, filtering out empty values, and converting to float
    train_data['Changed_Credit_Limit'] = train_data['Changed_Credit_Limit'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    test_data['Changed_Credit_Limit'] = test_data['Changed_Credit_Limit'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    train_data['Changed_Credit_Limit'] = train_data['Changed_Credit_Limit'][train_data['Changed_Credit_Limit'] != '']
    test_data['Changed_Credit_Limit'] = test_data['Changed_Credit_Limit'][test_data['Changed_Credit_Limit'] != '']
    train_data['Changed_Credit_Limit'] = train_data['Changed_Credit_Limit'].astype(float)
    test_data['Changed_Credit_Limit'] = test_data['Changed_Credit_Limit'].astype(float)

    # Update column lists
    categorical_cols.remove('Changed_Credit_Limit')
    numerical_cols.append("Change_Credit_Limit")

    # Clean 'Outstanding_Debt' by removing underscores
    train_data['Outstanding_Debt'] = train_data['Outstanding_Debt'].apply(lambda x: x.replace('_', "") if "_" in x else x)
    test_data['Outstanding_Debt'] = test_data['Outstanding_Debt'].apply(lambda x: x.replace('_', "") if "_" in x else x)
    train_data['Outstanding_Debt'] = train_data['Outstanding_Debt'].astype(float)
    test_data['Outstanding_Debt'] = test_data['Outstanding_Debt'].astype(float)

    # Update column lists
    categorical_cols.remove('Outstanding_Debt')
    numerical_cols.append('Outstanding_Debt')

    # Clean 'Age' by removing underscores and convert to integer
    train_data['Age'] = train_data['Age'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    test_data['Age'] = test_data['Age'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    train_data['Age'] = train_data['Age'].astype(int)
    test_data['Age'] = test_data['Age'].astype(int)

    # Update column lists
    categorical_cols.remove('Age')
    numerical_cols.append('Age')

    # Convert 'Monthly_Balance' to string, clean by removing underscores, and convert to float
    train_data['Monthly_Balance'] = train_data['Monthly_Balance'].astype(str)
    test_data['Monthly_Balance'] = test_data['Monthly_Balance'].astype(str)
    train_data['Monthly_Balance'] = train_data['Monthly_Balance'].apply(lambda x: x.replace("_", "") if isinstance(x, str) else x)
    test_data['Monthly_Balance'] = test_data['Monthly_Balance'].apply(lambda x: x.replace("_", "") if isinstance(x, str) else x)
    train_data['Monthly_Balance'] = pd.to_numeric(train_data['Monthly_Balance'], errors='coerce')
    test_data['Monthly_Balance'] = pd.to_numeric(test_data['Monthly_Balance'], errors='coerce')

    # Round 'Monthly_Balance' values to 2 decimal places
    train_data['Monthly_Balance'] = train_data['Monthly_Balance'].round(2)
    test_data['Monthly_Balance'] = test_data['Monthly_Balance'].round(2)

    # Update column lists
    categorical_cols.remove('Monthly_Balance')
    numerical_cols.append('Monthly_Balance')

    # Clean 'Amount_invested_monthly' by removing underscores and convert to float
    train_data['Amount_invested_monthly'] = train_data['Amount_invested_monthly'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    test_data['Amount_invested_monthly'] = test_data['Amount_invested_monthly'].apply(lambda x: x.replace("_", "") if "_" in x else x)
    train_data['Amount_invested_monthly'] = train_data['Amount_invested_monthly'].round(3)
    test_data['Amount_invested_monthly'] = test_data['Amount_invested_monthly'].round(3)
    train_data['Amount_invested_monthly'] = train_data['Amount_invested_monthly'].astype(float)
    test_data['Amount_invested_monthly'] = test_data['Amount_invested_monthly'].astype(float)

    # Update column lists
    categorical_cols.remove('Amount_invested_monthly')
    numerical_cols.append('Amount_invested_monthly')

    #drop columns 


    return train_data,test_data

if __name__=='__main__':
    train_data,test_data=load_data()
    train_data_pre,test_data_pre=profile_preprocess(train_data,test_data)

    data_profile_report(train_data_pre)

