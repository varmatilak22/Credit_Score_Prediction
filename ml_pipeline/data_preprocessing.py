# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import joblib  # For saving and loading Python objects
from sklearn.preprocessing import LabelEncoder  # For encoding categorical labels into numerical values
from imblearn.over_sampling import SMOTE  # For handling class imbalance in datasets
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
import re  # For regular expression operations
import sqlite3  # For SQLite database interaction
from sklearn.preprocessing import MinMaxScaler  # For scaling features to a range
import os  # For interacting with the operating system, including file paths


def load_data():
    # Establish a connection to the SQLite database 'sample.db' located in the 'data' directory
    # The database file path is constructed dynamically based on the current script's directory
    connection = sqlite3.connect(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sample.db'))
    
    # Use pandas to execute SQL queries and read data from the 'train_data' table into a DataFrame
    train_data = pd.read_sql_query('SELECT * FROM train_data;', connection)
    
    # Use pandas to execute SQL queries and read data from the 'test_data' table into a DataFrame
    test_data = pd.read_sql_query('SELECT * FROM test_data;', connection)

    # Close the database connection after reading the data
    connection.close()

    # Return the DataFrames containing the training and testing data
    return train_data, test_data


def cleaned_loan_types(loan_types):
    # Remove the word 'and' from each loan type in the list and strip any leading/trailing whitespace
    cleaned_loan_types = [loan.replace('and ', '').strip() for loan in loan_types]
    
    # Convert the cleaned list of loan types to a set to get unique values and return it
    return set(cleaned_loan_types)
def extract_unique_loans_types(descriptions):
    """
    Extracts unique loan types from a list of descriptions.
    
    Parameters:
    - descriptions (list of str): A list where each element is a string describing loan types.
    
    Returns:
    - set: A set of unique loan types cleaned of 'and' and any additional unwanted text.
    """
    # Create a set to store unique loan types
    unique_loans = set()

    # Iterate over each description in the list
    for desc in descriptions:
        if pd.isna(desc) or 'Not Specified' in desc:
            # Handle missing values or descriptions that say 'Not Specified'
            unique_loans.add('Not Specified')
        else:
            # Split the description by commas to separate individual loan types
            loan_types = [loan.strip() for loan in desc.split(',')]
            # Add the cleaned loan types to the set of unique loans
            unique_loans.update(loan_types)
    
    # Clean the unique loan types by removing unnecessary words or characters
    cleaned_loan = cleaned_loan_types(list(unique_loans))
    
    # Return the cleaned set of unique loan types
    return cleaned_loan


def preprocess(train_data, test_data):
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
    train_data['Change_Credit_Limit'] = train_data['Changed_Credit_Limit'].astype(float)
    test_data['Change_Credit_Limit'] = test_data['Changed_Credit_Limit'].astype(float)

    # Update column lists
    categorical_cols.remove('Changed_Credit_Limit')
    numerical_cols.append("Change_Credit_Limit")

    # Clean 'Outstanding_Debt' by removing underscores
    train_data['Outstanding_Debt'] = train_data['Outstanding_Debt'].apply(lambda x: x.replace('_', "") if "_" in x else x)
    test_data['Outstanding_Debt'] = test_data['Outstanding_Debt'].apply(lambda x: x.replace('_', "") if "_" in x else x)

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

    # Drop 'Changed_Credit_Limit' column
    train_data = train_data.drop('Changed_Credit_Limit', axis=1)
    test_data = test_data.drop('Changed_Credit_Limit', axis=1)

    # Extract target variable 'Credit_Score' and drop it from training data
    y = train_data['Credit_Score']
    train_data = train_data.drop(['Credit_Score'], axis=1)
    categorical_cols.remove('Credit_Score')

    # Map categorical 'Credit_Score' to numerical values
    label_to_num = {'Good': 2, 'Standard': 1, 'Poor': 0}
    y_encode = y.map(label_to_num)

    # Define and apply month mapping
    month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                  'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    train_data['Month'] = train_data['Month'].map(month_mapping)
    test_data['Month'] = test_data['Month'].map(month_mapping)

    # Replace '_______' in 'Occupation' with 'None' and map professions to numerical values
    train_data['Occupation'] = train_data['Occupation'].apply(lambda x: x.replace("_______", "None") if x == '_______' else x)
    profession_mapping = {'None': 1, 'Lawyer': 2, 'Mechanic': 3, 'Docter': 4, 'Architect': 5, 'Entreprenuer': 6,
                      'Teacher': 7, 'Media_Manager': 8, 'Scientist': 9, 'Engineer': 10, 'Accountant': 11, 'Developer': 12, 
                      'Writer': 13, 'Journalist': 14, 'Manager': 15, 'Musician': 16}
    train_data['Occupation'] = train_data['Occupation'].map(profession_mapping)
    test_data['Occupation'] = test_data['Occupation'].map(profession_mapping)

    # Extract unique loan types and create a mapping dictionary
    loan_types = train_data['Type_of_Loan'].unique()
    unique_loan_types = extract_unique_loans_types(loan_types)
    loan_type_mapping = {loan_type: index for index, loan_type in enumerate(unique_loan_types)}
    
    # Define a function to transform loan descriptions into integer lists based on a mapping
    def transform_row_to_integer(desc, loan_mapping):
        # If description is NaN or contains "Not Specified", return the mapped value for 'Not Specified'
        if pd.isna(desc) or "Not Specified" in desc:
            return list(set([loan_mapping['Not Specified']]))
        else:
            # Split the description into loan types and map each type to its integer value
            loan_types = [loan.strip() for loan in desc.split(',')]
            return list(set([loan_mapping[loan_type] for loan_type in loan_types if loan_type in loan_mapping]))

    # Apply the transformation function to the 'Type_of_Loan' column in both training and testing datasets
    train_data['Type_of_Loan'] = train_data['Type_of_Loan'].apply(lambda x: transform_row_to_integer(x, loan_type_mapping))
    test_data['Type_of_Loan'] = test_data['Type_of_Loan'].apply(lambda x: transform_row_to_integer(x, loan_type_mapping))

    # Update categorical and numerical columns lists based on the 'Type_of_Loan', 'Month', and 'Occupation' columns
    categorical_cols = list(set(categorical_cols) - set(['Type_of_Loan', 'Month', 'Occupation']))
    numerical_cols.extend(['Type_of_Loan', 'Month', 'Occupation'])

    # Define mapping for 'Credit_Mix' values and apply the mapping to both datasets
    mix_mapping = {'Standard': 1, 'Good': 2, 'Bad': 0, '_': 'None'}
    train_data['Credit_Mix'] = train_data['Credit_Mix'].map(mix_mapping)
    test_data['Credit_Mix'] = test_data['Credit_Mix'].map(mix_mapping)

    # Update column lists
    categorical_cols.remove('Credit_Mix')
    numerical_cols.append('Credit_Mix')

    # Define and apply mapping for 'Payment_Behaviour' column
    payment_behaviour_mapping = {'Low_spent_Small_value_payments': 1,
                             'High_spent_Medium_value_payments': 2,
                             'Low_spent_Medium_value_payments': 3,
                             'High_spent_Large_value_payments': 4,
                             'High_spent_Small_value_payments': 5,
                             'Low_spent_Large_value_payments': 6,
                             '!@9#%8': 0}
    train_data['Payment_Behaviour'] = train_data['Payment_Behaviour'].map(payment_behaviour_mapping)
    test_data['Payment_Behaviour'] = test_data['Payment_Behaviour'].map(payment_behaviour_mapping)

    # Update column lists
    categorical_cols.remove('Payment_Behaviour')
    numerical_cols.append('Payment_Behaviour')

    # Define and apply mapping for 'Payment_of_Min_Amount' column
    payment_mapping = {'Yes': 1, 'No': 0, 'NM': -1}
    train_data['Payment_of_Min_Amount'] = train_data['Payment_of_Min_Amount'].map(payment_mapping)
    test_data['Payment_of_Min_Amount'] = test_data['Payment_of_Min_Amount'].map(payment_mapping)

    # Update column lists
    categorical_cols.remove('Payment_of_Min_Amount')
    numerical_cols.append('Payment_of_Min_Amount')

    # Define a function to convert 'Credit_History_Age' from years and months to total months
    def convert_to_months(age_str):
        match = re.match(r"(\d+) Years and (\d+) Months", age_str)
        if match:
            years = int(match.group(1))
            months = int(match.group(2))
            total_months = years * 12 + months
            return total_months
        return None

    # Apply the conversion function to 'Credit_History_Age' column
    train_data['Credit_History_Age'] = train_data['Credit_History_Age'].map(convert_to_months)
    test_data['Credit_History_Age'] = test_data['Credit_History_Age'].map(convert_to_months)

    # Update column lists
    categorical_cols.remove('Credit_History_Age')
    numerical_cols.append('Credit_History_Age')

    # Data preprocessing for numerical columns, filling missing values
    train_data['Occupation'] = train_data['Occupation'].fillna(0)
    test_data['Occupation'] = test_data['Occupation'].fillna(0)
    train_data['Change_Credit_Limit'] = train_data['Change_Credit_Limit'].fillna(0)
    test_data['Change_Credit_Limit'] = test_data['Change_Credit_Limit'].fillna(0)

    # Replace 'None' values in 'Credit_Mix' with -1
    train_data['Credit_Mix'] = train_data['Credit_Mix'].replace('None', -1)
    test_data['Credit_Mix'] = test_data['Credit_Mix'].replace('None', -1)

    # Drop 'Type_of_Loan' column from both datasets
    train_type_loan = train_data['Type_of_Loan']
    test_type_loan = test_data['Type_of_Loan']
    train_data = train_data.drop(['Type_of_Loan'], axis=1)
    test_data = test_data.drop(['Type_of_Loan'], axis=1)

    # Convert 'Outstanding_Debt' to float
    train_data['Outstanding_Debt'] = train_data['Outstanding_Debt'].astype(float)
    test_data['Outstanding_Debt'] = test_data['Outstanding_Debt'].astype(float)

    # Return processed datasets and target variable
    return train_data, test_data, y_encode

# Define a function to split data into training and testing sets and apply SMOTE for oversampling
def split_data(data, y):
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=1)
        
    # Apply SMOTE to the training data for oversampling
    smote = SMOTE(random_state=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, X_test, y_train_resampled, y_test

# Main execution block
if __name__ == '__main__':
    # Load and preprocess data
    train_data, test_data = load_data()
    train_data_pre, test_data_pre, y_encode = preprocess(train_data, test_data)
    
    # Split data and apply SMOTE
    X_train, X_test, y_train, y_test = split_data(train_data_pre, y_encode)
    print(X_train.shape, X_test.shape)
