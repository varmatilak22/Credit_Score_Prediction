import sqlite3 
import pandas as pd

#Step 1: Read CSV file into dataframes
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

#Step 2: Connect to sqlite3 Database
conn=sqlite3.connect('sample.db')

#Step 3:Save Dataframe to sqlite dataframe
train_data.to_sql('train_data',conn,if_exists='replace',index=False)
test_data.to_sql('test_data',conn,if_exists='replace',index=False)

#Step 4: Close the Connection
conn.close()

print("Data Has been successfully saved to the SQLite Database.")
