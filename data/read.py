# Import the SQLite library to interact with SQLite databases
import sqlite3

# Establish a connection to the SQLite database 'sample.db'
# If 'sample.db' does not exist, it will be created
conn = sqlite3.connect('sample.db')

# Create a cursor object using the connection
# The cursor is used to execute SQL commands
cursor = conn.cursor()

# Execute an SQL query to select all records from the 'train_data' table
cursor.execute(
    'SELECT * FROM train_data;'
)

# Fetch all the rows returned by the above SQL query
# The result is stored in the variable 'data'
data = cursor.fetchall()

# Iterate over the first 5 rows in 'data' and print each row
for i in data[:5]:
    print(i)

# Execute an SQL query to select all records from the 'test_data' table
cursor.execute(
    'SELECT * FROM test_data;'
)

# Fetch all the rows returned by the above SQL query
# The result is stored in the variable 'data'
data = cursor.fetchall()

# Iterate over the first 5 rows in 'data' and print each row
for i in data[:5]:
    print(i)

# Close the database connection to free up resources
conn.close()
