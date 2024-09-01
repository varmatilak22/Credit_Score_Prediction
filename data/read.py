import sqlite3

conn=sqlite3.connect('sample.db')

cursor=conn.cursor()

cursor.execute(
    'select * from train_data;'
)

data=cursor.fetchall()

for i in data[:5]:
    print(i)

cursor.execute(
    'select * from test_data;'
)

data=cursor.fetchall()

for i in data[:5]:
    print(i)


conn.close()