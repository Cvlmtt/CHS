import pandas as pd
import DBTool as dbc


ds = pd.read_csv("~/bin/SpamDataset2/emails.csv")
columns = list()

print("Rows read: {}".format(len(ds.index)))
print("Columns read: {}".format(len(ds.columns)))

print("Dropping id column...")
ds = ds.drop(columns=['Email No.'])

print("Creating list with the lables...")
for col in ds.columns:
    columns.append(col)


print('\n\n-----------------------\n\n')

print("Actual rows {}".format(len(ds.index)))
print("Actual columns {}".format(len(ds.columns)))

connection = dbc.create_db_connection("localhost", "root", "Jgk#nfXGt#LTRcN3vRju", "spamDataFrame")

checkTable = "drop table if exist dataset"
dbc.execute_query(connection, checkTable)

createTableQuery = "create table dataset(\n"
for col in columns:
    createTableQuery = createTableQuery+col+"_KW varchar(255),\n"


createTableQuery = createTableQuery[:-2]+");"
print(createTableQuery)
"""f = open("query.txt", "w")
f.write(createTableQuery)
f.close"""


dbc.execute_query(connection, createTableQuery)