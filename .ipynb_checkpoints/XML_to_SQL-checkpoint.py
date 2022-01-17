"""
Library to read in an Apple Healthkit csv file, convert it to a pandas dataframe, and store it as a table in a SQLite database.

@author: Pankaj Meghani
@date: 2021/10/23
"""
import pandas as pd
import numpy as np
import datetime
import xml.etree.ElementTree as ET
import sqlite3
from sqlite3 import Error
import zipfile
import glob

np.random.seed(0)

def create_connection(path):
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection, connection.cursor()

def create_df(userID):
    tree = ET.parse(f'../datasets/apple_healthkit/export ({userID})/apple_health_export/export.xml') 
    root = tree.getroot()
    record_list = [x.attrib for x in root.iter('Record')]
    
    data = pd.DataFrame(record_list)
    for col in ['creationDate', 'startDate', 'endDate']:
        data[col] = pd.to_datetime(data[col])

    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    data['value'] = data['value'].fillna(1.0)
    data['type'] = data['type'].str.replace('HKQuantityTypeIdentifier', '')
    data['type'] = data['type'].str.replace('HKCategoryTypeIdentifier', '')
    return data

def write_dfs_to_db(dfs, userID, con):
    return pd.concat(dfs, keys=list(range(len(dfs))))
    
def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
if __name__ == "__main__":
    conn, cur = create_connection('./db.sqlite')
    print(glob.glob1(myPath,"*.zip"))
    
#     dfs = []
#     for i in range
#     df = create_df(0)
    
#     write_df_to_db(df, 0, conn)