"""
Library to read in an Apple Healthkit csv file, convert it to a pandas dataframe, and store it as a table in a SQLite database.

@author: Pankaj Meghani
@date: 2021/10/23
"""
from os import read
from sqlite3.dbapi2 import Connection, Cursor
import pandas as pd
import numpy as np
import datetime
import xml.etree.ElementTree as ET
import sqlite3
import zipfile
import glob

from src.SQL_Interface import SQL


class XML_to_SQL:
    def __init__(
        self, zip_path: str = "./Apple Healthkit data/", seed: int = 0
    ) -> None:
        self.base_path = zip_path
        np.random.seed(seed)

    def create_df(self, xml_data):
        # tree = ET.parse(ET.fromstring(xml_data))
        # root = tree.getroot()
        root = ET.fromstring(xml_data)
        record_list = [x.attrib for x in root.iter("Record")]

        data = pd.DataFrame(record_list)
        for col in ["creationDate", "startDate", "endDate"]:
            data[col] = pd.to_datetime(data[col])

        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data["value"] = data["value"].fillna(1.0)
        data["type"] = data["type"].str.replace("HKQuantityTypeIdentifier", "")
        data["type"] = data["type"].str.replace("HKCategoryTypeIdentifier", "")
        return data

    def concat_dfs(self, dfs, users):
        return pd.concat(dfs, keys=users)

    def read_zip(self, zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # zip_ref.extractall(extract_path)
            f = zip_ref.open(
                "apple_health_export/export.xml"
            )  # Allows us to work in-memory which is much faster
            xml_file = f.read()
            return xml_file

    def read_zips(
        self,
    ):
        zips = glob.glob1(self.base_path, "*.zip")
        zip_data = []
        user_ids = []
        for i in range(len(zips)):
            xml_data = self.read_zip(self.base_path + zips[i])
            zip_data.append(xml_data)
            user_ids.append(zips[i][len(zips[i]) - 6])
        return zip_data, user_ids

    def zips_to_db(self, path="./db.sqlite"):
        conn, cur = SQL(path).create_connection()
        zips, users = self.read_zips()

        dfs = []
        for zip in zips:
            dfs.append(self.create_df(zip))

        cdfs = (
            self.concat_dfs(dfs, users)
            .reset_index()
            .drop("level_1", axis=1)
            .rename(columns={"level_0": "user_id"})
        )

        cdfs.to_sql("healthkit_records", conn)

    def zips_to_db(self, conn: Connection, cur: Cursor):
        zips, users = self.read_zips()

        dfs = []
        for zip in zips:
            dfs.append(self.create_df(zip))

        cdfs = (
            self.concat_dfs(dfs, users)
            .reset_index()
            .drop("level_1", axis=1)
            .rename(columns={"level_0": "user_id"})
        )

        cdfs.to_sql("healthkit_records", conn)


if __name__ == "__main__":
    healthkit_reader = XML_to_SQL()
    healthkit_reader.zips_to_db()
