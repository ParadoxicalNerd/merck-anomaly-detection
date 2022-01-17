import sqlite3

class SQL:
    def __init__(self, path: str) -> None:
        self.path = path

    def create_connection(self):
        connection = None
        try:
            connection = sqlite3.connect(self.path)
            print("Connection to SQLite DB successful")
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")
        return connection, connection.cursor()
