
import sqlite3
from pathlib import Path

import pandas as pd


class MultiSourceLoader:
    """Load content from CSV, TXT, Markdown and SQLite sources."""

    def load_file(self, file_path):
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == ".csv":
            return self.load_csv(path)

        if extension == ".txt":
            return self.load_text(path)

        if extension in [".md", ".markdown"]:
            return self.load_text(path)

        raise ValueError(f"Unsupported file format: {extension}")

    def load_csv(self, file_path):
        df = pd.read_csv(file_path)

        documents = []

        for index, row in df.iterrows():
            content = "\n".join(
                f"{column}: {value}"
                for column, value in row.items()
            )

            documents.append({
                "content": content,
                "source": str(file_path),
                "source_type": "csv",
                "row": int(index) + 1
            })

        return documents

    def load_text(self, file_path):
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        return [{
            "content": content,
            "source": str(path),
            "source_type": path.suffix.lower().lstrip(".")
        }]

    def load_sqlite(self, database_path, table_name):
        connection = sqlite3.connect(database_path)

        try:
            df = pd.read_sql_query(
                f'SELECT * FROM "{table_name}"',
                connection
            )

            documents = []

            for index, row in df.iterrows():
                content = "\n".join(
                    f"{column}: {value}"
                    for column, value in row.items()
                )

                documents.append({
                    "content": content,
                    "source": f"{database_path}:{table_name}",
                    "source_type": "sqlite",
                    "row": int(index) + 1
                })

            return documents

        finally:
            connection.close()

    def list_sqlite_tables(self, database_path):
        connection = sqlite3.connect(database_path)

        try:
            cursor = connection.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' ORDER BY name"
            )

            return [row[0] for row in cursor.fetchall()]

        finally:
            connection.close()
