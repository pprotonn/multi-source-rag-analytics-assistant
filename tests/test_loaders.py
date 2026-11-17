

import sqlite3

import pytest

from src.loaders import MultiSourceLoader


@pytest.fixture
def loader():
    return MultiSourceLoader()


def test_load_csv(loader, tmp_path):
    file_path = tmp_path / "sales.csv"
    file_path.write_text(
        "product,revenue\nCloudSync,200000\nDataFlow,90000",
        encoding="utf-8"
    )

    documents = loader.load_file(file_path)

    assert len(documents) == 2
    assert documents[0]["source_type"] == "csv"
    assert "CloudSync" in documents[0]["content"]
    assert documents[0]["row"] == 1


def test_load_text(loader, tmp_path):
    file_path = tmp_path / "company.txt"
    file_path.write_text(
        "The company has 120 employees.",
        encoding="utf-8"
    )

    documents = loader.load_file(file_path)

    assert len(documents) == 1
    assert documents[0]["source_type"] == "txt"
    assert "120 employees" in documents[0]["content"]


def test_load_markdown(loader, tmp_path):
    file_path = tmp_path / "company.md"
    file_path.write_text(
        "# Company Overview\nThe company provides cloud services.",
        encoding="utf-8"
    )

    documents = loader.load_file(file_path)

    assert len(documents) == 1
    assert documents[0]["source_type"] == "md"
    assert "cloud services" in documents[0]["content"]


def test_load_sqlite(loader, tmp_path):
    database_path = tmp_path / "company.db"

    connection = sqlite3.connect(database_path)

    connection.execute(
        "CREATE TABLE employees (name TEXT, department TEXT)"
    )

    connection.execute(
        "INSERT INTO employees VALUES (?, ?)",
        ("Ravi", "Engineering")
    )

    connection.commit()
    connection.close()

    documents = loader.load_sqlite(
        database_path,
        "employees"
    )

    assert len(documents) == 1
    assert documents[0]["source_type"] == "sqlite"
    assert "Ravi" in documents[0]["content"]
    assert "Engineering" in documents[0]["content"]


def test_list_sqlite_tables(loader, tmp_path):
    database_path = tmp_path / "company.db"

    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE employees (name TEXT)")
    connection.execute("CREATE TABLE products (name TEXT)")
    connection.commit()
    connection.close()

    tables = loader.list_sqlite_tables(database_path)

    assert tables == ["employees", "products"]


def test_unsupported_file_format(loader, tmp_path):
    file_path = tmp_path / "document.pdf"
    file_path.write_text("Sample content", encoding="utf-8")

    with pytest.raises(ValueError):
        loader.load_file(file_path)
