
import pandas as pd


def find_column(df, candidates):
    """Find a column using case-insensitive matching."""
    lookup = {str(column).lower().strip(): column for column in df.columns}

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    return None


def calculate_metrics(dataframes):
    """Calculate business KPIs from uploaded CSV files."""

    if not dataframes:
        return {}

    combined = pd.concat(dataframes.values(), ignore_index=True)

    revenue_col = find_column(
        combined,
        ["revenue", "sales", "total_revenue", "amount"]
    )

    units_col = find_column(
        combined,
        ["units_sold", "quantity", "units", "qty"]
    )

    product_col = find_column(
        combined,
        ["product", "product_name", "item"]
    )

    metrics = {
        "revenue": None,
        "units": None,
        "products": None,
        "top_product": None,
        "revenue_by_product": None
    }

    if revenue_col:
        revenue = pd.to_numeric(
            combined[revenue_col], errors="coerce"
        ).fillna(0)

        metrics["revenue"] = float(revenue.sum())

    if units_col:
        units = pd.to_numeric(
            combined[units_col], errors="coerce"
        ).fillna(0)

        metrics["units"] = float(units.sum())

    if product_col:
        metrics["products"] = int(combined[product_col].nunique())

        if units_col:
            product_sales = combined.copy()
            product_sales["_units"] = pd.to_numeric(
                product_sales[units_col], errors="coerce"
            ).fillna(0)

            grouped = (
                product_sales.groupby(product_col)["_units"]
                .sum()
                .sort_values(ascending=False)
            )

            if not grouped.empty:
                metrics["top_product"] = str(grouped.index[0])

        if revenue_col:
            product_revenue = combined.copy()
            product_revenue["_revenue"] = pd.to_numeric(
                product_revenue[revenue_col], errors="coerce"
            ).fillna(0)

            metrics["revenue_by_product"] = (
                product_revenue.groupby(product_col)["_revenue"]
                .sum()
                .sort_values(ascending=False)
            )

    return metrics


def answer_analytics_question(question, dataframes):
    """Answer supported numerical questions using Pandas."""

    if not dataframes:
        return None

    combined = pd.concat(dataframes.values(), ignore_index=True)
    question = question.lower()

    revenue_col = find_column(
        combined,
        ["revenue", "sales", "total_revenue", "amount"]
    )

    units_col = find_column(
        combined,
        ["units_sold", "quantity", "units", "qty"]
    )

    product_col = find_column(
        combined,
        ["product", "product_name", "item"]
    )

    if revenue_col and any(
        word in question
        for word in ["revenue", "sales amount", "total sales"]
    ):
        total = pd.to_numeric(
            combined[revenue_col], errors="coerce"
        ).fillna(0).sum()

        return f"Total revenue across the uploaded CSV data is {total:,.2f}."

    if units_col and any(
        phrase in question
        for phrase in ["units sold", "total units", "quantity sold"]
    ):
        total = pd.to_numeric(
            combined[units_col], errors="coerce"
        ).fillna(0).sum()

        return f"Total units sold across the uploaded CSV data is {total:,.0f}."

    if product_col and units_col and any(
        phrase in question
        for phrase in ["top product", "best selling product", "best-selling product"]
    ):
        temp = combined.copy()

        temp["_units"] = pd.to_numeric(
            temp[units_col], errors="coerce"
        ).fillna(0)

        grouped = temp.groupby(product_col)["_units"].sum()

        if not grouped.empty:
            product = grouped.idxmax()
            units = grouped.max()

            return (
                f"The best-selling product is {product}, "
                f"with {units:,.0f} units sold."
            )

    return None
