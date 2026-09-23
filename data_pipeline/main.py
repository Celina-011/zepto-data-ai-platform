
import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
DB_PATH = Path(__file__).with_name("books.db")


def scrape_books(max_pages=5):
    rows = []

    for page in range(1, max_pages + 1):
        url = BASE_URL if page == 1 else f"{BASE_URL}catalogue/page-{page}.html"
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for article in soup.select("article.product_pod"):
            title = article.h3.a.get("title", "").strip()
            price_text = article.select_one(".price_color").get_text(strip=True)
            rating = article.select_one("p.star-rating")
            rating_text = next(
                (c for c in rating.get("class", []) if c != "star-rating"),
                "Unknown",
            )
            availability = article.select_one(".availability").get_text(" ", strip=True)

            # Category is available from each book page.
            href = article.h3.a.get("href")
            book_url = requests.compat.urljoin(url, href)
            detail = requests.get(book_url, timeout=20)
            detail.raise_for_status()
            detail_soup = BeautifulSoup(detail.text, "html.parser")

            breadcrumb = detail_soup.select("ul.breadcrumb li")
            category = breadcrumb[2].get_text(strip=True) if len(breadcrumb) >= 3 else "Unknown"

            rows.append(
                {
                    "title": title,
                    "price": price_text,
                    "star_rating": rating_text,
                    "availability": availability,
                    "category": category,
                }
            )

    return pd.DataFrame(rows)


def clean_books(df):
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }

    out = df.copy()

    # Clean price and convert it to numeric.
    out["price_gbp"] = (
        out["price"]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.replace("Â", "", regex=False)
        .str.strip()
    )
    out["price_gbp"] = pd.to_numeric(out["price_gbp"], errors="coerce")

    # Convert rating words to integers.
    out["rating"] = out["star_rating"].map(rating_map)

    # Convert availability text to Boolean.
    out["in_stock"] = out["availability"].str.contains(
        "In stock",
        case=False,
        na=False,
    )

    # Handle numeric parsing failures using median imputation.
    price_median = out["price_gbp"].median()

    if pd.isna(price_median):
        raise ValueError(
            "All price values failed to parse. Check the scraped price format."
        )

    out["price_gbp"] = out["price_gbp"].fillna(price_median)

    rating_median = out["rating"].median()

    if pd.isna(rating_median):
        raise ValueError(
            "All rating values failed to parse. Check the scraped rating format."
        )

    out["rating"] = (
        out["rating"]
        .fillna(rating_median)
        .round()
        .astype(int)
    )

    # Fixed project-defined conversion.
    out["price_inr"] = out["price_gbp"] * GBP_TO_INR

    # Drop rows where required text fields are unavailable.
    out = out.dropna(subset=["title", "category"]).copy()

    # Final safety check before SQLite insertion.
    out = out.dropna(
        subset=["price_gbp", "price_inr", "rating", "in_stock", "category"]
    ).copy()

    return out[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ]


def load_sqlite(df, db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.executescript(
            """
            DROP TABLE IF EXISTS books;
            DROP TABLE IF EXISTS categories;

            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL,
                in_stock INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id) REFERENCES categories(category_id)
            );
            """
        )

        categories = pd.DataFrame({"category_name": sorted(df["category"].unique())})
        categories.to_sql("categories", conn, if_exists="append", index=False)

        category_ids = pd.read_sql(
            "SELECT category_id, category_name FROM categories", conn
        )
        merged = df.merge(
            category_ids,
            left_on="category",
            right_on="category_name",
            how="left",
        )
        books = merged[
            ["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]
        ].copy()
        books["in_stock"] = books["in_stock"].astype(int)
        books.to_sql("books", conn, if_exists="append", index=False)


def run_queries(db_path=DB_PATH):
    queries = {
        "select_where": """
            SELECT title, price_inr
            FROM books
            WHERE price_inr > 1000;
        """,
        "order_by": """
            SELECT title, rating
            FROM books
            ORDER BY rating DESC, title
            LIMIT 10;
        """,
        "distinct": """
            SELECT DISTINCT category_name
            FROM categories
            ORDER BY category_name;
        """,
        "between": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30
            ORDER BY price_gbp;
        """,
        "join": """
            SELECT b.title, b.rating, c.category_name
            FROM books b
            JOIN categories c
                ON b.category_id = c.category_id
            ORDER BY b.rating DESC, b.title
            LIMIT 10;
        """,
    }

    output_path = Path(__file__).with_name("sql_results.txt")

    with sqlite3.connect(db_path) as conn:
        results = {}

        with open(output_path, "w", encoding="utf-8") as file:
            for name, query in queries.items():
                result = pd.read_sql(query, conn)
                results[name] = result

                # Save query string.
                file.write(f"\n{'=' * 80}\n")
                file.write(f"QUERY: {name}\n")
                file.write(f"{'=' * 80}\n")
                file.write(query.strip())
                file.write("\n\nOUTPUT:\n")
                file.write(result.to_string(index=False))
                file.write("\n")

                # Still display the result in the terminal.
                print(f"\n--- {name} ---")
                print(result.to_string(index=False))

    print(f"\nSQL queries and outputs saved to: {output_path}")

    return results


def main():
    raw = scrape_books(max_pages=5)
    if len(raw) < 60:
        raise RuntimeError(f"Expected at least 60 books, got {len(raw)}")

    cleaned = clean_books(raw)
    cleaned.to_csv(Path(__file__).with_name("cleaned_books.csv"), index=False)
    load_sqlite(cleaned)
    results = run_queries()

    # Reproduce the JOIN result with pandas.
    category_df = cleaned[["category"]].drop_duplicates().reset_index(drop=True)
    category_df["category_id"] = category_df.index + 1

    books_for_merge = cleaned.copy()
    books_for_merge["category_id"] = books_for_merge["category"].map(
        dict(zip(category_df["category"], category_df["category_id"]))
    )

    pandas_join = (
        books_for_merge[
            ["title", "rating", "category_id"]
        ]
        .merge(
            category_df,
            on="category_id",
            how="inner",
        )
        [["title", "rating", "category"]]
        .sort_values(
            ["rating", "title"],
            ascending=[False, True],
        )
        .head(10)
    )

    pandas_join.columns = ["title", "rating", "category_name"]

    print("\n--- SQL JOIN vs pandas.merge ---")
    print(results["join"].to_string(index=False))
    print(pandas_join.to_string(index=False))


if __name__ == "__main__":
    main()
