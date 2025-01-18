import sqlite3

def initialize_database():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            text TEXT,
            difficulty TEXT
        )
    """)
    conn.commit()
    conn.close()

def store_content(title, text, difficulty="medium"):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO content (title, text, difficulty) VALUES (?, ?, ?)", (title, text, difficulty))
    conn.commit()
    conn.close()

def retrieve_content(query):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM content WHERE text LIKE ?", (f"%{query}%",))
    results = cursor.fetchall()
    conn.close()
    return " ".join(row[0] for row in results) if results else "No relevant content found."
