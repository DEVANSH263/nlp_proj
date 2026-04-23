import sqlite3

conn = sqlite3.connect(r'c:\Documents\Projects\ss\database.db')
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cur.fetchall()]
print("Tables:", tables)

for t in tables:
    cur.execute(f"PRAGMA table_info({t})")
    cols = [c[1] for c in cur.fetchall()]
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    count = cur.fetchone()[0]
    print(f"\n--- {t} ({count} rows) ---")
    print("Columns:", cols)
    cur.execute(f"SELECT * FROM {t} LIMIT 5")
    for row in cur.fetchall():
        print(row)

conn.close()
