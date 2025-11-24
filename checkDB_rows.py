import sqlite3
conn = sqlite3.connect("betting_edge.db")
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM matches")
print("Matches:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM match_stats")
print("Match Stats:", cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM odds")
print("Odds:", cursor.fetchone()[0])

conn.close()
