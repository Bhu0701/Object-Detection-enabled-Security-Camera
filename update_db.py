import psycopg2

def get_db_connection():
    return psycopg2.connect(
        database="postgres",
        user="postgres",
        password="12345",
        host="localhost",
        port='5432'
    )

# Connect to database
conn = get_db_connection()
cur = conn.cursor()

# Drop existing table and create new one with session tracking
cur.execute("""
DROP TABLE IF EXISTS student;
CREATE TABLE student (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(50) NOT NULL,
    branch VARCHAR(50),
    last_login TIMESTAMP,
    session_start TIMESTAMP,
    session_end TIMESTAMP,
    total_time_spent INTERVAL
);
""")

conn.commit()
cur.close()
conn.close()
print("Database updated successfully!")
