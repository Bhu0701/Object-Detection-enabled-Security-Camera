import psycopg2

conn = psycopg2.connect(
    database="postgres",
    user="postgres",
    password="12345",
    host="localhost",
    port='5432'
)

cursor = conn.cursor()

# Drop table if exists and create new one/


create_table_query = """
DROP TABLE IF EXISTS student;
CREATE TABLE student (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(50) NOT NULL,
    branch VARCHAR(50)
);
"""

cursor.execute(create_table_query)
conn.commit()

# Test insert some sample data
cursor.execute("""
INSERT INTO student (name, username, password, branch) 
VALUES ('Test Student', 'test_user', 'test123', 'Computer Science')
""")
conn.commit()

# Verify the table contents
cursor.execute("SELECT * FROM student")
data = cursor.fetchall()
for d in data:
    print(d)

conn.close()