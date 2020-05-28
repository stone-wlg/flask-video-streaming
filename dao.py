import os
import psycopg2

connection = os.environ.get('CONNECTION')
conn = psycopg2.connect(connection)
cur = conn.cursor()

def query(sql):
  cur = conn.cursor()
  cur.execute(sql)
  data = cur.fetchall()
  cur.close()
  return data

def insert(sql):
  cur = conn.cursor()
  cur.execute(sql)
  conn.commit()
  cur.close()
