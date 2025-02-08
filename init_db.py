import sqlite3
import os
import docx

DB_NAME = 'docs.db'

def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            content TEXT   
        )
    """)
    conn.commit()
    conn.close()

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip() != ''])
    return text 

def add_doc(filename, content):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO docs (filename, content) VALUES (?, ?)', (filename, content))
        conn.commit()
    except sqlite3.IntegrityError:
        pass 
    conn.close()

def update_db(folder_path='base of doc'):
    create_database()
    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_docx(file_path)
            add_doc(filename, text)

if __name__ == '__main__':
    print('updt db with docs...')
    update_db()
    print('updt complete')