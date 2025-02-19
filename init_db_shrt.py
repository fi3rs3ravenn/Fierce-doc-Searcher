import os
import sqlite3
import openai
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

openai.api_key = "my api key"

DB_NAME = 'docs.db'

def summurize_text(text):
    prompt = f"""
    Read the text and formulate its content, describing it in detail and highlighting the main ideas:
    """
    try:
        response = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo',
            messages = [
                {'role':'system','content':'You are an assistant creating detailed document descriptions'},
                {'role':'user', 'content': prompt + text}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f'Error: {e}'

def process_and_update_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('ALTER TABLE docs ADD COLUMN summary TEXT')
    conn.commit()

    cursor.execute('SELECT filename, content FROM docs')
    docs = cursor.fetchall()

    for filename, content in docs:
        if content:
            summary = summurize_text(content)
            cursor.execute('UPDATE docs SET summary = ? WHERE filename = ?', (summary, filename))
            conn.commit()
            print(f'developed {filename}')

    conn.close()

if __name__ == '__main__':
    print('Starting work on docs...')
    process_and_update_db()
    print('Up to UpDate!!!')