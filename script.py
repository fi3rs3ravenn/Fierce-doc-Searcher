import os
from rank_bm25 import BM25Okapi
import docx 
import openai
import nltk
from nltk.corpus import stopwords
import pymorphy2
import sqlite3

openai.api_key = ""

DB_NAME = 'docs.db'

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text):
    words = text.split()
    lemmas = [morph.parse(word.lower())[0].normal_form for word in words if word.lower() not in stop_words]
    return lemmas


def load_docs():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT filename, content FROM docs')
    docs = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return docs

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip() != ''])
    return text 

def index_doc(documents):
    tokenized_docs = [preprocess_text(doc) for doc in documents.values() if doc]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, tokenized_docs

def searc_docs(query, docs, bm25, top_n=3):
    if not docs:
        return []

    tokenized_query = preprocess_text(query)
    scores = bm25.get_scores(tokenized_query)
    best_match_index = scores.argsort()[-top_n:][::-1]  

    doc_names, doc_texts = zip(*docs.items()) if docs else ([], [])

    results = [(doc_names[idx], doc_texts[idx]) for idx in best_match_index]
    return results

def split_text(text, chunk_size=1500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def analyze_gpt(query , top_docs):
    refined_results = []
    for doc_name , doc_text in top_docs:
        text_chunks = split_text(doc_text)
        full_answer = ""

        for chunk in text_chunks:
            prompt = f"""
            I have text of law:
            {chunk}
            Customer has a question: '{query}'
            Give an answer according to text of doc.
            """
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {'role':'system' , 'content':'you are a helper in searching and analyzing docs'},
                        {'role':'user' , 'content':prompt}
                    ]
                )
                gpt_answer = response['choices'][0]['message']['content']
                full_answer += " " + gpt_answer
            except Exception as e:
                full_answer += f" Error: {e}"

        refined_results.append((doc_name, full_answer))
    
    return refined_results

if __name__ == '__main__':
    print('loading docs...')
    docs = load_docs()
    print(f'loaded {len(docs)}')

    print('indexation of docs...')
    bm25, tokenized_docs = index_doc(docs)
    print('The end of indexation')

    while True:
        user_query = input('Enter the law name (or / to exit) :  ')
        if user_query == '/':
            break 

        top_docs = searc_docs(user_query, docs, bm25 , top_n=3)
        refined_results = analyze_gpt(user_query, top_docs)

        if refined_results:
            print('\n The most relevant docs:')
            for i , (doc_name, gpt_answer) in enumerate(refined_results, 1):
                print(f'\n Doc {i} --> {doc_name}')
                print(f'AI answer --> {gpt_answer}')
        else:
            print('there is no docs')
