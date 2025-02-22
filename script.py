import os
from rank_bm25 import BM25Okapi
import docx 
import openai
import nltk
from nltk.corpus import stopwords
import pymorphy3
import sqlite3

openai.api_key = ""

DB_NAME = 'docs.db'

nltk.download('stopwords')
nltk.download('punkt')  
stop_words = set(stopwords.words('russian'))

morph = pymorphy3.MorphAnalyzer()

def preprocess_text(text):
    words = text.split()
    lemmas = [morph.parse(word.lower())[0].normal_form for word in words if word.lower() not in stop_words]
    return lemmas

def load_docs_from_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT filename, content, summary FROM docs')
    docs = {}
    for row in cursor.fetchall():
        filename = row[0]
        content = row[1] if row[1] else ''
        summary = row[2] if row[2] else ''
        docs[filename] = {'content': content , 'summary':summary}
    conn.close()
    return docs

def index_doc(documents):
    tokenized_docs = []
    for doc in documents.values():
        text_for_index = doc['summary'] if doc['summary'] else doc['content']
        tokenized_docs.append(preprocess_text(text_for_index))
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, tokenized_docs

def searc_docs(query, docs, bm25, top_n=3):
    if not docs:
        return []

    tokenized_query = preprocess_text(query)
    if not tokenized_query:  
        return []

    scores = bm25.get_scores(tokenized_query)
    if not any(scores):  
        return []

    best_match_index = scores.argsort()[-top_n:][::-1]
    doc_keys = list(docs.keys())
    results = [(doc_keys[idx], docs[doc_keys[idx]]) for idx in best_match_index]
    return results

def split_text(text, chunk_size=1500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def analyze_gpt(query , top_docs):
    refined_results = []
    for doc_name , doc_data in top_docs:
        text_to_analyze = doc_data.get('summary') if doc_data.get('summary') else doc_data.get('content')
        text_chunks = split_text(text_to_analyze)
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
    print('Loading documents from database...')
    docs = load_docs_from_db()
    print(f'Loaded {len(docs)} documents.')

    print('Indexing documents...')
    bm25, tokenized_docs = index_doc(docs)
    print('Indexing completed.')

    while True:
        user_query = input('Enter the law name (or / to exit): ')
        if user_query == '/':
            break 

        top_docs = searc_docs(user_query, docs, bm25 , top_n=3)
        refined_results = analyze_gpt(user_query, top_docs)

        if refined_results:
            print('\nThe most relevant docs:')
            for i , (doc_name, gpt_answer) in enumerate(refined_results, 1):
                print(f'\n Doc {i} --> {doc_name}')
                print(f'AI answer --> {gpt_answer[:500]}...')  
        else:
            print('No relevant documents found.')
