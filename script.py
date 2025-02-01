import os
from rank_bm25 import BM25Okapi
import docx 
import openai

openai.api_key = "my own api key"

def load_docs(folder_path='base of doc'):
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            file_path = os.path.join(folder_path, filename)
            docs[filename] = extract_text_from_docx(file_path)
    return docs

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip() != ''])
    return text 

def index_doc(documents):
    tokenized_docs = [doc.split() for doc in documents.values()]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25 , tokenized_docs

def searc_docs(query , docs, bm25, top_n=3):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    best_match_index = scores.argsort()[-top_n:][::-1]  
    
    results = []
    for idx in best_match_index:
        best_doc_name = list(docs.keys())[idx]
        best_doc_text = list(docs.values())[idx]
        results.append((best_doc_name, best_doc_text))

    return results

def analyze_gpt(query , top_docs):
    refined_results = []
    for doc_name , doc_text in top_docs:
        prompt = f"""
        I have text of law:
        {doc_text[:2000]}
        Customer have a question: '{query}'
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
            refined_results.append((doc_name, gpt_answer))
        except Exception as e:
            refined_results.append((doc_name, f'Error {e}'))
        
        if len(refined_results) == 3:
            break
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
    