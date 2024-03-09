import json
import numpy as np
from numpy.linalg import norm
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import argparse
import time 

class RetreivalAugmentation:
    def __init__(self, threshold):
        self.ps = PorterStemmer()
        self.inverted_index = {}
        self.index = {}
        self.query = ''
        self.tfidf_table  = {}
        self.query_token_docs = set()
        self.threshold = threshold

        
    def tokenize(self, text, corpus_tokens):
        token_to_count = defaultdict(int)
        tokens = word_tokenize(text)
        for t in tokens:
            stem_token = self.ps.stem(t.lower())
            if stem_token in corpus_tokens:
                token_to_count[stem_token] += 1
        return token_to_count

    def generate_query_vector(self, tokens, query_token_count):
        query_vector = []
        for token in tokens:
            if token in query_token_count:
                query_vector.append(query_token_count[token])
            else:
                query_vector.append(0)
        return query_vector
    
    def generate_document_vector(self, doc_id, tokens):
        doc_vector = []
        for token in tokens:
            if token in self.index[doc_id]:
                doc_vector.append(self.index[doc_id][token])
            else:
                doc_vector.append(0)
        return doc_vector

    def cosine_similarity(self, vector_a, vector_b):
        return np.dot(vector_a,vector_b)/(norm(vector_a)*norm(vector_b))

    def start(self):
        """
        Called on Each query
        """
        print('start query')
        currtime = time.perf_counter()
        tokens = list(self.inverted_index.keys())    
        query_token_count =  self.tokenize(self.query, tokens)
        query_vector = self.generate_query_vector(tokens, query_token_count)
        for token in query_token_count:
            self.query_token_docs.update(self.inverted_index[token].keys())
        
        doc_ids_list = list(self.query_token_docs)

        updated_doc_ids = {}

        print('no of docs for cosine sim' , len(doc_ids_list))
        time1 = time.perf_counter()
        for doc_id in doc_ids_list:
            time3 = time.perf_counter()
            document_vector = self.generate_document_vector(doc_id, tokens)
            print('generate doc vector- ',  time.perf_counter() - time3)
            time4 = time.perf_counter()
            updated_doc_ids[doc_id] = self.cosine_similarity(query_vector,document_vector)
            print('cosine func - ',  time.perf_counter() - time4)
            
        print('cosine similarity - ',  time.perf_counter() - time1)
        sorted_doc_id_dict = dict(sorted(updated_doc_ids.items(), key=lambda item: item[1]))
        threshold_doc_id_dict = dict(list(sorted_doc_id_dict.items())[-self.threshold:])
        
        tfidf_docid_dict = {}
        time2 = time.perf_counter()
        for token in query_token_count:
            for doc_id in threshold_doc_id_dict:
                if doc_id in self.tfidf_table[token]:
                    if doc_id not in tfidf_docid_dict:
                        tfidf_docid_dict[doc_id] = self.tfidf_table[token][doc_id]
                    else:
                        tfidf_docid_dict[doc_id] += self.tfidf_table[token][doc_id]
        print('idf score loop - ',  time.perf_counter() - time2)
        sorted_tfidf_docid_dict = dict(sorted(tfidf_docid_dict.items(), key=lambda item: item[1]))
        top_results = dict(list(sorted_tfidf_docid_dict.items())[-5:])
        print('final time', currtime - time.perf_counter())
        print(top_results.keys())
        print('-'*100)
        return top_results.keys()
    
    def query_call(self, parser):
        # Add arguments
        # print("Query called")
        # parser.add_argument('query', type=str, help='Your Query')
        # args = parser.parse_args()
        self.query = "Iftekhar ahmed"
        self.start()
        self.query = "machine learning"
        self.start()
        self.query = "ACM"
        self.start()
        self.query = "master of software engineering"
        self.start()
        

    def initate_call(self):
        print("Initiate call called")
        parser = argparse.ArgumentParser(
            description='A simple CLI example',
            epilog="Hey welcome to this Command line interface"
        )
        with open('index.json', 'r') as index_file:
            self.index = json.load(index_file)

        with open('inverted_index.json', 'r') as inverted_index_file:
            self.inverted_index = json.load(inverted_index_file)
        
        with open('tfidf.json', 'r') as tfidf_file:
            self.tfidf_table = json.load(tfidf_file)
        self.query_call(parser)


if __name__ == "__main__":
    rag = RetreivalAugmentation(200)
    rag.initate_call()





