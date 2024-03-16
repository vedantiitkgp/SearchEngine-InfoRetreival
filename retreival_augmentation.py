import json
import gzip
import numpy as np
from numpy.linalg import norm
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class RetreivalAugmentation:
    def __init__(self, threshold):
        self.ps = PorterStemmer()
        self.inverted_index = {}
        self.index = {}
        self.query = ''
        self.tfidf_table  = {}
        self.query_token_docs = set()
        self.threshold = threshold
        self.inverted_index_dict= {}


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
    
    def generate_document_vector(self, doc_id, doc_vector, index_dict):
        for token in self.index[doc_id]:
            doc_vector[index_dict[token]] = self.index[doc_id][token]
        return doc_vector

    def start(self):
        """
        Called on Each query
        """
        tokens = list(self.inverted_index.keys())
        query_token_count =  self.tokenize(self.query, tokens)
        query_vector = self.generate_query_vector(tokens, query_token_count)
        self.query_token_docs = set()
        for token in query_token_count:
            self.query_token_docs.update(self.inverted_index[token].keys())
        doc_ids_list = list(self.query_token_docs)

        print("No of Docs Related to this query :", len(doc_ids_list))

        filtered_docid_dict = {}
        for token in query_token_count:
            for doc_id in doc_ids_list:
                if doc_id in self.tfidf_table[token]:
                    if doc_id not in filtered_docid_dict:
                        filtered_docid_dict[doc_id] = self.tfidf_table[token][doc_id]
                    else:
                        filtered_docid_dict[doc_id] += self.tfidf_table[token][doc_id]


        sorted_filtered_docid_dict = dict(sorted(filtered_docid_dict.items(), key=lambda item: item[1]))
        doc_ids_list = list(dict(list(sorted_filtered_docid_dict.items())[-500:]).keys())
      
        doc_id_vectors = [[0]*len(tokens) for _ in range(len(doc_ids_list))]
        
        for i, doc_id in enumerate(doc_ids_list):
            doc_id_vectors[i] = self.generate_document_vector(doc_id, doc_id_vectors[i], self.inverted_index_dict)

        cosine_doc_ids = list(cosine_similarity([query_vector],doc_id_vectors)[0])
        updated_doc_ids = {key: value for key, value in zip(doc_ids_list, cosine_doc_ids)}

        sorted_doc_id_dict = dict(sorted(updated_doc_ids.items(), key=lambda item: item[1]))
        threshold_doc_id_dict = dict(list(sorted_doc_id_dict.items())[-self.threshold:])

        tfidf_docid_dict = {}
        for token in query_token_count:
            for doc_id in threshold_doc_id_dict:
                if doc_id in self.tfidf_table[token]:
                    if doc_id not in tfidf_docid_dict:
                        tfidf_docid_dict[doc_id] = self.tfidf_table[token][doc_id]
                    else:
                        tfidf_docid_dict[doc_id] += self.tfidf_table[token][doc_id]

        sorted_tfidf_docid_dict = dict(sorted(tfidf_docid_dict.items(), key=lambda item: item[1]))
        top_results = dict(list(sorted_tfidf_docid_dict.items())[-5:])
        for result in reversed(top_results):
            print(f"- {result} \n")
        print('-'*100)
        return

    def query_call(self):
        print("-" * 50)
        print("-" + " " * 48 + "-")
        print("-" + " " * 6 + "/ \\ / \\ / \\ / \\ / \\ / \\" + " " * 6 + "-")
        print("-" + " " * 6 + "\\ / \\ / \\ / \\ / \\ / \\/" + " " * 6 + "-")
        print("-" + " " * 6 + "/ \\ / \\ / \\ / \\ / \\ / \\" + " " * 6 + "-")
        print("-" + " " * 6 + "\\ / \\ / \\ / \\ / \\ / \\/" + " " * 6 + "-")
        print("-" + " " * 48 + "-")
        print("-" * 50)
        query = input("Enter your Query: ")
        print("--- Processing this query ---")
        self.query = query
        self.start()
        while True:
          permission = input("Do you want to run another query: (y/n) ")
          if permission == "y":
            query = input("Enter your new Query: ")
            print("--- Processing this query ---")
            self.query = query
            self.start()
          else:
            break

    def initate_call(self):
        print('--- Reading Index and TfIdf Json ---- ')

        with gzip.open('compressed_index.json.gz', 'rb') as index_file:
            compressed_index_file = index_file.read()
            decompressed_index_file = gzip.decompress(compressed_index_file)
            self.index = json.load(decompressed_index_file)
        
        with gzip.open('compressed_inverted_index.json.gz', 'rb') as inverted_index_file:
            compressed_inverted_index_file = inverted_index_file.read()
            decompressed_inverted_index_file = gzip.decompress(compressed_inverted_index_file)
            self.inverted_index = json.load(decompressed_inverted_index_file)
            self.inverted_index_dict = {key: index for index, key in enumerate(self.inverted_index.keys())}

        with gzip.open('compressed_tfidf.json.gz', 'rb') as tfidf_file:
            compressed_tfidf_file = tfidf_file.read()
            decompressed_tfidf_file = gzip.decompress(compressed_tfidf_file) 
            self.tfidf_table = json.load(decompressed_tfidf_file)

        print("Please wait before entering a query")
        self.query_call()


if __name__ == "__main__":
    rag = RetreivalAugmentation(200)
    rag.initate_call()











