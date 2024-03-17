import json
import gzip
import math
import time
import numpy as np
from numpy.linalg import norm
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class RetreivalAugmentation:
    def __init__(self, threshold):
        self.ps = PorterStemmer()
        self.inverted_index = {}
        self.pagerank = {}
        self.query = ''
        self.tfidf_table  = {}
        self.query_token_docs = set()
        self.inverted_index_dict= {}
        self.pagerank_scores = {}
        

    def tokenize(self, text, corpus_tokens):
        token_to_count = defaultdict(int)
        tokens = word_tokenize(text)
        for t in tokens:
            stem_token = self.ps.stem(t.lower())
            if stem_token in corpus_tokens:
                token_to_count[stem_token] += 1
        return token_to_count
        
    def start(self):
        """
        Called on Each query
        """
        currTime = time.perf_counter()
        query_token_count =  self.tokenize(self.query, self.tokens)
        query_idf_weights = {term: 1+ math.log10(query_token_count[term])
                             for term in query_token_count.keys()}
        
        query_tfidf_relevance_score = defaultdict(float)
                
        #Cosine similarity calculations
        for term in query_token_count:
            for url in self.inverted_index[term]:
                tfidf = self.tfidf_table[term][url]  
                tfidf_score = query_idf_weights[term] * tfidf   
                query_tfidf_relevance_score[url] += tfidf_score
        
        # Calculate relevance score based on cosine similarity and pagerank
        for url, score in query_tfidf_relevance_score.items():
            query_tfidf_relevance_score[url] = 0.8*score + 0.2*self.pagerank[url] 

        sorted_query_tfidf_relevance_score = dict(sorted(query_tfidf_relevance_score.items(), key=lambda item: item[1]))
        top_results = dict(list(sorted_query_tfidf_relevance_score.items())[-5:])
        
        for result in reversed(top_results):
            print(f"- {result, sorted_query_tfidf_relevance_score[result]} \n")
        endTime = time.perf_counter()
        print('Time taken to run this query - ', endTime-currTime)
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
          
    def calculate_page_rank(self):
        url_graph = nx.DiGraph()
        for token , url_dict in self.inverted_index.items():     
            for url, count in url_dict.items():
                url_graph.add_node(url)
            url_graph.add_edges_from([(url, terms[0]) for terms in url_dict])

        self.pageScores = nx.pagerank(url_graph)
        return 
    
    def initate_call(self):
        print('--- Reading Inverted Index, pagerank and TfIdf Json ---- ')
        
        with gzip.open('compressed_pagerank.json.gz', 'rb') as pagerank_file:
            compressed_pagerank_file = pagerank_file.read()
            decompressed_pagerank_file = gzip.decompress(compressed_pagerank_file)
            self.pagerank = json.loads(decompressed_pagerank_file.decode('utf-8'))
        
        with gzip.open('compressed_inverted_index.json.gz', 'rb') as inverted_index_file:
            compressed_inverted_index_file = inverted_index_file.read()
            decompressed_inverted_index_file = gzip.decompress(compressed_inverted_index_file)
            self.inverted_index = json.loads(decompressed_inverted_index_file.decode('utf-8'))
            self.inverted_index_dict = {key: index for index, key in enumerate(self.inverted_index.keys())}

        with gzip.open('compressed_tfidf.json.gz', 'rb') as tfidf_file:
            compressed_tfidf_file = tfidf_file.read()
            decompressed_tfidf_file = gzip.decompress(compressed_tfidf_file) 
            self.tfidf_table = json.loads(decompressed_tfidf_file.decode('utf-8'))
        

        print("Please wait before entering a query")
        #self.calculate_page_rank()
        self.query_call()


if __name__ == "__main__":
    rag = RetreivalAugmentation(200)
    rag.initate_call()











