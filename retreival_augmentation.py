import json
import os
import gzip
import math
import time
from tkinter import ttk as tk
from tkinter import *
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import networkx as nx

class RetreivalAugmentation:
    def __init__(self, tfidf_folder, neighbors_folder):
        self.tfidf_folder = tfidf_folder
        self.neighbors_folder = neighbors_folder
        self.ps = PorterStemmer()
        self.pagerank = defaultdict(dict)
        self.query = ''
        self.tfidf_table  = defaultdict(dict)
        self.query_token_docs = set()
        self.pagerank_scores = defaultdict(dict)
        self.results = []
        self.G = nx.Graph() 
        


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
            for url in self.tfidf_table[term]:
                tfidf = self.tfidf_table[term][url]  
                tfidf_score = query_idf_weights[term] * tfidf   
                query_tfidf_relevance_score[url] += tfidf_score
        
        # Calculate relevance score based on cosine similarity and pagerank
        for url, score in query_tfidf_relevance_score.items():
            query_tfidf_relevance_score[url] = 0.8*score +10*self.pagerank.get(url,0) 

        sorted_query_tfidf_relevance_score = dict(sorted(query_tfidf_relevance_score.items(), key=lambda item: item[1]))
        top_results = list(sorted_query_tfidf_relevance_score.keys())[-5:]
        
        for result in reversed(top_results):
            print(f"- {result}, {sorted_query_tfidf_relevance_score[result]} \n")

        endTime = time.perf_counter()
        print('Time taken to run this query - ', endTime-currTime)
        print('-'*100)
        top_results.reverse()
        return top_results

    def query_call(self, inp): 
        self.query = inp
        self.results = self.start()

    def open_link(event, url):
        # Define the URL you want to open
        # Code to open the URL in a web browser
        # This example uses the webbrowser module
        import webbrowser
        webbrowser.open(url)
    
    def show_results(self, query):
        root1 = Tk()
        root1.title('Top 5 results')
        root1.geometry('450x300') 
        frm = tk.Frame(root1, padding=10)
        frm.grid()
        heading = 'Top 5 results for Query - '
        if not self.results:
            heading = 'No Results returned for query'
        tk.Label(frm, text=heading+query, font=("Arial", 12, "bold")).grid(column=0, row=0)

        if self.results:
            # Create the labels for each result and bind them to open links
            lbls = []  # List to hold label widgets
            for i in range(5):
                lbl = tk.Label(frm, text=self.results[i], foreground="blue", cursor="hand2")
                lbl.grid(column=0, row=i+1)
                lbl.bind("<Button-1>", lambda event, link=self.results[i]: self.open_link(link))
                lbls.append(lbl)  # Append label to the list
        
        # Create a Quit button to close the window
        tk.Button(frm, text="Quit", command=root1.destroy).grid(column=0, row=7)
        
        root1.mainloop()

        
    def initate_call(self):
        print('---  Reading TfIdf Json ---- ')
        for filename in os.listdir(self.tfidf_folder):
            if filename.endswith('.gz'):
                file_path = os.path.join(self.tfidf_folder, filename)
                with gzip.open(file_path, 'rb') as tfidf_file:
                    compressed_tfidf_file = tfidf_file.read()
                    decompressed_tfidf_file = gzip.decompress(compressed_tfidf_file) 
                    partial_tfidf_index = json.loads(decompressed_tfidf_file.decode('utf-8'))
                    for term, postings in partial_tfidf_index.items():
                        for url in postings:
                            self.tfidf_table[term][url] = postings[url]
        self.tokens = self.tfidf_table.keys()

        for filename in os.listdir(self.neighbors_folder):
            if filename.startswith('neighbors') and filename.endswith('.gz'):
                file_path = os.path.join(self.neighbors_folder, filename)
                with gzip.open(file_path, 'rb') as neighbors_file:
                    compressed_neighbors_file = neighbors_file.read()
                    decompressed_neighbors_file = gzip.decompress(compressed_neighbors_file) 
                    partial_neighbors_index = json.loads(decompressed_neighbors_file.decode('utf-8'))
                    for url in partial_neighbors_index:
                        for link in partial_neighbors_index[url]:
                            self.G.add_edge(url,link)

        self.pagerank = nx.pagerank(self.G)

        def search_query():
            inp = entry.get()  # Get the search query from the entry widget
            # Perform the search (replace this with your actual search function)
            self.query_call(inp)
            self.show_results(inp)

        root = Tk()
        root.title('Search Engine')
        root.geometry('450x300') 
        entry = tk.Entry(root, textvariable = 'Search Engine', justify = CENTER) 
  
        # focus_force is used to take focus 
        # as soon as application starts 
        entry.focus_force() 
        entry.pack(side = TOP, ipadx = 30, ipady = 6) 
        
        search = tk.Button(root, text = 'Search', command = lambda : search_query()) 
        search.pack(side = TOP, pady = 10) 
        
        root.mainloop()
                              
        #GUI partially taken from https://www.geeksforgeeks.org/how-to-get-the-input-from-tkinter-text-box/
        
        
if __name__ == "__main__":
    rag = RetreivalAugmentation('tfidf','index')
    rag.initate_call()











