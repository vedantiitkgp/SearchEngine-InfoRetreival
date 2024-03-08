import os
import json
import time
import nltk
import re
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from bs4 import BeautifulSoup

class Inverted_Indexer:
    def __init__(self, dataset_path, index_file, inverted_index_file, tfidf_file):
        self.dataset_path = dataset_path
        self.index_file = index_file
        self.inverted_index_file = inverted_index_file
        self.tfidf_file = tfidf_file

        self.ps = PorterStemmer()
        self.doc_id = 0
        self.doc_id_to_url = {}
        self.inverted_index = defaultdict(dict)
        self.index= defaultdict(dict)
        self.count_docs = defaultdict(int)
        self.tfidf_table = defaultdict(dict)

    
    def process(self, file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            file = json.load(f)
            self.doc_id += 1
            self.doc_id_to_url = file["url"]
            soup = BeautifulSoup(file["content"], "html.parser")
            token_to_count = self.tokenize(soup.get_text(separator=' '))
            
            for token, count in token_to_count.items():
                self.inverted_index[token][file["url"]] = count
                self.index[file['url']][token] = count
                if not file['url'] in self.count_docs:
                    self.count_docs[file["url"]] = count
                else:
                    self.count_docs[file["url"]] += count
            
            ## Adding extra weight to heading words
            headings = soup.find_all(re.compile('^h[1-6]$'))
            for heading in headings:
                heading_text = heading.get_text(separator=' ').strip()
                heading_tokens = self.tokenize(heading_text)
                for heading_token in heading_tokens:
                    if heading_token in self.inverted_index:
                        if file['url'] in self.inverted_index[heading_token]:
                            self.inverted_index[heading_token][file["url"]] += 1
                        else:
                            print("CASE 1",heading_token,file['url'])
                            self.inverted_index[heading_token][file["url"]] = 1
                    else:
                        print("CASE 2",heading_token)
                        self.inverted_index[heading_token][file["url"]] = 1

    def tokenize(self, text):
        token_to_count = defaultdict(int)
        tokens = word_tokenize(text)
        for t in tokens:
            token_to_count[str(self.ps.stem(t.lower()))] += 1
        return token_to_count

    def dump(self):
        data = defaultdict(dict)
        with open(self.inverted_index_file, "w", encoding='utf-8') as f:
            json.dump(self.inverted_index, f)
        with open(self.index_file, "w", encoding='utf-8') as f:
            json.dump(self.index, f)

    """
    Function to Calculate TF-IDF Score
    """
    def calculate_dump_tfidf(self):
        for term , term_dict in self.inverted_index.items():
            count_urls = len(term_dict)
            for url, count in term_dict.items():
                term_freq = count / self.count_docs[url]
                inverse_doc_freq = self.doc_id / count_urls
                self.tfidf_table[term][url] = term_freq * inverse_doc_freq
            
        with open(self.tfidf_file, "w", encoding='utf-8') as f:
            json.dump(self.tfidf_table, f)
    
    def start(self):
        start = time.perf_counter()

        count = 0
        for subdir, dirs, files in os.walk(self.dataset_path):
            for name in files:
                if name != ".DS_Store":
                    self.process(subdir + os.sep + name)
                    print(name, "Done")
            count = count + 1

        self.dump()
        self.calculate_dump_tfidf()

        end = time.perf_counter()
        print("START - Total number of documents:{}".format(self.doc_id))
        print("START - Total Number of tokens: {}".format(len(self.inverted_index)))
        print("START - Program execution time: {} seconds".format(end - start))

if __name__ == '__main__':
    dataset_path = "DEV/"
    inverted_index_file = "inverted_index.json"
    index_file = "index.json"
    tfidf_file = "tfidf.json"

    i = Inverted_Indexer(dataset_path, index_file, inverted_index_file, tfidf_file)
    i.start()