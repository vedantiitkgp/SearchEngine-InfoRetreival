import os
import json
import time
import nltk
import gzip
import re
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from bs4 import BeautifulSoup
import networkx as nx
from urllib.parse import urlparse, urljoin, parse_qs


class Inverted_Indexer:
    def __init__(self, dataset_path, pagerank_file, inverted_index_file, tfidf_file):
        self.dataset_path = dataset_path
        self.pagerank_file = pagerank_file
        self.inverted_index_file = inverted_index_file
        self.tfidf_file = tfidf_file
        self.G = nx.DiGraph()
        self.ps = PorterStemmer()
        self.doc_id = 0
        self.doc_id_to_url = {}
        self.inverted_index = defaultdict(dict)
        self.pagerank= defaultdict(dict)
        self.count_docs = defaultdict(int)
        self.tfidf_table = defaultdict(dict)  

    def modify_if_relative(self, relative_url,parent_url):
        if relative_url:
            possibleInd = relative_url.find('#')
            if possibleInd != -1:
                relative_url = relative_url[:possibleInd]

            relative_url = relative_url.lower()
                            
        if relative_url and (relative_url.startswith("/") or relative_url.startswith("../") ):
            path_levels = relative_url.count("../")

            parent_components = urlparse(parent_url)
            parent_domain = f"{parent_components.scheme}://{parent_components.netloc}"
            parent_path = parent_components.path

            if relative_url.startswith("/"):
                return urljoin(parent_domain, parent_path + "/" + relative_url)

            # Remove that many directory levels from base path
            for i in range(path_levels):
                parent_components = urlparse(parent_path)
                parent_path = parent_components.path.rsplit("/", 1)[0]
            
            parent_url = urljoin(parent_domain,parent_path)

            return urljoin(parent_url,relative_url.split("/../")[-1])
            
        elif relative_url and not ( relative_url.startswith('https') or relative_url.startswith('http') or relative_url.startswith('ftp') ):
            return urljoin(parent_url, relative_url)
             
        return relative_url
    
    def process(self, file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            file = json.load(f)
            self.doc_id += 1
            self.doc_id_to_url = file["url"]
            soup = BeautifulSoup(file["content"], "html.parser")
            links = [self.modify_if_relative(link.get('href'),file["url"]) for link in soup.find_all('a')]
            for link in links:
                if link is not None:          
                    self.G.add_edge(file["url"], link)  

            token_to_count = self.tokenize(soup.get_text(separator=' '))
            
            for token, count in token_to_count.items():
                self.inverted_index[token][file["url"]] = count
                self.pagerank[file["url"]] = 0
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
        
        inverted_index_string = json.dumps(self.inverted_index)
        compressed_inverted_index = gzip.compress(inverted_index_string.encode('utf-8'))

        with gzip.open(self.inverted_index_file, 'wb') as compressed_file:
            compressed_file.write(compressed_inverted_index)
        
        pagerank_string = json.dumps(self.pagerank)
        compressed_pagerank = gzip.compress(pagerank_string.encode('utf-8'))
            
        with gzip.open(self.pagerank_file, 'wb') as compressed_file:
            compressed_file.write(compressed_pagerank)
        

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

        tfidf_string = json.dumps(self.tfidf_table)
        compressed_tfidf = gzip.compress(tfidf_string.encode('utf-8'))
            
        with gzip.open(self.tfidf_file, 'wb') as compressed_file:
            compressed_file.write(compressed_tfidf)
            
    def start(self):
        start = time.perf_counter()

        count = 0
        for subdir, dirs, files in os.walk(self.dataset_path):
            for name in files:
                if name != ".DS_Store":
                    self.process(subdir + os.sep + name)
                    print(name, "Done")
            count = count + 1

        pagerank_scores = nx.pagerank(self.G)
        # with open("pagerankScores1.json", "w", encoding='utf-8') as f:
        #     json.dump(pagerank_scores, f)

        for url in self.pagerank:
            if url not in pagerank_scores:
                self.pagerank[url] = 0
                print('Not in pagerank', url)
            else:
                self.pagerank[url] = 50*pagerank_scores[url]

        self.dump()
        self.calculate_dump_tfidf()

        end = time.perf_counter()
        print("START - Total number of documents:{}".format(self.doc_id))
        print("START - Total Number of tokens: {}".format(len(self.inverted_index)))
        print("START - Program execution time: {} seconds".format(end - start))

if __name__ == '__main__':
    dataset_path = "DEV/"
    inverted_index_file = "compressed_inverted_index.json.gz"
    pagerank_file = "compressed_pagerank.json.gz"
    tfidf_file = "compressed_tfidf.json.gz"
    

    i = Inverted_Indexer(dataset_path, pagerank_file, inverted_index_file, tfidf_file)
    i.start()