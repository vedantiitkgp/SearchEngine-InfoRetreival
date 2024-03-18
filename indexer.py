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
from urllib.parse import urlparse, urljoin


class Inverted_Indexer:
    def __init__(self, dataset_path, inverted_index_file, tfidf_file, neighbors_index_file):
        self.dataset_path = dataset_path
        self.inverted_index_file = inverted_index_file
        self.neighbors_index_file = neighbors_index_file
        self.tfidf_file = tfidf_file
        self.ps = PorterStemmer()
        self.doc_id = 0
        self.inverted_index = defaultdict(dict)
        self.neighbors = defaultdict(dict)
        self.count_docs = defaultdict(int)
        self.tfidf_table = defaultdict(dict) 

        self.index_dir = os.path.join(os.getcwd(), 'index')
        self.neighbor_dir = os.path.join(os.getcwd(), 'neighbor')
        self.tfidf_dir = os.path.join(os.getcwd(), 'tfidf')

        # Create directories if they don't exist
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        if not os.path.exists(self.tfidf_dir):
            os.makedirs(self.tfidf_dir)
 

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
            soup = BeautifulSoup(file["content"], "html.parser")
            links = [self.modify_if_relative(link.get('href'),file["url"]) for link in soup.find_all('a')]
            for link in links:
                if link is not None:          
                    if file["url"] not in self.neighbors:
                        self.neighbors[file["url"]] = []

                    if link not in self.neighbors[file["url"]]:
                        self.neighbors[file["url"]].append(link)
                    

            token_to_count = self.tokenize(soup.get_text(separator=' '))
            
            for token, count in token_to_count.items():
                self.inverted_index[token][file["url"]] = count
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

    def dump(self, part):
        data = defaultdict(dict)
        
        inverted_index_string = json.dumps(self.inverted_index)
        compressed_inverted_index = gzip.compress(inverted_index_string.encode('utf-8'))
        
        print(os.path.join(self.index_dir,f"{self.inverted_index_file}_{part}.json.gz"))
        with gzip.open(os.path.join(self.index_dir,f"{self.inverted_index_file}_{part}.json.gz"), 'wb') as compressed_file:
            compressed_file.write(compressed_inverted_index)
        self.inverted_index = defaultdict(dict)

        neighbors_string = json.dumps(self.neighbors)
        compressed_neighbors_index = gzip.compress(neighbors_string.encode('utf-8'))

        print(os.path.join(self.neighbor_dir,f"{self.neighbors_index_file}_{part}.json.gz"))
        with gzip.open(os.path.join(self.neighbor_dir,f"{self.neighbors_index_file}_{part}.json.gz"), 'wb') as compressed_file:
            compressed_file.write(compressed_neighbors_index)
        self.neighbors = defaultdict(dict)
        

    """
    Function to Calculate TF-IDF Score
    """
    def calculate_dump_tfidf(self, part):
        for term , term_dict in self.inverted_index.items():
            count_urls = len(term_dict)
            for url, count in term_dict.items():
                term_freq = count / self.count_docs[url]
                inverse_doc_freq = self.doc_id / count_urls
                self.tfidf_table[term][url] = term_freq * inverse_doc_freq

        tfidf_string = json.dumps(self.tfidf_table)
        compressed_tfidf = gzip.compress(tfidf_string.encode('utf-8'))
            
        with gzip.open(os.path.join(self.tfidf_dir,f"{self.tfidf_file}_{part}.json.gz"), 'wb') as compressed_file:
            compressed_file.write(compressed_tfidf)
        self.tfidf_table = defaultdict(dict)
    
    def start(self):
        start = time.perf_counter()

        no_dir = 0
        for _, _, _ in os.walk(self.dataset_path):
            no_dir += 1

        print("No of directories :",no_dir)
        count = 0
        part = 1
        for subdir, dirs, files in os.walk(self.dataset_path):
            for name in files:
                if name != ".DS_Store":
                    self.process(subdir + os.sep + name)
                    print(name, "Done")

            if count == part * int(no_dir/4):
                print("-"*100)
                print(f"Part - {part}")
                print("-"*100)
                self.calculate_dump_tfidf(part)
                self.dump(part)
                part+=1
            count +=1
            print("Count --", count)
        
        self.calculate_dump_tfidf(part)
        self.dump(part)
        part+=1

        end = time.perf_counter()
        print("START - Total number of documents:{}".format(self.doc_id))
        print("START - Total Number of tokens: {}".format(len(self.inverted_index)))
        print("START - Program execution time: {} seconds".format(end - start))

if __name__ == '__main__':
    dataset_path = "DEV/"
    inverted_index_file = "compressed_inverted_index"
    neighbors_index_file = "neighbors_index"
    tfidf_file = "compressed_tfidf"
    

    i = Inverted_Indexer(dataset_path, inverted_index_file, tfidf_file, neighbors_index_file)
    i.start()