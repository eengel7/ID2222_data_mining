import string
import unicodedata

from bs4 import BeautifulSoup


class Preprocessor:
    def __init__(self, lowercase=True, remove_punctuation=True, remove_accents=True, remove_numbers=True, normalize_whitespace=True):
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.remove_accents = remove_accents
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace

    # Removing accents 
    def strip_accents(self, doc):
        doc_nfkd = unicodedata.normalize('NFKD', doc)
        doc_ascii = doc_nfkd.encode('ASCII', 'ignore').decode('ascii')
        return doc_ascii

    #Removing numbers
    def delete_numbers(self, doc):
        list_without_numbers = []
        for i in doc:
            if i not in '0123456789':
                list_without_numbers.append(i)
            else:
                list_without_numbers.append(' ')
        string_without_numbers = ''.join(list_without_numbers)
        return string_without_numbers


    def preprocess_document(self, doc):
        if self.lowercase:
            doc = doc.lower()

        if self.remove_accents:
            doc = self.strip_accents(doc)

        if self.remove_numbers:
            doc = self.delete_numbers(doc)

        if self.remove_punctuation:
            doc = ''.join(char for char in doc if char not in string.punctuation)

        if self.normalize_whitespace:
            doc = ' '.join(doc.split())
        return doc

    def preprocess_documents(self, docs):
        return [self.preprocess_document(doc) for doc in docs]