import string
import unicodedata

class Preprocessor:
    def __init__(self, lowercase=True, remove_punctuation=True, remove_accents=True, normalize_whitespace=True):
        self.lowercase = lowercase
        self.remove_accents = remove_accents
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace

    def strip_accents(self, doc):
        doc_nfkd = unicodedata.normalize('NFKD', doc)
        doc_ascii = doc_nfkd.encode('ASCII', 'ignore').decode('ascii')

        return doc_ascii

    def preprocess_document(self, doc):
        if self.lowercase:
            doc = doc.lower()

        if self.remove_accents:
            doc = self.strip_accents(doc)

        if self.remove_punctuation:
            doc = ''.join(char for char in doc if char not in string.punctuation)

        if self.normalize_whitespace:
            doc = ' '.join(doc.split())

        return doc

    def preprocess_documents(self, docs):
        return [self.preprocess_document(doc) for doc in docs]