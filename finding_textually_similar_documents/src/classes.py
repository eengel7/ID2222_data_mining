import hashlib
import itertools
import math
import numpy
from collections import defaultdict
import sympy
from scipy import sparse


class shingling:

    def __init__(self, k_shingles=10):
        self.k_shingles = k_shingles

    def create_shingles(self, doc):
        shingles = {doc[i:i + self.k_shingles] for i in range(len(doc) - self.k_shingles + 1)}
        return shingles

    def hash_shingles(self, doc):
        hashes = {hash(i) for i in self.create_shingles(doc)}
        return hashes

    def create_doc_shingles(self, doc_list):
        doc_shingles = [self.create_shingles(doc) for doc in doc_list]
        unique_shingles = set(shingle for shingles in doc_shingles for shingle in shingles)
        shingle_idx = {shingle: idx for idx, shingle in enumerate(sorted(unique_shingles))}
        return doc_shingles, shingle_idx

    def create_char_matrix(self, doc_list):
        doc_shingles, shingle_idx = self.create_doc_shingles(doc_list)
        n_docs, n_shingles = len(doc_shingles), len(shingle_idx)

        vals = [(shingle_idx[shingle], doc_id, 1) for doc_id, shingles in enumerate(doc_shingles) for shingle in shingles]
        shingle_indices, doc_indices, data = zip(*vals)

        char_matrix = sparse.csr_matrix((data, (shingle_indices, doc_indices)), shape=(n_shingles, n_docs), dtype=numpy.bool_)
        return char_matrix

class compare_sets:

    def jaccard_similarity(set_1, set_2):
        set_1, set_2 = set(set_1), set(set_2)
        similarity = len(set_1.intersection(set_2)) / len(set_1.union(set_2))
        return similarity

class min_hashing:

    def __init__(self, n_signature=500):
        self.n_signature = n_signature
    
    def compute_signature_hash(self, char_matrix):
        n_signature, (n_shingles, n_docs) = self.n_signature, char_matrix.shape
        signature = numpy.full((n_signature, n_docs), numpy.inf)
        p = sympy.nextprime(n_shingles)
        a = 2 * numpy.random.randint(0, p//2, n_signature) + 1
        b = numpy.random.randint(0, p, n_signature)

        for row_idx, doc_ids in enumerate(char_matrix.tolil().rows):
            hashes = self.compute_universal_hash(row_idx, a, b, p, n_shingles)
            for doc_id in doc_ids:
                signature[:, doc_id] = numpy.where(hashes < signature[:, doc_id], hashes, signature[:, doc_id])
        return signature

    def compute_universal_hash(self, x, a, b, p, m):
        return((a*x + b) % p) % m

class compare_signatures:

    def sig_similarity(signature, doc1, doc2):
        return numpy.mean(signature[:, doc1] == signature[:, doc2])

class lsh:

    def __init__(self, n_bands=100, sim_threshold=0.8):
        self.n_bands = n_bands
        self.sim_threshold = sim_threshold

    def find_similar(self, signature):
        candidate_pairs = self.find_candidates(signature)
        similar_documents = []

        for candidate in candidate_pairs:
            doc_similarity = compare_signatures.sig_similarity(signature, *candidate)
            if doc_similarity > self.sim_threshold:
                similar_documents.append(candidate)
        return similar_documents

    def find_candidates(self, signature):
        n_bands, (n_signature, n_docs) = self.n_bands, signature.shape
        rows_band = math.ceil(n_signature / n_bands)

        candidate_pairs = set()
        column_buckets = defaultdict(list)

        for band_idx in range(n_bands):
            band = signature[band_idx * rows_band:(band_idx+1)*rows_band]
            for doc_id, column in enumerate(band.T):
                column_buckets[tuple(column)].append(doc_id)

            for doc_ids in column_buckets.values():
                pairwise_combos = itertools.combinations(doc_ids,2)
                candidate_pairs.update(pairwise_combos)
            
            column_buckets.clear()

        return candidate_pairs

