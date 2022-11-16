import hashlib
import itertools
import math
from collections import defaultdict

import numpy 
import sympy
from scipy import sparse


class Shingling:

    def __init__(self, k_shingles=10):
        self.k_shingles = k_shingles

    def create_shingles(self, doc):

        # Creates shingles by iterating through the document and saving k_shingles characters as one shingle
        shingles = {doc[i:i + self.k_shingles] for i in range(len(doc) - self.k_shingles + 1)}
        return shingles

    def hash_shingles(self, doc):

        # By using hashlib we hash the shingles
        hashes = {hash(i) for i in self.create_shingles(doc)}
        return hashes

    def create_doc_shingles(self, doc_list):

        # Returns shingles from each document and their corresponding id in the overall set
        doc_shingles = [self.create_shingles(doc) for doc in doc_list]

        # Sort unique shingles to map them to integer
        unique_shingles = set(shingle for shingles in doc_shingles for shingle in shingles)
        shingle_idx = {shingle: idx for idx, shingle in enumerate(sorted(unique_shingles))}
        return doc_shingles, shingle_idx

    def create_char_matrix(self, doc_list):

        doc_shingles, shingle_idx = self.create_doc_shingles(doc_list)
        n_docs, n_shingles = len(doc_shingles), len(shingle_idx)
        
        # For each document we create a vector that contains an one-hot encoder for the occurence of a shingle
        vals = [(shingle_idx[shingle], doc_id, 1) for doc_id, shingles in enumerate(doc_shingles) for shingle in shingles]
        shingle_indices, doc_indices, data = zip(*vals)
        # Combine the vectors to a sparse matrix
        char_matrix = sparse.csr_matrix((data, (shingle_indices, doc_indices)), shape=(n_shingles, n_docs), dtype=numpy.bool_)
        return char_matrix

class Compare_sets:

    def jaccard_similarity(set_1, set_2):
        set_1, set_2 = set(set_1), set(set_2)
        # Compute the intersection and union of set_1 and set_2 and compute the jaccard_similarity as defined in the lecture
        similarity = len(set_1.intersection(set_2)) / len(set_1.union(set_2))
        return similarity

class Min_hashing:

    def __init__(self, n_signature=500):
        self.n_signature = n_signature

    def compute_signature_perm(self, char_matrix):
        n_signature, (n_shingles, n_docs) = self.n_signature, char_matrix.shape

        # Initialisation of the signature matrix of size n_signature x n_docs with zeros
        signature = numpy.zeros((n_signature, n_docs), dtype=numpy.int32)

        for idx in range(n_signature):
            # Permute the rows of the characteristic matrix
            rand_idxs = numpy.random.permutation(n_shingles)
            char_matrix_perm = char_matrix[rand_idxs, :]

            # we set h(C) as the number of the first (in the permuted order) row in which column C has 1
            signature[idx, :] = numpy.argmax(char_matrix_perm, axis=0)

        return signature

class Compare_signatures:
    
    def sig_similarity(signature, doc1, doc2):
        # the fraction of minhash functions (rows) in which the signatures agree
        return numpy.mean(signature[:, doc1] == signature[:, doc2])

class Lsh:

    def __init__(self, n_bands=100, sim_threshold=0.8):
        self.n_bands = n_bands
        self.sim_threshold = sim_threshold
    
    def find_candidates(self, signature):
        
        # set the size of the bands and the corresponding row size
        n_bands, (n_signature, n_docs) = self.n_bands, signature.shape
        rows_band = math.ceil(n_signature / n_bands)

        #initialize sets
        candidate_pairs = set()
        column_buckets = defaultdict(list)

        for band_idx in range(n_bands):
            #obtain the rows that are parts of each band
            band = signature[band_idx * rows_band:(band_idx+1)*rows_band]
            
            #iterate over each of the columns that belong to each band
            for doc_id, column in enumerate(band.T):
                # convert each vector to a tuple so it is hashable
                # store each doc within band with identical hashes
                column_buckets[tuple(column)].append(doc_id)

            for doc_ids in column_buckets.values():
               # create combinations from each bucket as candidate pairs
                pairwise_combos = itertools.combinations(doc_ids,2)
                candidate_pairs.update(pairwise_combos)
            
            # clear each bucket
            # columns with same vector will not hash to the same bucket
            column_buckets.clear()

        return candidate_pairs


    def find_similar(self, signature):
        candidate_pairs = self.find_candidates(signature)
        # initialize
        similar_documents = []

        # use predefined class function to compare lsh signatures of each candidate pair
        for candidate in candidate_pairs:
            doc_similarity = Compare_signatures.sig_similarity(signature, *candidate)
            if doc_similarity > self.sim_threshold:
                similar_documents.append(candidate)
        return similar_documents

    
