# BM25 and Hybrid Search

BM25 is a classic lexical retrieval algorithm that ranks documents by term frequency
and inverse document frequency. It complements semantic vector search by capturing
exact keyword matches.

Hybrid retrieval combines BM25 and vector search results using Reciprocal Rank Fusion (RRF).
RRF scores each document by the sum of 1/(60 + rank) across source rankings, with
optional source weights.

In Ragline, BM25 is built-in via rank_bm25 with in-memory rebuild on startup.
