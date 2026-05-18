# Vector Databases

Vector databases like Chroma store high-dimensional embeddings for similarity search.
They enable semantic retrieval in RAG systems by indexing documents as numerical vectors.

Popular vector databases include Chroma, Qdrant, Weaviate, and Pinecone. Each offers
different trade-offs between performance, scalability, and ease of deployment.

In Ragline, Chroma is the default vector store, accessed via PersistentClient for
local file-based persistence.
