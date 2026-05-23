# Typical Ragline Usage

The most common way to use Ragline is through the `RAG` context manager. First, create a
`RaglineConfig` (or load one from YAML with `RAG.from_yaml`), then open a `with RAG(cfg) as rag:`
block. Inside the block, call `rag.ingest(path)` to index a file or directory — Ragline parses,
chunks, embeds, and stores documents in a ChromaDB collection automatically.

Once documents are ingested, call `rag.query("your question")` to run the full retrieval and
generation pipeline. The returned `QueryResult` contains the answer string, ranked source
documents, a relevance grade score, and metadata about retrieval rounds. Closing the context
manager releases all provider resources cleanly, ensuring no file handles or network connections
are left open.
