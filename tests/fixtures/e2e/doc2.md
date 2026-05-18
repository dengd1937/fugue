# Retrieval-Augmented Generation

RAG combines retrieval with generation to produce answers grounded in source documents.
The typical pipeline includes: query transformation, multi-route retrieval, document
grading, post-processing (RRF + reranking), and final answer generation.

LangGraph provides the underlying graph orchestration in Ragline's engine module, with
each node handling a specific stage of the pipeline.
