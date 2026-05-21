# Ragline Design Principles

Ragline is built around a config-driven philosophy: every aspect of the retrieval-augmented
generation pipeline — from chunking strategy to retrieval mode and answer generation — is
controlled through a single `RaglineConfig` dataclass or a YAML file. This approach keeps
application code lean and makes it easy to experiment with different configurations without
touching business logic.

The pipeline is structured as a directed graph where each node represents a distinct stage:
query transformation, retrieval, grading, processing, and generation. Handlers for each stage
are registered in global registries, allowing third-party plugins to extend behaviour by simply
providing a named handler and registering it before instantiating `RAG`.
