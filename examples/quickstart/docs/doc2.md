# Ragline Plugin Mechanism

Ragline exposes seven named registries — `transform_registry`, `retriever_registry`,
`processor_registry`, `grader_registry`, `generator_registry`, `parser_registry`, and
`chunker_registry` — that map string keys to handler callables. Built-in handlers such as
`rewrite`, `vector`, `rrf`, `score`, `basic`, `auto`, and `recursive` are registered
automatically when `RAG` is instantiated.

Third-party packages can add custom handlers by calling `registry.register("my-handler", fn)`
before creating a `RAG` instance. Ragline also supports automatic plugin discovery via the
`ragline.plugins` entry-point group: any installed package that declares this entry point will
have its `register()` function called during `RAG.__init__`, enabling seamless extension without
modifying library source code.
