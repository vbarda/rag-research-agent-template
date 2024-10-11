"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

from typing import Optional
import json

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from retrieval_graph import retrieval
from retrieval_graph.configuration import IndexConfiguration
from retrieval_graph.state import IndexState, reduce_docs


async def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    If docs are not provided in the state, they will be loaded
    from the configuration.docs_file JSON file.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.r
    """
    if not config:
        raise ValueError("Configuration required to run index_docs.")

    configuration = IndexConfiguration.from_runnable_config(config)
    docs = state.docs
    if not docs:
        with open(configuration.docs_file) as f:
            serialized_docs = json.load(f)
            docs = reduce_docs([], serialized_docs)

    with retrieval.make_retriever(config) as retriever:
        await retriever.aadd_documents(docs)

    return {"docs": "delete"}

# Define a new graph

builder = StateGraph(IndexState, config_schema=IndexConfiguration)
builder.add_node(index_docs)
builder.add_edge("__start__", "index_docs")
# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile()
graph.name = "IndexGraph"
