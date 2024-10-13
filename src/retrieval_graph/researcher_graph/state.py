from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.documents import Document

from shared.state import reduce_docs


@dataclass(kw_only=True)
class QueryState:
    query: str


@dataclass(kw_only=True)
class ResearcherState:
    question: str
    """A question provided by the retriever agent."""
    queries: list[str] = field(default_factory=list)
    """A list of search queries based on the question that the researcher generates."""
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""

    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.