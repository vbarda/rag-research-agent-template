"""State management for the index graph.

This module defines the state structures and reduction functions used in the
index graph. It includes definitions for document indexing.

Classes:
    IndexState: Represents the state for document indexing operations.

Functions:
    reduce_docs: Processes and reduces document inputs into a sequence of Documents.

The module also includes type definitions and utility functions to support
these state management operations.
"""

from dataclasses import dataclass
from typing import Annotated, Sequence

from langchain_core.documents import Document
from shared.state import reduce_docs

############################  Doc Indexing State  #############################


# The index state defines the simple IO for the single-node index graph
@dataclass(kw_only=True)
class IndexState:
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed and the retriever used for searching
    these documents.
    """

    docs: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can index."""