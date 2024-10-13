from typing import Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.researcher_graph.state import ResearcherState, QueryState
from shared import retrieval
from shared.utils import load_chat_model

GENERATE_QUERIES_PROMPT = """\
Generate 3 search queries to search for to answer the user's question. \
These search queries should be diverse in nature - do not generate \
repetitive ones."""


def generate_queries(state: ResearcherState, *, config: RunnableConfig) -> dict[str, list[str]]:
    class Response(TypedDict):
        queries: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Response)
    messages = [
       {"role": "system", "content": GENERATE_QUERIES_PROMPT},
       {"role": "human", "content": state.question}
    ]
    response = model.invoke(messages)
    return {"queries": response["queries"]}


def retrieve_documents(state: QueryState, *, config: RunnableConfig) -> dict[str, list[Document]]:
    with retrieval.make_retriever(config) as retriever:
        response = retriever.invoke(state["query"], config)
        return {"documents": response}


def retrieve_in_parallel(state: ResearcherState) -> Literal["retrieve_documents"]:
    return [Send("retrieve_documents", {"query": query}) for query in state.queries]


builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(retrieve_documents)
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges("generate_queries", retrieve_in_parallel)
builder.add_edge("retrieve_documents", END)
graph = builder.compile()
graph.name = "Reseacher"