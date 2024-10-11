from typing import Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from retrieval_graph.state import ResearcherState, QueryState
from retrieval_graph.configuration import Configuration
from retrieval_graph.utils import load_chat_model
from retrieval_graph.prompts import GENERATE_QUERIES_PROMPT
from retrieval_graph import retrieval


def generate_queries(state: ResearcherState, *, config: RunnableConfig) -> dict[str, list[str]]:
    class Response(TypedDict):
        queries: list[str]

    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Response)
    messages = [
       {"role": "system", "content": GENERATE_QUERIES_PROMPT},
       {"role": "human", "content": state.question}
    ]
    response = model.invoke(messages)
    return {"queries": response["queries"]}


def retrieve(state: QueryState, *, config: RunnableConfig) -> dict[str, list[Document]]:
    with retrieval.make_retriever(config) as retriever:
        response = retriever.invoke(state["query"], config)
        return {"documents": response}


def retrieve_in_parallel(state: ResearcherState) -> Literal["retrieve"]:
    return [Send("retrieve", {"query": query}) for query in state.queries]


builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(retrieve)
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges("generate_queries", retrieve_in_parallel)
builder.add_edge("retrieve", END)
graph = builder.compile()
graph.name = "Reseacher"