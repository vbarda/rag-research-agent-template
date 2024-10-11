from typing import Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from retrieval_graph.state import ResearcherState, QueryState
from retrieval_graph.configuration import Configuration
from retrieval_graph import retrieval

from retrieval_graph.utils import load_chat_model

generate_queries_prompt = """Generate 3 search queries to search for \
to answer the user's question. These search queries should be diverse in nature - do not generate \
repetitive ones."""


def generate_queries(state: ResearcherState, config: RunnableConfig):
    messages = [
       {"role": "system", "content": generate_queries_prompt},
       {"role": "human", "content": state.sub_question}
    ]

    class Response(TypedDict):
        queries: list[str]

    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    response = model.with_structured_output(Response).invoke(messages)
    return {"queries": response["queries"]}


def query(state: QueryState, config: RunnableConfig):
    with retrieval.make_retriever(config) as retriever:
        response = retriever.invoke(state["query"], config)
        return {"documents": response}


def do_queries(state: ResearcherState) -> Literal['query']:
    return [Send("query", {"query": q}) for q in state.queries]


builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(query)
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges("generate_queries", do_queries)
builder.add_edge("query", END)
graph = builder.compile()