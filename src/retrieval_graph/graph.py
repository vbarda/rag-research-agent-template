"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from typing import Literal, TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START

from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.researcher_graph.graph import graph as researcher
from retrieval_graph.state import AgentState, InputState, Router
from shared.utils import format_docs, load_chat_model


def analyze_and_route_query(state: AgentState, config: RunnableConfig) -> dict[str, list[str]]:
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    messages = [{"role": "system", "content": configuration.router_system_prompt}] + state.messages
    response = model.with_structured_output(Router).invoke(messages)
    return {"router": response}


def route_query(state: AgentState) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    _type = state.router["type"]
    if _type == "langchain":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")    


def ask_for_more_info(state: AgentState, *, config: RunnableConfig) -> dict[str, list[str]]:
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.more_info_system_prompt.format(logic=state.router["logic"])
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = model.invoke(messages)
    return {"messages": [response]}


def respond_to_general_query(state: AgentState, *, config: RunnableConfig) -> dict[str, list[str]]:
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    messages = [{"role": "system", "content": configuration.general_system_prompt.format(logic=state.router["logic"])}] + state.messages
    response = model.invoke(messages)
    return {"messages": [response]}


def create_research_plan(state: AgentState, *, config: RunnableConfig) -> dict[str, list[str]]:
    class Plan(TypedDict):
        """Generate research plan."""
        steps: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Plan)
    messages = [{"role": "system", "content": configuration.research_plan_system_prompt}] + state.messages
    response = model.invoke(messages)
    return {"steps": response["steps"]}


def conduct_research(state: AgentState) -> dict[str, list[str]]:
    result = researcher.invoke({"question": state.steps[0]})
    return {
        "documents": result["documents"],
        "steps": state.steps[1:]
    }


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"


def respond(state: AgentState, *, config: RunnableConfig) -> dict[str, list[str]]:
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    context = format_docs(state.documents)
    prompt = configuration.response_system_prompt.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = model.invoke(messages)
    return {"messages": response}


builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node(analyze_and_route_query)
builder.add_node(ask_for_more_info)
builder.add_node(respond_to_general_query)
builder.add_node(conduct_research)
builder.add_node(create_research_plan)
builder.add_node(respond)

builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
graph = builder.compile()
graph.name = "RetrievalGraph"