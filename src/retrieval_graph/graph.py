"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from typing import Literal, TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START

from retrieval_graph.configuration import Configuration
from retrieval_graph.prompts import ROUTER_SYSTEM_PROMPT, MORE_INFO_PROMPT, GENERAL_PROMPT, GENERATE_QUESTIONS_SYSTEM_PROMPT
from retrieval_graph.research_agent_graph import graph as researcher
from retrieval_graph.state import AgentState, InputState, Router
from retrieval_graph.utils import format_docs, load_chat_model


def analyze_and_route_query(state: AgentState, config: RunnableConfig) -> dict[str, list[str]]:
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}] + state.messages
    response = model.with_structured_output(Router).invoke(messages)
    return {"router": response}


def route_query(state: AgentState) -> Literal["generate_questions", "ask_for_more_info", "respond_to_general_query"]:
    _type = state.router["type"]
    if _type == "langchain":
        return "generate_questions"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")    


def ask_for_more_info(state: AgentState, *, config: RunnableConfig) -> dict[str, list[str]]:
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = MORE_INFO_PROMPT.format(logic=state.router["logic"])
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = model.invoke(messages)
    return {"messages": [response]}


def respond_to_general_query(state: AgentState, *, config: RunnableConfig) -> dict[str, list[str]]:
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    messages = [{"role": "system", "content": GENERAL_PROMPT.format(logic=state.router["logic"])}] + state.messages
    response = model.invoke(messages)
    return {"messages": [response]}


def generate_questions(state: AgentState, *, config: RunnableConfig) -> dict[str, list[str]]:
    class Plan(TypedDict):
        """Ask research questions."""
        questions: list[str]

    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Plan)
    messages = [{"role": "system", "content": GENERATE_QUESTIONS_SYSTEM_PROMPT}] + state.messages
    response = model.invoke(messages)
    return {"questions": response["questions"]}


def research_node(state: AgentState) -> dict[str, list[str]]:
    result = researcher.invoke({"question": state.questions[0]})
    return {
        "documents": result["documents"],
        "questions": state.questions[1:]
    }


def check_finished(state: AgentState) -> Literal["respond", "research_node"]:
    if len(state.questions or []) > 0:
        return "research_node"
    else:
        return "respond"


def respond(state: AgentState, *, config: RunnableConfig) -> dict[str, list[str]]:
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    context = format_docs(state.documents)
    prompt = configuration.response_system_prompt.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = model.invoke(messages)
    return {"messages": response}


builder = StateGraph(AgentState, input=InputState, output=InputState, config_schema=Configuration)
builder.add_node(analyze_and_route_query)
builder.add_node(ask_for_more_info)
builder.add_node(respond_to_general_query)
builder.add_node(research_node)
builder.add_node(generate_questions)
builder.add_node(respond)

builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)
builder.add_edge("generate_questions", "research_node")
builder.add_conditional_edges("research_node", check_finished)
graph = builder.compile()
graph.name = "RetrievalGraph"