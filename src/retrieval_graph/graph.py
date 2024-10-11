"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text, load_chat_model

# Define the function that calls the model


# class SearchQuery(BaseModel):
#     """Search the indexed documents for a query."""

#     query: str


# async def generate_query(
#     state: State, *, config: RunnableConfig
# ) -> dict[str, list[str]]:
#     """Generate a search query based on the current state and configuration.

#     This function analyzes the messages in the state and generates an appropriate
#     search query. For the first message, it uses the user's input directly.
#     For subsequent messages, it uses a language model to generate a refined query.

#     Args:
#         state (State): The current state containing messages and other information.
#         config (RunnableConfig | None, optional): Configuration for the query generation process.

#     Returns:
#         dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

#     Behavior:
#         - If there's only one message (first user input), it uses that as the query.
#         - For subsequent messages, it uses a language model to generate a refined query.
#         - The function uses the configuration to set up the prompt and model for query generation.
#     """
#     messages = state.messages
#     if len(messages) == 1:
#         # It's the first user question. We will use the input directly to search.
#         human_input = get_message_text(messages[-1])
#         return {"queries": [human_input]}
#     else:
#         configuration = Configuration.from_runnable_config(config)
#         # Feel free to customize the prompt, model, and other logic!
#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", configuration.query_system_prompt),
#                 ("placeholder", "{messages}"),
#             ]
#         )
#         model = load_chat_model(configuration.query_model).with_structured_output(
#             SearchQuery
#         )

#         message_value = await prompt.ainvoke(
#             {
#                 "messages": state.messages,
#                 "queries": "\n- ".join(state.queries),
#                 "system_time": datetime.now(tz=timezone.utc).isoformat(),
#             },
#             config,
#         )
#         generated = cast(SearchQuery, await model.ainvoke(message_value, config))
#         return {
#             "queries": [generated.query],
#         }


# async def retrieve(
#     state: State, *, config: RunnableConfig
# ) -> dict[str, list[Document]]:
#     """Retrieve documents based on the latest query in the state.

#     This function takes the current state and configuration, uses the latest query
#     from the state to retrieve relevant documents using the retriever, and returns
#     the retrieved documents.

#     Args:
#         state (State): The current state containing queries and the retriever.
#         config (RunnableConfig | None, optional): Configuration for the retrieval process.

#     Returns:
#         dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
#         containing a list of retrieved Document objects.
#     """
#     with retrieval.make_retriever(config) as retriever:
#         response = await retriever.ainvoke(state.queries[-1], config)
#         return {"retrieved_docs": response}


# async def respond(
#     state: State, *, config: RunnableConfig
# ) -> dict[str, list[BaseMessage]]:
#     """Call the LLM powering our "agent"."""
#     configuration = Configuration.from_runnable_config(config)
#     # Feel free to customize the prompt, model, and other logic!
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", configuration.response_system_prompt),
#             ("placeholder", "{messages}"),
#         ]
#     )
#     model = load_chat_model(configuration.response_model)

#     retrieved_docs = format_docs(state.retrieved_docs)
#     message_value = await prompt.ainvoke(
#         {
#             "messages": state.messages,
#             "retrieved_docs": retrieved_docs,
#             "system_time": datetime.now(tz=timezone.utc).isoformat(),
#         },
#         config,
#     )
#     response = await model.ainvoke(message_value, config)
#     # We return a list, because this will get added to the existing list
#     return {"messages": [response]}


# # Define a new graph (It's just a pipe)


# builder = StateGraph(State, input=InputState, config_schema=Configuration)

# builder.add_node(generate_query)
# builder.add_node(retrieve)
# builder.add_node(respond)
# builder.add_edge("__start__", "generate_query")
# builder.add_edge("generate_query", "retrieve")
# builder.add_edge("retrieve", "respond")

# # Finally, we compile it!
# # This compiles it into a graph you can invoke and deploy.
# graph = builder.compile(
#     interrupt_before=[],  # if you want to update the state before calling the tools
#     interrupt_after=[],
# )
# graph.name = "RetrievalGraph"


# NEW AGENT


router_prompt = """You are a LangChain Developer advocate. Your job is help people using the LangChain platform answer any issues they are running into.

A user will come to you with an inquiry. Your first job is to classify what type of inquiry it is. The types of inquiries you should classify it as are:

## `more-info`
Classify a user inquiry as this if you need more information before you will be able to help them. Examples include:
- The user complains about an error but doesnt provide the error 
- The user says something isn't working but doesnt explain why/how it's not working

## `langchain`
Classify a user inquiry as this if it can be answered by looking up information related to the LangChain open source package. The LangChain open source package \
is a python SDK for working with LLMs. It integrates with various LLMs, databases and APIs.

## `general`
Classify a user inquiry as this if it is just a general question"""


from retrieval_graph.state import AgentState, Router
from retrieval_graph.research_agent_graph import graph as researcher
from typing import Literal, TypedDict


def route_at_start_node(state: AgentState, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    messages = [{"role": "system", "content": router_prompt}] + state.messages
    response = model.with_structured_output(Router).invoke(messages)
    return {"router": response}


def route_at_start(state: AgentState) -> Literal["generate_questions", "more_info", "general"]:
    _type = state.router["type"]
    if _type == "langchain":
        return "generate_questions"
    elif _type == "more-info":
        return "more_info"
    elif _type == "general":
        return "general"
    else:
        raise ValueError(f"Unknown router type {_type}")    


more_info_prompt = """You are a LangChain Developer advocate. Your job is help people using the LangChain platform answer any issues they are running into.

Your boss has determined that more information is needed before doing any research on behalf of the user. This was their logic:

<logic>
{logic}
</logic>

Respond to the user and try to get any more relevant information. Do not overwhelm them!! Be nice, and only ask them a single follow up question."""


def more_info(state: AgentState, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    messages = [{"role": "system", "content": more_info_prompt.format(logic=state.router["logic"])}] + state.messages
    response = model.invoke(messages)
    return {"messages": [response]}


general_prompt = """You are a LangChain Developer advocate. Your job is help people using the LangChain platform answer any issues they are running into.

Your boss has determined that the user is asking a general question, not one related to the LangGraph platform. This was their logic:

<logic>
{logic}
</logic>

Respond to the user. Politely decline to answer and tell them you can only answer questions about LangChain related topics, and that if their question is about LangChain they should clarify how it is.\
Be nice to them though - they are still a user!"""


def general(state: AgentState, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    response = model.invoke(messages)
    messages = [{"role": "system", "content": general_prompt.format(logic=state.router["logic"])}] + state.messages
    return {"messages": [response]}


generate_questions_prompt = """You are a LangChain expert, here to assist with any and all questions or issues with LangChain, LangGraph, LangSmith, or any related functionality. Users may come to you with questions or issues.

You are world class researcher. Based on the conversation below, generate a plan for how you will research the answer to their question. \
The plan should generally not be more than 3 steps long, it can be as short as one. The length of the plan depends on the question.

You have access to the following documentation sources:
- Conceptual docs
- Integration docs
- How-to guides

You do not need to specify where you want to research for all steps of the plan, but it's sometimes helpful.
"""

class Plan(TypedDict):
    """Ask research questions."""
    steps: list[str]


def generate_questions(state: AgentState, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    messages = [{"role": "system", "content": generate_questions_prompt}] + state.messages
    response = model.with_structured_output(Plan).invoke(messages)
    return {"steps": response["steps"]}


def research_node(state: AgentState, config: RunnableConfig):
    # TODO: fix this logic -- we need to add researcher directly as a node
    # to enable subgraph rendering in the UI
    result = researcher.invoke({"sub_question": state.steps[0]})
    return {
        "documents": result["documents"],
        "steps": state.steps[1:]
    }


def check_finished(state: AgentState) -> Literal['generate', 'research_node']:
    if len(state.steps or []) > 0:
        return "research_node"
    else:
        return "generate"


def remove_question(state: AgentState):
    return {"steps": state.steps[1:]}


RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer for the \
given question based solely on the provided search results (URL and content). \
Do NOT ramble, and adjust your response length based on the question. If they ask \
a question that can be answered in one sentence, do that. If 5 paragraphs of detail is needed, \
do that. You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the individual sentence or paragraph that reference them. \
Do not put them all at the end, but rather sprinkle them throughout. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end. DO NOT PUT THEM ALL THAT END, PUT THEM IN THE BULLET POINTS.

If there is nothing in the context relevant to the question at hand, do NOT make up an answer. \
Rather, tell them why you're unsure and ask for any additional information that may help you answer better.

Sometimes, what a user is asking may NOT be possible. Do NOT tell them that things are possible if you don't \
see evidence for it in the context below. If you don't see based in the information below that something is possible, \
do NOT say that it is - instead say that you're not sure.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

"""

def generate(state: AgentState, config: RunnableConfig):
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    context = format_docs(state.documents)
    prompt = RESPONSE_TEMPLATE.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = model.invoke(messages)
    return {"messages": response}

builder = StateGraph(AgentState, input=InputState, output=InputState, config_schema=Configuration)
builder.add_node(route_at_start_node)
builder.add_node(more_info)
builder.add_node(general)
builder.add_node(research_node)
builder.add_node(generate_questions)
builder.add_node(generate)

builder.add_edge(START, "route_at_start_node")
builder.add_conditional_edges("route_at_start_node", route_at_start)
builder.add_edge("generate_questions", "research_node")
builder.add_conditional_edges("research_node", check_finished)
builder.add_edge("generate", END)
builder.add_edge("general", END)
builder.add_edge("more_info", END)
graph = builder.compile()