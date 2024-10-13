from retrieval_graph.configuration import AgentConfiguration


def test_configuration_from_none() -> None:
    AgentConfiguration.from_runnable_config({"user_id": "foo"})
