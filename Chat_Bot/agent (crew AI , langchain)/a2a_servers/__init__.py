"""One A2A server per expert. Each expert is built with a different framework:

    disease_server    CrewAI      (port 9201)
    fertilize_server  LangChain   (port 9202)
    help_server       LangGraph   (port 9203)

Run them with: python -m agent.a2a_servers.run_agents
"""
