"""
agents - the four AI agents and the LangGraph workflow that connects them.

    planner.py    Agent 1  decides WHAT to search for (and whether to search at all)
    retriever.py  Agent 2  searches the vector DB, drops weak / unsafe chunks
    reasoner.py   Agent 3  asks the LLM for an answer that cites its sources
    validator.py  Agent 4  checks the answer; can send it back to the Reasoner once
    graph.py      LangGraph StateGraph that runs them in order with a retry loop
"""
