"""Agents package."""

def QueryClassifier():
    from .classifier import QueryClassifier as _cls
    return _cls()

def QueryType():
    from .classifier import QueryType as _cls
    return _cls

def ReasoningEngine():
    from .reasoning import ReasoningEngine as _cls
    return _cls()

def AgentOrchestrator():
    from .orchestrator import AgentOrchestrator as _cls
    return _cls()

def QueryPlanner():
    from .planner import QueryPlanner as _cls
    return _cls()

__all__ = ["QueryClassifier", "QueryType", "ReasoningEngine", "AgentOrchestrator", "QueryPlanner"]
