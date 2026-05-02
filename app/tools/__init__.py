"""Tools package — lazy registration imports."""

# ToolRegistry is always needed
from .base import BaseTool, ToolSpec, ToolRegistry

def register_vector_tools(*a, **kw):
    from .vector_search import register_vector_tools as _fn
    return _fn(*a, **kw)

def register_keyword_tool(*a, **kw):
    from .keyword_search import register_keyword_tool as _fn
    return _fn(*a, **kw)

def register_sql_tool(*a, **kw):
    from .sql_query import register_sql_tool as _fn
    return _fn(*a, **kw)

def register_incident_tool(*a, **kw):
    from .incident_query import register_incident_tool as _fn
    return _fn(*a, **kw)

def register_entity_tool(*a, **kw):
    from .entity_lookup import register_entity_tool as _fn
    return _fn(*a, **kw)

__all__ = [
    "BaseTool", "ToolSpec", "ToolRegistry",
    "register_vector_tools", "register_keyword_tool",
    "register_sql_tool", "register_incident_tool", "register_entity_tool",
]
