"""Conversation memory — stores and retrieves conversation history."""

import logging
import uuid
from datetime import datetime
from typing import Any

from app.models.schemas import Conversation, ConversationTurn

logger = logging.getLogger(__name__)


class ConversationStore:
    """In-memory conversation store. For production, swap to Redis or PostgreSQL."""

    def __init__(self):
        self._conversations: dict[str, Conversation] = {}

    def create(self) -> str:
        """Create a new conversation and return its ID."""
        conv_id = str(uuid.uuid4())
        self._conversations[conv_id] = Conversation(id=conv_id)
        logger.debug("Created conversation %s", conv_id)
        return conv_id

    def get(self, conv_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        return self._conversations.get(conv_id)

    def add_turn(self, conv_id: str, role: str, content: str, query_type: str | None = None) -> None:
        """Add a turn to an existing conversation."""
        conv = self._conversations.get(conv_id)
        if conv is None:
            # Auto-create if missing
            conv = Conversation(id=conv_id)
            self._conversations[conv_id] = conv

        conv.turns.append(ConversationTurn(
            role=role,
            content=content,
            query_type=query_type,
        ))
        conv.updated_at = datetime.utcnow()

    def get_history(self, conv_id: str, last_n: int = 6) -> list[dict[str, Any]]:
        """Get the last N turns as simple dicts for prompt injection."""
        conv = self._conversations.get(conv_id)
        if not conv:
            return []
        return [
            {"role": t.role, "content": t.content}
            for t in conv.turns[-last_n:]
        ]

    def summarize_context(self, conv_id: str) -> str:
        """Generate a brief context summary from conversation history."""
        conv = self._conversations.get(conv_id)
        if not conv or len(conv.turns) < 2:
            return ""

        # Collect entity references and topics from recent turns
        topics: set[str] = set()
        for turn in conv.turns[-4:]:
            # Simple extraction: quoted strings and capitalized words
            import re
            quoted = re.findall(r'"([^"]+)"', turn.content)
            topics.update(quoted)

        if topics:
            return f"Context from prior turns: {', '.join(sorted(topics)[:5])}."
        return ""

    def delete(self, conv_id: str) -> bool:
        """Delete a conversation."""
        if conv_id in self._conversations:
            del self._conversations[conv_id]
            return True
        return False
