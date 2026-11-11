"""Shared bounded context policy for classifier and extractor prompts."""

# Four complete conversational exchanges: user/assistant × 4.
CONVERSATION_CONTEXT_TURNS = 8

__all__ = ["CONVERSATION_CONTEXT_TURNS"]
