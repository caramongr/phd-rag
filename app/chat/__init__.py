"""The simple persistent chat (RAG-015).

A chat is a saved conversation with optional attached working documents, answered by the
local model over its full context window. It is deliberately *not* a replacement for the
academic modes: nothing here changes RAG-009/RAG-011/RAG-012/RAG-013, which stay reachable
under Advanced. The chat reuses the project's existing pieces — the parsing layer for
attachments, the RAG-009 retrieval service when corpus material is explicitly asked for, and
the RAG-011 Ollama client for generation — instead of introducing a second stack.
"""
