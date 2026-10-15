"""
vectordb - embeddings + ChromaDB vector store.

    embeddings.py  text -> 384-number vectors (all-MiniLM-L6-v2, offline fallback)
    store.py       add / search / delete chunks in ChromaDB (cosine similarity)
    data/          created at runtime: ChromaDB files, uploaded copies, question history
"""
