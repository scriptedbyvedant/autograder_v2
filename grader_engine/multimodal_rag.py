
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
import streamlit as st

# Check if the required libraries are installed
try: import faiss
except ImportError: raise ImportError("FAISS not installed. `pip install faiss-cpu`")

try: from sentence_transformers import SentenceTransformer
except ImportError: raise ImportError("SentenceTransformers not installed. `pip install sentence-transformers`")

class MultimodalVectorStore:
    """
    A vector store for managing text embeddings for Retrieval-Augmented Generation.
    NOTE: In this open-source configuration, it only handles text.
    """
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        try:
            # Use a robust, text-only open-source model
            self.embedding_model = SentenceTransformer(embedding_model_name)
        except Exception as e:
            raise IOError(f"Failed to load SentenceTransformer model '{embedding_model_name}'. "
                          f"This might be due to a network issue if downloading for the first time. Original error: {e}")

        embedding_dim_val = self.embedding_model.get_sentence_embedding_dimension()
        if embedding_dim_val is None:
            raise ValueError(f"Could not get embedding dimension for model '{embedding_model_name}'.")
        self.embedding_dim = int(embedding_dim_val)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.items: List[Dict[str, Any]] = []
        self.by_q: Dict[str, List[int]] = defaultdict(list)

    def add(self, doc_id: str, content: Any, content_type: str, meta: Dict[str, Any]):
        # This open-source version only processes text content.
        if content_type != 'text':
            return

        try:
            embedding = self.embedding_model.encode([content])[0]
            np_embedding = np.array([embedding], dtype=np.float32)
            self.index.add(np_embedding)
            item_index = len(self.items)
            self.items.append({"id": doc_id, "content": content, "content_type": content_type, "meta": meta})
            if "q_id" in meta: self.by_q[meta["q_id"]].append(item_index)
        except Exception as e:
            st.warning(f"Could not create embedding for doc '{doc_id}': {e}")

    def search(self, query: str, k: int = 3, search_by_q_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if search_by_q_id and search_by_q_id in self.by_q:
            item_indices = self.by_q[search_by_q_id]
            return [self.items[i] for i in item_indices[:k]] if item_indices else []
        
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            np_query_embedding = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(np_query_embedding, k)
            return [self.items[i] for i in indices[0] if i < len(self.items)]
        except Exception as e:
            st.error(f"Failed to perform vector search: {e}")
            return []

def retrieve_multimodal_context(q_id: str, question: str, k: int = 3) -> Dict[str, Any]:
    """
    Retrieves text context using the Vector Store from the session state.
    """
    if 'multimodal_vs' not in st.session_state or st.session_state['multimodal_vs'] is None:
        # Attempt to initialize it if it's missing
        try:
            st.session_state['multimodal_vs'] = MultimodalVectorStore()
        except (IOError, ValueError) as e:
            st.warning(f"Could not initialize RAG Vector Store: {e}. Context feature will be disabled.")
            st.session_state['multimodal_vs'] = None
            return {"context": []}
    
    vs_instance = st.session_state['multimodal_vs']
    if vs_instance is None:
        return {"context": []}
    
    hits = vs_instance.search(question, k=k, search_by_q_id=q_id)
    if not hits:
        hits = vs_instance.search(question, k=k)
        
    return {"context": hits}

