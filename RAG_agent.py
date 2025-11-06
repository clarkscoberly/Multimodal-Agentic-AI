import os
import json
from typing import List, Optional
import asyncio
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Sample data for the lab - AI/ML focused documents
SAMPLE_DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    "Computer vision allows machines to interpret and understand visual information from the world.",
    "Reinforcement learning is a type of machine learning where agents learn to make decisions through rewards and penalties.",
    "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
    "Unsupervised learning finds hidden patterns in data without labeled examples.",
    "Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks.",
    "Generative AI can create new content including text, images, code, and more.",
    "Large language models are trained on vast amounts of text data to understand and generate human-like text."
]

# Consistent query examples used throughout the lab
DEMO_QUERIES = {
    "basic": "What is machine learning?",
    "technical": "neural networks deep learning", 
    "learning_types": "different types of learning",
    "advanced": "How do neural networks work in deep learning?",
    "applications": "What are the applications of AI?",
    "comprehensive": "What are the main approaches to machine learning?",
    "specific": "supervised learning techniques"
}

from typing import List, Optional
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import chromadb
import logging

logger = logging.getLogger(__name__)

def split_data(raw_text: str, chunk_size: int = 500, chunk_overlap: int = 20):
    """Split raw text into nodes."""
    try:
        document = Document(text=raw_text)
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents([document])
        return nodes
    except Exception as error:
        print(f'Error in split_data: {error}')
        return []

class RetrieverBuilder:
    def __init__(self):
        """Initialize with OpenAI embeddings (can swap to HuggingFace if preferred)."""
        # self.embeddings = OpenAIEmbedding(model="text-embedding-3-small")
        self.embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.similarity_top_k = 5





"""
Create an alias for every column that should have one.
Create an alias for table

CategoryDim name should always be CategoryName

"""
    def build_hybrid_retriever(self, nodes: List[Document]):
        """Build a hybrid retriever using BM25 + in-memory Chroma."""
        try:
            # --- In-memory Chroma ---
            chroma_client = chromadb.Client()
            chroma_collection = chroma_client.create_collection("hybrid_collection")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Vector index (embeds + stores nodes in Chroma)
            vector_index = VectorStoreIndex.from_nodes(
                nodes, storage_context=storage_context, embed_model=self.embeddings
            )
            vector_retriever = vector_index.as_retriever(similarity_top_k=self.similarity_top_k)
            logger.info("Vector retriever created successfully.")

            # BM25 retriever (lexical search)
            bm25_retriever = BM25Retriever.from_documents([n.to_document() for n in nodes], similarity_top_k=self.similarity_top_k)
            logger.info("BM25 retriever created successfully.")

            # Hybrid retriever
            hybrid_retriever = QueryFusionRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                similarity_top_k=5,
                num_queries=1,
                mode="reciprocal_rerank",
            )
            logger.info("Hybrid retriever created successfully (in-memory).")

            return hybrid_retriever

        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise


# appropriate call to this class
"""
builder = RetrieverBuilder() # Initalize
retriever = builder.build_hybrid_retriever(docs) # Create vector embeddings for data
results = retriever.retrieve(user_query) # Compare embeddings to user_query to find most relevant documents. (These are in 500 char chunks)
"""
builder = RetrieverBuilder() 
nodes = split_data(SAMPLE_DOCUMENTS)
retriever = builder.build_hybrid_retriever(nodes) 
results = retriever.retrieve('Neural networks and deep learning')

print(results)