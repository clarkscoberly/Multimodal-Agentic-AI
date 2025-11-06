import os 
import json
from typing import List, Optional
import asyncio 
import warnings 
import numpy as np
warnings.filterwarnings('ignore')
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.readers import SimpleDirectoryReader
import chromadb
import logging

logger = logging.getLogger(__name__)

def split_data(raw_text: str = 'Error', chunk_size: int = 500, chunk_overlap: int = 20):
    # Split data into nodes
    documents = SimpleDirectoryReader("./data").load_data()
    try:
        document = Document(text=raw_text)
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if len(documents) > 0:
            nodes = splitter.get_nodes_from_documents([documents])
        else:
            nodes = splitter.get_nodes_from_documents([document])
        return nodes  
    except Exception as error:
        print(f'split_data() error: {error}')
        return []
    
class RetrieverBuilder:
    # Initialize embedding (can swap in main for different embedding model) text-embedding-3-small
    def __init__(self, top_k: int = 5, embedding_model: str = 'BAAI/bge-small-en-v1.5'):
        self.embeddings = HuggingFaceEmbedding(model_name=embedding_model)
        self.similarity_top_k = top_k

    def build_hybrid_retriever(self, nodes: List[Document]):
        # Hybrid retriever using BM25 + in-memory ChromaDb
        try:
            chroma_client = chromadb.Client()
            chroma_collection = chroma_client.create_collection('hybrid_collection')
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Embed and store in Chroma
            vector_index = VectorStoreIndex.build_index_from_nodes(
                nodes, 
                storage_context=storage_context, 
                embed_model=self.embeddings
            )
            vector_retriever = vector_index.as_retriever(similarity_top_k=self.similarity_top_k)
            logger.info("Vector retriever created successfully.")

            # BM25 retriever (lexical search)
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=self.similarity_top_k
            )
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

def retrieve_results(user_query: str):
    builder = RetrieverBuilder() 
    nodes = split_data()
    retriever = builder.build_hybrid_retriever(nodes) 
    results = retriever.retrieve(user_query)
    return results

retriever = RetrieverBuilder()