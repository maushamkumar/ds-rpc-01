# app/rag/retrieval.py
from typing import List, Dict, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Data class for storing retrieval results"""
    content: str
    source: str
    department: str
    score: float
    metadata: Dict
    chunk_id: int

class DocumentRetriever:
    """
    Advanced document retrieval system with role-based access control
    and intelligent ranking
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db",
                 similarity_threshold: float = 0.7):
        """
        Initialize the document retriever
        
        Args:
            model_name: Name of the sentence transformer model
            persist_directory: Directory to persist ChromaDB
            similarity_threshold: Minimum similarity score for results
        """
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.similarity_threshold = similarity_threshold
        self.collections = {}
        self._load_collections()
    
    def _load_collections(self):
        """Load existing collections from ChromaDB"""
        try:
            existing_collections = self.client.list_collections()
            for collection in existing_collections:
                dept_name = collection.name.replace("_docs", "")
                self.collections[dept_name] = collection
            logger.info(f"Loaded {len(self.collections)} collections")
        except Exception as e:
            logger.error(f"Error loading collections: {e}")
    
    def get_collection(self, department: str) -> Optional[chromadb.Collection]:
        """Get collection for a specific department"""
        collection_name = f"{department}_docs"
        
        if department not in self.collections:
            try:
                self.collections[department] = self.client.get_or_create_collection(
                    name=collection_name
                )
            except Exception as e:
                logger.error(f"Error creating collection for {department}: {e}")
                return None
        
        return self.collections[department]
    
    def hybrid_search(self, 
                     query: str, 
                     allowed_departments: List[str],
                     top_k: int = 10,
                     rerank: bool = True) -> List[RetrievalResult]:
        """
        Perform hybrid search across allowed departments with optional reranking
        
        Args:
            query: User query
            allowed_departments: List of departments user has access to
            top_k: Number of top results to return
            rerank: Whether to rerank results using query-document similarity
        
        Returns:
            List of RetrievalResult objects
        """
        all_results = []
        query_embedding = self.model.encode([query])[0]
        
        # Search across all allowed departments
        for department in allowed_departments:
            dept_results = self._search_department(
                query_embedding, department, top_k * 2  # Get more results for reranking
            )
            all_results.extend(dept_results)
        
        if not all_results:
            logger.warning(f"No results found for query: {query}")
            return []
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in all_results 
            if result.score >= self.similarity_threshold
        ]
        
        # Rerank if requested
        if rerank and filtered_results:
            filtered_results = self._rerank_results(query, filtered_results)
        
        # Sort by score and return top_k
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        return filtered_results[:top_k]
    
    def _search_department(self, 
                          query_embedding: np.ndarray,
                          department: str,
                          n_results: int) -> List[RetrievalResult]:
        """Search within a specific department"""
        collection = self.get_collection(department)
        if not collection:
            return []
        
        try:
            search_results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            results = []
            if search_results['documents'] and search_results['documents'][0]:
                for i in range(len(search_results['documents'][0])):
                    # Convert distance to similarity score (1 - normalized_distance)
                    distance = search_results['distances'][0][i]
                    similarity_score = max(0, 1 - distance)
                    
                    result = RetrievalResult(
                        content=search_results['documents'][0][i],
                        source=search_results['metadatas'][0][i].get('source', 'Unknown'),
                        department=department,
                        score=similarity_score,
                        metadata=search_results['metadatas'][0][i],
                        chunk_id=search_results['metadatas'][0][i].get('chunk_id', 0)
                    )
                    results.append(result)
            
            logger.info(f"Found {len(results)} results in {department}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching in {department}: {e}")
            return []
    
    def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder or advanced similarity metrics
        """
        try:
            # Get query embedding
            query_embedding = self.model.encode([query])
            
            # Get content embeddings for all results
            contents = [result.content for result in results]
            content_embeddings = self.model.encode(contents)
            
            # Calculate cosine similarity
            similarities = np.dot(content_embeddings, query_embedding.T).flatten()
            
            # Update scores with reranked similarities
            for i, result in enumerate(results):
                # Combine original score with reranked similarity
                result.score = (result.score * 0.6) + (similarities[i] * 0.4)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return results
    
    def semantic_search_with_filters(self,
                                   query: str,
                                   allowed_departments: List[str],
                                   source_filter: Optional[str] = None,
                                   date_filter: Optional[str] = None,
                                   top_k: int = 5) -> List[RetrievalResult]:
        """
        Perform semantic search with additional filters
        
        Args:
            query: User query
            allowed_departments: Departments user has access to
            source_filter: Filter by specific source file
            date_filter: Filter by date (if available in metadata)
            top_k: Number of results to return
        """
        results = self.hybrid_search(query, allowed_departments, top_k * 2)
        
        # Apply filters
        if source_filter:
            results = [r for r in results if source_filter.lower() in r.source.lower()]
        
        if date_filter:
            results = [r for r in results if r.metadata.get('date') == date_filter]
        
        return results[:top_k]
    
    def get_context_window(self,
                          result: RetrievalResult,
                          window_size: int = 2) -> List[RetrievalResult]:
        """
        Get surrounding context chunks for a given result
        
        Args:
            result: The main result
            window_size: Number of chunks before and after to include
        """
        collection = self.get_collection(result.department)
        if not collection:
            return [result]
        
        try:
            # Get chunks around the current chunk
            context_results = []
            source = result.source
            current_chunk = result.chunk_id
            
            # Search for related chunks from the same document
            for offset in range(-window_size, window_size + 1):
                target_chunk = current_chunk + offset
                if target_chunk < 0:
                    continue
                
                # Query for specific chunk
                chunk_results = collection.get(
                    ids=[f"{source}_{target_chunk}"],
                    include=['documents', 'metadatas']
                )
                
                if chunk_results['documents']:
                    context_result = RetrievalResult(
                        content=chunk_results['documents'][0],
                        source=source,
                        department=result.department,
                        score=result.score if offset == 0 else result.score * 0.8,
                        metadata=chunk_results['metadatas'][0],
                        chunk_id=target_chunk
                    )
                    context_results.append(context_result)
            
            return context_results if context_results else [result]
            
        except Exception as e:
            logger.error(f"Error getting context window: {e}")
            return [result]
    
    def get_department_stats(self, department: str) -> Dict:
        """Get statistics about a department's document collection"""
        collection = self.get_collection(department)
        if not collection:
            return {}
        
        try:
            count = collection.count()
            # Get sample of documents to analyze
            sample = collection.peek(limit=min(100, count))
            
            sources = set()
            if sample['metadatas']:
                sources = set(meta.get('source', 'Unknown') for meta in sample['metadatas'])
            
            return {
                'total_chunks': count,
                'unique_sources': len(sources),
                'sources': list(sources)
            }
        except Exception as e:
            logger.error(f"Error getting stats for {department}: {e}")
            return {}
    
    def search_similar_documents(self,
                               document_content: str,
                               allowed_departments: List[str],
                               top_k: int = 5,
                               exclude_source: Optional[str] = None) -> List[RetrievalResult]:
        """
        Find documents similar to a given document content
        
        Args:
            document_content: Content to find similar documents for
            allowed_departments: Departments to search in
            top_k: Number of similar documents to return
            exclude_source: Source to exclude from results
        """
        results = self.hybrid_search(document_content, allowed_departments, top_k * 2)
        
        # Exclude specified source
        if exclude_source:
            results = [r for r in results if r.source != exclude_source]
        
        return results[:top_k]

