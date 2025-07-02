from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Optional, Tuple
import uuid
import logging
from pathlib import Path
import hashlib
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Enhanced embedding manager with better chunking, metadata handling,
    and integration with the retrieval system
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2", 
                 persist_directory: str = "./chroma_db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the embedding manager
        
        Args:
            model_name: Name of the sentence transformer model
            persist_directory: Directory to persist ChromaDB
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collections = {}
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(exist_ok=True)
        
        logger.info(f"Initialized EmbeddingManager with model: {model_name}")
    
    def create_collection(self, department: str) -> chromadb.Collection:
        """
        Create or get a collection for a department
        
        Args:
            department: Department name
            
        Returns:
            ChromaDB collection
        """
        collection_name = f"{department}_docs"
        
        try:
            # Try to get existing collection first
            collection = self.client.get_collection(name=collection_name)
            logger.info(f"Found existing collection: {collection_name}")
        except:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"department": department, "created_at": str(datetime.now())}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        self.collections[department] = collection
        return collection
    
    def add_documents(self, department: str, documents: List[Dict]) -> int:
        """
        Add documents to a department's collection
        
        Args:
            department: Department name
            documents: List of document dictionaries
            
        Returns:
            Number of chunks added
        """
        if department not in self.collections:
            self.create_collection(department)
        
        collection = self.collections[department]
        total_chunks = 0
        
        for doc in documents:
            try:
                chunks_added = self._process_and_add_document(collection, doc, department)
                total_chunks += chunks_added
                logger.info(f"Added {chunks_added} chunks from {doc.get('source', 'Unknown')}")
            except Exception as e:
                logger.error(f"Error processing document {doc.get('source', 'Unknown')}: {e}")
        
        logger.info(f"Total chunks added to {department}: {total_chunks}")
        return total_chunks
    
    def _process_and_add_document(self, 
                                 collection: chromadb.Collection,
                                 document: Dict,
                                 department: str) -> int:
        """
        Process a single document and add its chunks to the collection
        
        Args:
            collection: ChromaDB collection
            document: Document dictionary
            department: Department name
            
        Returns:
            Number of chunks added
        """
        content = document.get("content", "")
        source = document.get("source", "unknown")
        metadata = document.get("metadata", {})
        
        # Generate document hash for deduplication
        doc_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Check if document already exists
        try:
            existing = collection.get(where={"doc_hash": doc_hash})
            if existing['ids']:
                logger.info(f"Document {source} already exists, skipping...")
                return 0
        except:
            pass  # Collection might be empty
        
        # Split document into chunks
        chunks = self._split_text_advanced(content)
        
        if not chunks:
            logger.warning(f"No chunks generated for document: {source}")
            return 0
        
        # Generate embeddings for all chunks
        embeddings = self.model.encode(chunks)
        
        # Prepare data for batch insertion
        ids = []
        metadatas = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{source}_{i}"
            ids.append(chunk_id)
            
            chunk_metadata = {
                "source": source,
                "department": department,
                "chunk_id": i,
                "doc_hash": doc_hash,
                "chunk_length": len(chunk),
                "created_at": str(datetime.now()),
                **metadata  # Include original metadata
            }
            metadatas.append(chunk_metadata)
        
        # Add to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(chunks)
    
    def _split_text_advanced(self, text: str) -> List[str]:
        """
        Advanced text splitting with sentence boundary awareness
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Simple sentence-aware splitting
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:  # Save current chunk if not empty
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    if len(chunks) > 0 and self.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is longer than chunk_size
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple rules
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting - can be enhanced with spaCy or NLTK
        import re
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def update_document(self, department: str, document: Dict) -> int:
        """
        Update an existing document (delete old chunks and add new ones)
        
        Args:
            department: Department name
            document: Updated document
            
        Returns:
            Number of new chunks added
        """
        source = document.get("source", "unknown")
        
        # Delete existing chunks for this document
        self.delete_document(department, source)
        
        # Add updated document
        return self.add_documents(department, [document])
    
    def delete_document(self, department: str, source: str) -> int:
        """
        Delete all chunks for a specific document
        
        Args:
            department: Department name
            source: Document source
            
        Returns:
            Number of chunks deleted
        """
        if department not in self.collections:
            return 0
        
        collection = self.collections[department]
        
        try:
            # Get all chunks for this document
            existing_chunks = collection.get(where={"source": source})
            
            if existing_chunks['ids']:
                collection.delete(ids=existing_chunks['ids'])
                logger.info(f"Deleted {len(existing_chunks['ids'])} chunks for {source}")
                return len(existing_chunks['ids'])
        except Exception as e:
            logger.error(f"Error deleting document {source}: {e}")
        
        return 0
    
    def get_collection_stats(self, department: str) -> Dict:
        """
        Get statistics about a department's collection
        
        Args:
            department: Department name
            
        Returns:
            Dictionary with collection statistics
        """
        if department not in self.collections:
            return {"error": "Collection not found"}
        
        collection = self.collections[department]
        
        try:
            count = collection.count()
            
            # Get sample data for analysis
            sample_size = min(100, count)
            if sample_size > 0:
                sample = collection.peek(limit=sample_size)
                
                sources = set()
                total_length = 0
                
                if sample['metadatas']:
                    for metadata in sample['metadatas']:
                        sources.add(metadata.get('source', 'Unknown'))
                        total_length += metadata.get('chunk_length', 0)
                
                avg_chunk_length = total_length / len(sample['metadatas']) if sample['metadatas'] else 0
                
                return {
                    "total_chunks": count,
                    "unique_sources": len(sources),
                    "sources": list(sources),
                    "average_chunk_length": avg_chunk_length,
                    "sample_size": sample_size
                }
            else:
                return {
                    "total_chunks": 0,
                    "unique_sources": 0,
                    "sources": [],
                    "average_chunk_length": 0,
                    "sample_size": 0
                }
        except Exception as e:
            logger.error(f"Error getting stats for {department}: {e}")
            return {"error": str(e)}
    
    def search_chunks(self, 
                     query: str, 
                     departments: List[str], 
                     n_results: int = 5,
                     filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for chunks across departments (legacy method for compatibility)
        
        Args:
            query: Search query
            departments: List of departments to search
            n_results: Number of results to return
            filter_metadata: Additional metadata filters
            
        Returns:
            List of search results
        """
        query_embedding = self.model.encode([query])[0]
        results = []
        
        for dept in departments:
            if dept not in self.collections:
                continue
            
            collection = self.collections[dept]
            
            try:
                where_filter = filter_metadata or {}
                
                search_results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results,
                    where=where_filter,
                    include=['documents', 'metadatas', 'distances']
                )
                
                if search_results['documents'] and search_results['documents'][0]:
                    for i in range(len(search_results['documents'][0])):
                        results.append({
                            "content": search_results['documents'][0][i],
                            "metadata": search_results['metadatas'][0][i],
                            "distance": search_results['distances'][0][i],
                            "relevance_score": 1 / (1 + search_results['distances'][0][i])
                        })
                        
            except Exception as e:
                logger.error(f"Error searching in department {dept}: {e}")
        
        # Sort results by relevance score (higher is better)
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Return top n_results
        return results[:n_results]
    
    def query_similar_chunks(self, 
                            query: str, 
                            department: str,
                            n_results: int = 5,
                            min_relevance: float = 0.3,
                            include_metadata: bool = True) -> List[Dict]:
        """
        Enhanced query method with better filtering and scoring
        
        Args:
            query: Search query
            department: Department to search in
            n_results: Number of results to return
            min_relevance: Minimum relevance threshold
            include_metadata: Whether to include full metadata
            
        Returns:
            List of relevant chunks with enhanced metadata
        """
        if department not in self.collections:
            self.create_collection(department)
            return []
        
        collection = self.collections[department]
        query_embedding = self.model.encode([query])
        
        try:
            search_results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(n_results * 2, 20),  # Get more results for filtering
                include=['documents', 'metadatas', 'distances']
            )
            
            results = []
            
            if search_results['documents'] and search_results['documents'][0]:
                for i in range(len(search_results['documents'][0])):
                    distance = search_results['distances'][0][i]
                    relevance_score = 1 / (1 + distance)
                    
                    # Filter by minimum relevance
                    if relevance_score >= min_relevance:
                        result = {
                            "content": search_results['documents'][0][i],
                            "relevance_score": relevance_score,
                            "distance": distance,
                            "department": department
                        }
                        
                        if include_metadata:
                            result["metadata"] = search_results['metadatas'][0][i]
                        else:
                            # Include only essential metadata
                            metadata = search_results['metadatas'][0][i]
                            result["source"] = metadata.get("source", "Unknown")
                            result["chunk_id"] = metadata.get("chunk_id", 0)
                        
                        results.append(result)
            
            # Sort by relevance and return top results
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error querying department {department}: {e}")
            return []
    
    def batch_search(self, 
                    queries: List[str], 
                    departments: List[str],
                    n_results_per_query: int = 3) -> Dict[str, List[Dict]]:
        """
        Perform batch search for multiple queries
        
        Args:
            queries: List of search queries
            departments: List of departments to search
            n_results_per_query: Number of results per query
            
        Returns:
            Dictionary mapping queries to their results
        """
        results = {}
        
        for query in queries:
            try:
                query_results = self.search_chunks(
                    query=query,
                    departments=departments,
                    n_results=n_results_per_query
                )
                results[query] = query_results
            except Exception as e:
                logger.error(f"Error in batch search for query '{query}': {e}")
                results[query] = []
        
        return results
    
    def get_all_departments(self) -> List[str]:
        """
        Get list of all available departments
        
        Returns:
            List of department names
        """
        try:
            collections = self.client.list_collections()
            departments = []
            
            for collection in collections:
                # Extract department name from collection name
                if collection.name.endswith('_docs'):
                    dept_name = collection.name[:-5]  # Remove '_docs' suffix
                    departments.append(dept_name)
            
            return sorted(departments)
        except Exception as e:
            logger.error(f"Error getting departments: {e}")
            return []
    
    def reindex_department(self, department: str) -> Dict:
        """
        Reindex all documents in a department (useful for model updates)
        
        Args:
            department: Department name
            
        Returns:
            Dictionary with reindexing results
        """
        if department not in self.collections:
            return {"error": "Department not found"}
        
        collection = self.collections[department]
        
        try:
            # Get all documents
            all_data = collection.get(include=['documents', 'metadatas'])
            
            if not all_data['documents']:
                return {"message": "No documents to reindex", "reindexed": 0}
            
            # Group chunks by source document
            documents_by_source = {}
            for i, (doc, metadata) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
                source = metadata.get('source', 'unknown')
                if source not in documents_by_source:
                    documents_by_source[source] = []
                documents_by_source[source].append(doc)
            
            # Delete old collection
            self.client.delete_collection(f"{department}_docs")
            
            # Recreate collection
            self.create_collection(department)
            
            # Re-add documents
            total_reindexed = 0
            for source, chunks in documents_by_source.items():
                # Reconstruct document content
                content = " ".join(chunks)
                document = {
                    "content": content,
                    "source": source,
                    "metadata": {"reindexed": True, "reindex_date": str(datetime.now())}
                }
                
                chunks_added = self.add_documents(department, [document])
                total_reindexed += chunks_added
            
            return {
                "message": f"Successfully reindexed department {department}",
                "reindexed": total_reindexed,
                "sources": len(documents_by_source)
            }
            
        except Exception as e:
            logger.error(f"Error reindexing department {department}: {e}")
            return {"error": str(e)}
    
    def export_department_data(self, department: str, output_path: str) -> bool:
        """
        Export department data to JSON file
        
        Args:
            department: Department name
            output_path: Path to output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        if department not in self.collections:
            logger.error(f"Department {department} not found")
            return False
        
        collection = self.collections[department]
        
        try:
            # Get all data
            all_data = collection.get(include=['documents', 'metadatas'])
            
            export_data = {
                "department": department,
                "export_date": str(datetime.now()),
                "total_chunks": len(all_data['documents']),
                "data": []
            }
            
            for i, (doc, metadata) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
                export_data["data"].append({
                    "chunk_id": i,
                    "content": doc,
                    "metadata": metadata
                })
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(all_data['documents'])} chunks to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting department {department}: {e}")
            return False
    
    def health_check(self) -> Dict:
        """
        Perform health check on the embedding manager
        
        Returns:
            Dictionary with health status
        """
        try:
            # Test model
            test_embedding = self.model.encode(["test"])
            
            # Test database connection
            collections = self.client.list_collections()
            
            # Get statistics
            total_chunks = 0
            departments = []
            
            for collection in collections:
                if collection.name.endswith('_docs'):
                    dept_name = collection.name[:-5]
                    departments.append(dept_name)
                    try:
                        count = collection.count()
                        total_chunks += count
                    except:
                        pass
            
            return {
                "status": "healthy",
                "model_status": "ok",
                "database_status": "ok",
                "total_departments": len(departments),
                "total_chunks": total_chunks,
                "departments": departments,
                "model_name": self.model._modules['0'].auto_model.name_or_path if hasattr(self.model, '_modules') else "unknown",
                "check_time": str(datetime.now())
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "check_time": str(datetime.now())
            }

# Utility functions for backward compatibility and convenience

def create_embedding_manager(model_name: str = "all-MiniLM-L6-v2", 
                           persist_directory: str = "./chroma_db") -> EmbeddingManager:
    """
    Factory function to create an embedding manager
    
    Args:
        model_name: Name of the sentence transformer model
        persist_directory: Directory to persist ChromaDB
        
    Returns:
        Initialized EmbeddingManager instance
    """
    return EmbeddingManager(model_name=model_name, persist_directory=persist_directory)

def quick_search(query: str, 
                departments: List[str], 
                embedding_manager: Optional[EmbeddingManager] = None) -> List[Dict]:
    """
    Quick search function for convenience
    
    Args:
        query: Search query
        departments: List of departments to search
        embedding_manager: Optional pre-initialized embedding manager
        
    Returns:
        Search results
    """
    if embedding_manager is None:
        embedding_manager = create_embedding_manager()
    
    return embedding_manager.search_chunks(query, departments)