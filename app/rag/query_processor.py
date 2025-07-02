
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from app.rag.retrieval import DocumentRetriever, RetrievalResult

@dataclass
class ProcessedQuery:
    """Processed query with extracted information"""
    original_query: str
    cleaned_query: str
    keywords: List[str]
    intent: str
    entities: Dict[str, str]
    filters: Dict[str, str]

class QueryProcessor:
    """
    Process user queries to extract intent, entities, and filters
    """
    
    def __init__(self):
        self.intent_patterns = {
            'search': ['find', 'search', 'look for', 'show me', 'get'],
            'compare': ['compare', 'difference', 'versus', 'vs', 'contrast'],
            'analyze': ['analyze', 'analysis', 'examine', 'evaluate'],
            'summarize': ['summarize', 'summary', 'overview', 'brief'],
            'explain': ['explain', 'what is', 'how does', 'why'],
            'list': ['list', 'enumerate', 'show all', 'give me all']
        }
        
        self.entity_patterns = {
            'date': r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b',
            'department': r'\b(finance|marketing|hr|engineering|general)\b',
            'document_type': r'\b(report|analysis|memo|policy|guide)\b'
        }
    
    def process_query(self, query: str) -> ProcessedQuery:
        """
        Process a user query and extract relevant information
        
        Args:
            query: Raw user query
            
        Returns:
            ProcessedQuery object with extracted information
        """
        cleaned_query = self._clean_query(query)
        keywords = self._extract_keywords(cleaned_query)
        intent = self._detect_intent(cleaned_query)
        entities = self._extract_entities(query)
        filters = self._extract_filters(query)
        
        return ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned_query,
            keywords=keywords,
            intent=intent,
            entities=entities,
            filters=filters
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove special characters (keep alphanumeric and spaces)
        query = re.sub(r'[^\w\s-]', ' ', query)
        
        return query
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query"""
        # Simple keyword extraction (can be enhanced with NLP libraries)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those'
        }
        
        words = query.split()
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return intent
        
        return 'search'  # Default intent
    
    def _extract_entities(self, query: str) -> Dict[str, str]:
        """Extract named entities from the query"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches[0] if len(matches) == 1 else matches
        
        return entities
    
    def _extract_filters(self, query: str) -> Dict[str, str]:
        """Extract filters from the query"""
        filters = {}
        
        # Date filters
        if 'recent' in query or 'latest' in query:
            filters['date_order'] = 'desc'
        elif 'oldest' in query or 'earliest' in query:
            filters['date_order'] = 'asc'
        
        # Source filters
        source_match = re.search(r'from\s+([a-zA-Z0-9_\-\.]+)', query)
        if source_match:
            filters['source'] = source_match.group(1)
        
        return filters

# Enhanced retrieval system integration
class EnhancedRetriever(DocumentRetriever):
    """
    Enhanced retriever that combines document retrieval with query processing
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_processor = QueryProcessor()
    
    def intelligent_search(self,
                          query: str,
                          allowed_departments: List[str],
                          user_role: str,
                          top_k: int = 5) -> Tuple[List[RetrievalResult], ProcessedQuery]:
        """
        Perform intelligent search with query processing
        
        Args:
            query: User query
            allowed_departments: Departments user has access to
            user_role: User's role for context-aware responses
            top_k: Number of results to return
        
        Returns:
            Tuple of (results, processed_query)
        """
        # Process the query
        processed_query = self.query_processor.process_query(query)
        
        # Adjust search based on intent
        if processed_query.intent == 'compare':
            top_k = max(top_k, 8)  # Get more results for comparison
        elif processed_query.intent == 'summarize':
            top_k = max(top_k, 10)  # Get more context for summarization
        
        # Use entity filters if available
        source_filter = processed_query.entities.get('document_type')
        date_filter = processed_query.filters.get('source')
        
        # Perform search
        if source_filter or date_filter:
            results = self.semantic_search_with_filters(
                processed_query.cleaned_query,
                allowed_departments,
                source_filter=source_filter,
                date_filter=date_filter,
                top_k=top_k
            )
        else:
            results = self.hybrid_search(
                processed_query.cleaned_query,
                allowed_departments,
                top_k=top_k
            )
        
        # Post-process results based on intent
        if processed_query.intent == 'compare' and len(results) > 1:
            # Group results by source for comparison
            results = self._group_for_comparison(results)
        
        return results, processed_query
    
    def _group_for_comparison(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Group results for comparison queries"""
        # Simple grouping by source - can be enhanced
        source_groups = {}
        for result in results:
            if result.source not in source_groups:
                source_groups[result.source] = []
            source_groups[result.source].append(result)
        
        # Take top result from each source
        grouped_results = []
        for source, group in source_groups.items():
            grouped_results.append(max(group, key=lambda x: x.score))
        
        return sorted(grouped_results, key=lambda x: x.score, reverse=True)