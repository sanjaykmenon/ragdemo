# RAG Best Practices: Levels 1-4 Implementation Guide

Based on Jason Liu's RAG complexity framework, this guide covers the foundational to intermediate levels of RAG implementation with practical Python examples for financial document processing.

## Table of Contents
1. [Level 1: The Basics](#level-1-the-basics)
2. [Level 2: More Structured Processing](#level-2-more-structured-processing)
3. [Level 3: Observability](#level-3-observability)
4. [Level 4: Evaluation](#level-4-evaluation)
5. [Implementation Roadmap](#implementation-roadmap)

## Level 1: The Basics

### Core Components
- Recursive file system traversal
- Text chunking with generators
- Batch embedding requests
- Vector storage
- Basic query-response pipeline

### Document Processing Pipeline

**File System Traversal**
```python
import os
from pathlib import Path
from typing import Generator, List

def traverse_documents(root_path: str) -> Generator[str, None, None]:
    """Recursively traverse file system to generate text."""
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(('.pdf', '.txt', '.docx', '.md')):
                file_path = os.path.join(root, file)
                yield extract_text_from_file(file_path)

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various file formats."""
    if file_path.endswith('.pdf'):
        # Use PyPDF2 or pdfplumber
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            return '\n'.join([page.extract_text() for page in pdf.pages])
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    # Add more file type handlers as needed
```

**Text Chunking with Generators**
```python
def chunk_text_generator(text: str, chunk_size: int = 1000, overlap: int = 200) -> Generator[str, None, None]:
    """Memory-efficient text chunking using generators."""
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Don't break in the middle of sentences for financial documents
        if end < len(text) and not text[end].isspace():
            # Find the last sentence boundary
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            boundary = max(last_period, last_newline)
            
            if boundary > start + chunk_size // 2:  # Ensure minimum chunk size
                end = start + boundary + 1
                chunk = text[start:end]
        
        yield chunk.strip()
        start = end - overlap

def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters for English)."""
    return len(text) // 4
```

**Batch Embedding Requests**
```python
import asyncio
from typing import List, AsyncGenerator
import openai

class EmbeddingProcessor:
    def __init__(self, api_key: str, batch_size: int = 100):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.batch_size = batch_size
    
    async def batch_embed_chunks(self, chunks: List[str]) -> AsyncGenerator[List[List[float]], None]:
        """Batch process chunks for embedding generation."""
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            embeddings = await self.get_embeddings_batch(batch)
            yield embeddings
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 1536 for _ in texts]
```

**Vector Storage**
```python
import sqlite3
import json
import numpy as np
from typing import List, Tuple

class SimpleVectorDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database for vector storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def store_document(self, content: str, embedding: List[float], metadata: dict = None):
        """Store document with its embedding."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO documents (content, embedding, metadata)
            VALUES (?, ?, ?)
        ''', (content, json.dumps(embedding), json.dumps(metadata or {})))
        conn.commit()
        conn.close()
    
    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Perform similarity search."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT content, embedding FROM documents')
        
        results = []
        for content, embedding_str in cursor.fetchall():
            doc_embedding = json.loads(embedding_str)
            similarity = cosine_similarity(query_embedding, doc_embedding)
            results.append((content, similarity))
        
        conn.close()
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)
```

## Level 2: More Structured Processing

### Enhanced Processing Components
- Better async operations
- Advanced chunking strategies
- Retry mechanisms
- Query expansion and rewriting
- Structured responses with citations

### Advanced Chunking Strategies

**Semantic Chunking**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticChunker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.8
    
    def semantic_chunk(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """Chunk text based on semantic similarity."""
        sentences = self.split_into_sentences(text)
        if not sentences:
            return [text]
        
        # Get embeddings for all sentences
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i]
            
            # Calculate similarity with current chunk
            similarity = cosine_similarity(current_embedding, sentence_embedding)
            
            # Check if we should start a new chunk
            chunk_text = ' '.join(current_chunk + [sentence])
            if (similarity < self.similarity_threshold or 
                self.estimate_tokens(chunk_text) > max_chunk_size):
                
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_embedding = sentence_embedding
            else:
                current_chunk.append(sentence)
                # Update embedding as moving average
                current_embedding = (current_embedding + sentence_embedding) / 2
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting - can be enhanced with spaCy or NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
```

**Financial Document Specific Chunking**
```python
class FinancialDocumentChunker:
    def __init__(self):
        self.section_patterns = {
            'financial_statements': [
                r'INCOME STATEMENT', r'BALANCE SHEET', r'CASH FLOW',
                r'STATEMENT OF.*EARNINGS', r'CONSOLIDATED.*OPERATIONS'
            ],
            'risk_factors': [
                r'RISK FACTORS', r'MARKET RISK', r'CREDIT RISK',
                r'OPERATIONAL RISK', r'REGULATORY RISK'
            ],
            'management_discussion': [
                r'MANAGEMENT.*DISCUSSION', r'MD&A', r'BUSINESS OVERVIEW'
            ]
        }
    
    def chunk_financial_document(self, text: str, doc_type: str = None) -> List[dict]:
        """Chunk financial documents preserving structure."""
        if doc_type == 'earnings_report':
            return self.chunk_earnings_report(text)
        elif doc_type == 'annual_report':
            return self.chunk_annual_report(text)
        else:
            return self.generic_financial_chunk(text)
    
    def chunk_earnings_report(self, text: str) -> List[dict]:
        """Chunk earnings reports by sections."""
        chunks = []
        
        # Look for common earnings report sections
        sections = self.identify_sections(text, [
            'EXECUTIVE SUMMARY', 'FINANCIAL HIGHLIGHTS', 'REVENUE ANALYSIS',
            'EXPENSE ANALYSIS', 'OUTLOOK', 'Q&A'
        ])
        
        for section_name, section_text in sections.items():
            # Further chunk large sections
            if len(section_text) > 2000:
                sub_chunks = self.chunk_by_paragraphs(section_text, max_size=800)
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        'content': sub_chunk,
                        'section': section_name,
                        'sub_section': i,
                        'chunk_type': 'earnings_section'
                    })
            else:
                chunks.append({
                    'content': section_text,
                    'section': section_name,
                    'chunk_type': 'earnings_section'
                })
        
        return chunks
    
    def preserve_table_structure(self, text: str) -> List[dict]:
        """Preserve financial tables as single chunks."""
        import re
        
        # Detect table patterns
        table_pattern = r'\|.*\|.*\n(\|.*\|.*\n)+'
        tables = re.finditer(table_pattern, text)
        
        chunks = []
        last_end = 0
        
        for match in tables:
            # Add text before table
            if match.start() > last_end:
                pre_table_text = text[last_end:match.start()].strip()
                if pre_table_text:
                    chunks.extend(self.chunk_by_paragraphs(pre_table_text))
            
            # Add table as single chunk
            table_text = match.group(0)
            chunks.append({
                'content': table_text,
                'chunk_type': 'financial_table',
                'preserve_formatting': True
            })
            
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(text):
            remaining_text = text[last_end:].trip()
            if remaining_text:
                chunks.extend(self.chunk_by_paragraphs(remaining_text))
        
        return chunks
```

### Enhanced Query Processing

**Query Expansion and Rewriting**
```python
class FinancialQueryProcessor:
    def __init__(self):
        self.financial_synonyms = {
            'revenue': ['sales', 'turnover', 'income', 'top line', 'gross revenue'],
            'profit': ['earnings', 'net income', 'bottom line', 'net profit'],
            'margin': ['profitability', 'profit margin', 'operating margin'],
            'debt': ['liabilities', 'borrowings', 'obligations', 'leverage'],
            'assets': ['holdings', 'resources', 'property', 'investments'],
            'growth': ['increase', 'expansion', 'improvement', 'gains'],
            'decline': ['decrease', 'reduction', 'drop', 'fall'],
            'cash flow': ['cash generation', 'operating cash flow', 'free cash flow']
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with financial synonyms."""
        expanded_queries = [query]  # Always include original
        
        query_lower = query.lower()
        for term, synonyms in self.financial_synonyms.items():
            if term in query_lower:
                for synonym in synonyms[:2]:  # Limit to avoid explosion
                    expanded_query = query.replace(term, synonym)
                    expanded_queries.append(expanded_query)
        
        return expanded_queries
    
    def rewrite_temporal_queries(self, query: str) -> List[str]:
        """Rewrite queries to be more specific about time periods."""
        temporal_rewrites = []
        
        if 'latest' in query.lower():
            temporal_rewrites.extend([
                query.replace('latest', 'most recent quarter'),
                query.replace('latest', 'current fiscal year'),
                query.replace('latest', '2024')
            ])
        
        if 'recent' in query.lower():
            temporal_rewrites extend([
                query.replace('recent', 'last quarter'),
                query.replace('recent', 'past year')
            ])
        
        return temporal rewrites if temporal_rewrites else [query]
```

**Structured Response Generation**
```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Citation:
    text: str
    source: str
    confidence: float
    page_number: Optional[int] = None

@dataclass
class StructuredResponse:
    answer: str
    citations: List[Citation]
    confidence_score: float
    retrieved_chunks: List[str]
    query_understanding: str

class ResponseGenerator:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def generate_structured_response(
        self, 
        query: str, 
        relevant_chunks: List[Tuple[str, float]]
    ) -> StructuredResponse:
        """Generate a structured response with citations."""
        
        # Prepare context from relevant chunks
        context = self.prepare_context(relevant_chunks)
        
        # Generate response with citation requirements
        prompt = self.create_citation_prompt(query, context)
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Provide accurate answers with specific citations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        # Parse response and extract citations
        answer, citations = self.parse_response_with_citations(
            response.choices[0].message.content,
            relevant_chunks
        )
        
        return StructuredResponse(
            answer=answer,
            citations=citations,
            confidence_score=self.calculate_confidence(relevant_chunks),
            retrieved_chunks=[chunk[0] for chunk in relevant_chunks],
            query_understanding=self.understand_query_intent(query)
        )
    
    def create_citation_prompt(self, query: str, context: str) -> str:
        """Create a prompt that encourages proper citations."""
        return f"""
        Based on the following financial documents, answer this question: {query}

        Context:
        {context}

        Instructions:
        1. Provide a clear, accurate answer
        2. Include specific numbers and figures when relevant
        3. Cite your sources using [Source: document_name] format
        4. If information is not available, state this clearly
        5. Focus on factual information only

        Answer:
        """
    
    def parse_response_with_citations(
        self, 
        response_text: str, 
        source_chunks: List[Tuple[str, float]]
    ) -> Tuple[str, List[Citation]]:
        """Parse response and extract citations."""
        import re
        
        citations = []
        citation_pattern = r'\[Source: ([^\]]+)\]'
        
        # Find all citations in the response
        citation_matches = re.finditer(citation_pattern, response_text)
        
        for match in citation_matches:
            source_name = match.group(1)
            
            # Find the most relevant chunk for this citation
            best_chunk = None
            best_score = 0
            
            for chunk, score in source_chunks:
                if source_name.lower() in chunk.lower() or score > best_score:
                    best_chunk = chunk
                    best_score = score
            
            if best_chunk:
                citations.append(Citation(
                    text=best_chunk[:200] + "...",  # Truncate for display
                    source=source_name,
                    confidence=best_score
                ))
        
        # Clean response text
        answer = re.sub(citation_pattern, '', response_text). strip()
        
        return answer, citations
```

## Level 3: Observability

### Comprehensive Logging and Monitoring

**Wide Event Tracking**
```python
import logging
import json
from datetime import datetime
from typing import Dict, Any
import structlog

class RAGObservability:
    def __init__(self):
        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        self.logger = structlog.get_logger()
    
    def log_query_pipeline(
        self,
        query: str,
        query_embeddings_latency: float,
        retrieval_latency: float,
        generation_latency: float,
        retrieved_docs: List[Tuple[str, float]],
        final_response: str,
        user_id: str = None
    ):
        """Log complete query pipeline execution."""
        
        event_data = {
            "event_type": "rag_query_complete",
            "user_id": user_id,
            "query": {
                "text": query,
                "length": len(query),
                "query_type": self.classify_query(query),
                "embedding_latency_ms": query_embeddings_latency * 1000
            },
            "retrieval": {
                "latency_ms": retrieval_latency * 1000,
                "documents_found": len(retrieved_docs),
                "avg_similarity": sum(score for _, score in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
                "top_similarity": max(score for _, score in retrieved_docs) if retrieved_docs else 0
            },
            "generation": {
                "latency_ms": generation_latency * 1000,
                "response_length": len(final_response),
                "contains_numbers": self.contains_financial_numbers(final_response)
            },
            "total_latency_ms": (query_embeddings_latency + retrieval_latency + generation_latency) * 1000
        }
        
        self.logger.info("RAG query completed", **event_data)
        return event_data
    
    def log_retrieval_performance(
        self,
        query: str,
        embedding_time: float,
        search_time: float,
        rerank_time: float,
        total_candidates: int,
        final_results: int
    ):
        """Log detailed retrieval performance metrics."""
        
        self.logger.info(
            "Retrieval performance",
            query=query,
            embedding_latency_ms=embedding_time * 1000,
            search_latency_ms=search_time * 1000,
            rerank_latency_ms=rerank_time * 1000,
            candidates_found=total_candidates,
            results_returned=final_results,
            retrieval_ratio=final_results / total_candidates if total_candidates > 0 else 0
        )
    
    def log_user_feedback(
        self,
        query_id: str,
        query: str,
        response: str,
        rating: int,
        feedback_text: str = None
    ):
        """Log user feedback for continuous improvement."""
        
        self.logger.info(
            "User feedback received",
            query_id=query_id,
            query=query,
            response_preview=response[:100],
            rating=rating,
            feedback_text=feedback_text,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def classify_query(self, query: str) -> str:
        """Classify query type for analytics."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['revenue', 'sales', 'income']):
            return 'revenue_query'
        elif any(word in query_lower for word in ['profit', 'earnings', 'margin']):
            return 'profitability_query'
        elif any(word in query_lower for word in ['risk', 'exposure', 'volatility']):
            return 'risk_query'
        elif any(word in query_lower for word in ['cash flow', 'liquidity']):
            return 'cash_flow_query'
        elif any(word in query_lower for word in ['compare', 'vs', 'difference']):
            return 'comparison_query'
        else:
            return 'general_query'
    
    def contains_financial_numbers(self, text: str) -> bool:
        """Check if response contains financial figures."""
        import re
        # Look for currency symbols, percentages, or large numbers
        financial_patterns = [
            r'\$[\d,]+\.?\d*[MBK]?',  # Currency
            r'\d+\.?\d*%',            # Percentages
            r'\d{1,3}(,\d{3})*\.?\d*', # Large numbers with commas
        ]
        
        return any(re.search(pattern, text) for pattern in financial_patterns)
```

**Performance Monitoring**
```python
import time
from functools import wraps
from typing: Callable, Any
import psutil
import threading

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background monitoring of system resources."""
        def monitor_resources():
            while True:
                self.metrics.update({
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                })
                time.sleep(10)  # Update every 10 seconds
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def timing_decorator(self, operation_name: str):
        """Decorator to time function execution."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    execution_time = time.time() - start_time
                    self.log_performance(operation_name, execution_time, success, error)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    execution_time = time.time() - start_time
                    self.log_performance(operation_name, execution_time, success, error)
                
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def log_performance(self, operation: str, duration: float, success: bool, error: str = None):
        """Log performance metrics."""
        metrics = {
            'operation': operation,
            'duration_ms': duration * 1000,
            'success': success,
            'error': error,
            'cpu_percent': self.metrics.get('cpu_percent', 0),
            'memory_percent': self.metrics.get('memory_percent', 0),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger = structlog.get_logger()
        logger.info("Performance metric", **metrics)
```

## Level 4: Evaluation

### Synthetic Data Generation and Evaluation Framework

**Evaluation Dataset Generation**
```python
from typing import List, Dict
import random

class FinancialRAGEvaluator:
    def __init__(self):
        self.question_templates = {
            'factual': [
                "What was the {metric} in {period}?",
                "How much {metric} did the company report for {period}?",
                "What is the value of {metric} according to the {document_type}?"
            ],
            'comparative': [
                "How did {metric} change from {period1} to {period2}?",
                "Compare the {metric} between {period1} and {period2}",
                "What was the growth in {metric} year over year?"
            ],
            'analytical': [
                "What are the main factors affecting {metric}?",
                "Why did {metric} {change_direction} in {period}?",
                "What risks are associated with {metric}?"
            ]
        }
        
        self.financial_metrics = [
            'revenue', 'net income', 'gross profit', 'operating income',
            'cash flow', 'total assets', 'debt', 'equity', 'margins'
        ]
        
        self.time_periods = [
            'Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024',
            'fiscal year 2024', 'last quarter', 'this year'
        ]
    
    def generate_evaluation_questions(self, num_questions: int = 50) -> List[Dict]:
        """Generate evaluation questions for financial documents."""
        questions = []
        
        for _ in range(num_questions):
            question_type = random.choice(list(self.question_templates.keys()))
            template = random.choice(self.question_templates[question_type])
            
            if question_type == 'factual':
                question = template.format(
                    metric=random.choice(self.financial_metrics),
                    period=random.choice(self.time_periods),
                    document_type=random.choice(['annual report', 'quarterly filing', 'earnings statement'])
                )
                difficulty = 'easy'
                
            elif question_type == 'comparative':
                periods = random.sample(self.time_periods, 2)
                question = template.format(
                    metric=random.choice(self.financial_metrics),
                    period1=periods[0],
                    period2=periods[1]
                )
                difficulty = 'medium'
                
            elif question_type == 'analytical':
                question = template.format(
                    metric=random.choice(self.financial_metrics),
                    period=random.choice(self.time_periods),
                    change_direction=random.choice(['increase', 'decrease', 'improve', 'decline'])
                )
                difficulty = 'hard'
            
            questions.append({
                'question': question,
                'type': question_type,
                'difficulty': difficulty,
                'expected_retrieval_type': self.determine_retrieval_type(question),
                'evaluation_criteria': self.get_evaluation_criteria(question_type)
            })
        
        return questions
    
    def determine_retrieval_type(self, question: str) -> str:
        """Determine what type of document should be retrieved."""
        question_lower = question.lower()
        
        if 'risk' in question_lower:
            return 'risk_assessment'
        elif any(term in question_lower for term in ['cash flow', 'liquidity']):
            return 'cash_flow_statement'
        elif any(term in question_lower for term in ['revenue', 'profit', 'earnings']):
            return 'income_statement'
        else:
            return 'general_financial'
    
    def get_evaluation_criteria(self, question_type: str) -> List[str]:
        """Define evaluation criteria for different question types."""
        criteria_map = {
            'factual': ['accuracy', 'completeness', 'source_citation'],
            'comparative': ['accuracy', 'mathematical_correctness', 'temporal_understanding'],
            'analytical': ['reasoning_quality', 'context_awareness', 'risk_identification']
        }
        return criteria_map.get(question_type, ['general_quality'])
```

**Automated Evaluation System**
```python
import asyncio
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

class AutomatedEvaluator:
    def __init__(self, rag_system, similarity_model_name: str = "all-MiniLM-L6-v2"):
        self.rag_system = rag_system
        self.similarity_model = SentenceTransformer(similarity_model_name)
        self.evaluation_metrics = [
            'answer_relevance',
            'factual_accuracy', 
            'completeness',
            'citation_quality',
            'response_time'
        ]
    
    async def evaluate_system(self, evaluation_questions: List[Dict]) -> Dict[str, Any]:
        """Evaluate RAG system against a set of questions."""
        results = []
        
        for question_data in evaluation_questions:
            question = question_data['question']
            start_time = time.time()
            
            # Get system response
            try:
                response = await self.rag_system.query(question)
                response_time = time.time() - start_time
                success = True
            except Exception as e:
                response = None
                response_time = time.time() - start_time
                success = False
                print(f"Error processing question: {question}. Error: {e}")
                continue
            
            # Evaluate response
            evaluation_scores = await self.evaluate_response(
                question_data, response, response_time
            )
            
            results.append({
                'question': question,
                'question_type': question_data['type'],
                'difficulty': question_data['difficulty'],
                'system_response': response.get('answer', '') if response else '',
                'response_time': response_time,
                'success': success,
                'evaluation_scores': evaluation_scores
            })
        
        # Aggregate results
        aggregated_metrics = self.aggregate_evaluation_results(results)
        
        return {
            'individual_results': results,
            'summary_metrics': aggregated_metrics,
            'total_questions': len(evaluation_questions),
            'successful_responses': sum(1 for r in results if r['success'])
        }
    
    async def evaluate_response(
        self, 
        question_data: Dict, 
        response: Dict, 
        response_time: float
    ) -> Dict[str, float]:
        """Evaluate individual response across multiple dimensions."""
        
        scores = {}
        
        # 1. Answer Relevance (semantic similarity to question)
        scores['answer_relevance'] = self.calculate_answer_relevance(
            question_data['question'], 
            response.get('answer', '')
        )
        
        # 2. Response Time Performance
        scores['response_time_score'] = self.evaluate_response_time(response_time)
        
        # 3. Citation Quality
        scores['citation_quality'] = self.evaluate_citations(
            response.get('citations', []),
            response.get('retrieved_chunks', [])
        )
        
        # 4. Financial Accuracy (check for financial figures)
        scores['financial_accuracy'] = self.check_financial_consistency(
            response.get('answer', ''),
            response.get('retrieved_chunks', [])
        )
        
        # 5. Completeness based on question type
        scores['completeness'] = self.assess_completeness(
            question_data, 
            response.get('answer', '')
        )
        
        # Overall score (weighted average)
        weights = {
            'answer_relevance': 0.3,
            'response_time_score': 0.1,
            'citation_quality': 0.2,
            'financial_accuracy': 0.25,
            'completeness': 0.15
        }
        
        scores['overall_score'] = sum(
            scores[metric] * weight 
            for metric, weight in weights.items()
        )
        
        return scores
    
    def calculate_answer_relevance(self, question: str, answer: str) -> float:
        """Calculate semantic similarity between question and answer."""
        if not answer:
            return 0.0
        
        question_embedding = self.similarity_model.encode([question])
        answer_embedding = self.similarity_model.encode([answer])
        
        similarity = cosine_similarity(question_embedding, answer_embedding)[0][0]
        return float(similarity)
    
    def evaluate_response_time(self, response_time: float) -> float:
        """Evaluate response time performance."""
        # Good: < 2s, Acceptable: < 5s, Poor: > 5s
        if response_time < 2:
            return 1.0
        elif response_time < 5:
            return 0.8
        elif response_time < 10:
            return 0.6
        else:
            return 0.3
    
    def check_financial_consistency(
        self, 
        answer: str, 
        retrieved_chunks: List[str]
    ) -> float:
        """Check if financial figures in answer match source documents."""
        import re
        
        # Extract financial figures from answer
        answer_figures = self.extract_financial_figures(answer)
        
        if not answer_figures:
            return 1.0  # No figures to verify
        
        # Extract figures from source documents
        source_figures = []
        for chunk in retrieved_chunks:
            source_figures.extend(self.extract_financial_figures(chunk))
        
        if not source_figures:
            return 0.5  # Answer has figures but sources don't
        
        # Check if answer figures can be found in sources
        verified_count = 0
        for answer_fig in answer_figures:
            if any(abs(answer_fig - source_fig) / max(answer_fig, source_fig) < 0.05 
                   for source_fig in source_figures):
                verified_count += 1
        
        return verified_count / len(answer_figures)
    
    def extract_financial_figures(self, text: str) -> List[float]:
        """Extract numerical financial figures from text."""
        import re
        
        patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M)',  # $123.4 million
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|B)',  # $123.4 billion
            r'(\d+(?:\.\d+)?)%',  # 15.5%
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)',  # $123,456.78
        ]
        
        figures = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value_str = match.group(1).replace(',', '')
                    value = float(value_str)
                    
                    # Adjust for units
                    if 'million' in match.group(0).lower() or 'M' in match.group(0):
                        value *= 1_000_000
                    elif 'billion' in match.group(0).lower() or 'B' in match.group(0):
                        value *= 1_000_000_000
                    
                    figures.append(value)
                except ValueError:
                    continue
        
        return figures
    
    def aggregate_evaluation_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate evaluation results into summary metrics."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {'error': 'No successful responses to evaluate'}
        
        # Calculate averages for each metric
        metrics = {}
        for metric in self.evaluation_metrics:
            scores = [
                r['evaluation_scores'].get(metric, 0) 
                for r in successful_results 
                if 'evaluation_scores' in r
            ]
            if scores:
                metrics[f'avg_{metric}'] = np.mean(scores)
                metrics[f'std_{metric}'] = np.std(scores)
        
        # Performance by question type
        question_types = set(r['question_type'] for r in successful_results)
        type_performance = {}
        
        for q_type in question_types:
            type_results = [r for r in successful_results if r['question_type'] == q_type]
            if type_results:
                avg_score = np.mean([
                    r['evaluation_scores'].get('overall_score', 0) 
                    for r in type_results
                ])
                type_performance[q_type] = avg_score
        
        return {
            'overall_metrics': metrics,
            'performance_by_type': type_performance,
            'success_rate': len(successful_results) / len(results),
            'avg_response_time': np.mean([r['response_time'] for r in successful_results])
        }
```

## Implementation Roadmap

### Phase 1: Level 1 Implementation
1. **Set up basic document processing pipeline**
   - Implement file traversal for your document directory
   - Create chunking generator with overlap
   - Set up batch embedding processing

2. **Create simple vector storage**
   - Initialize SQLite database for vectors
   - Implement basic similarity search
   - Store document chunks with metadata

3. **Build basic query pipeline**
   - Embed user queries
   - Retrieve similar chunks
   - Generate simple responses

### Phase 2: Level 2 Enhancements
1. **Improve chunking strategies**
   - Implement semantic chunking
   - Add financial document-specific chunking
   - Preserve table structures

2. **Add query enhancement**
   - Build financial synonym expansion
   - Implement temporal query rewriting
   - Create structured response format

3. **Enhance async processing**
   - Batch embedding requests
   - Parallel query processing
   - Add retry mechanisms

### Phase 3: Level 3 Observability
1. **Implement comprehensive logging**
   - Set up structured logging
   - Track query pipeline metrics
   - Monitor retrieval performance

2. **Add performance monitoring**
   - System resource tracking
   - Response time monitoring
   - Error rate tracking

3. **User feedback collection**
   - Implement feedback endpoints
   - Store user ratings and comments
   - Track improvement opportunities

### Phase 4: Level 4 Evaluation
1. **Create evaluation framework**
   - Generate synthetic test questions
   - Implement automated evaluation
   - Set up performance benchmarks

2. **Continuous evaluation**
   - Regular system evaluation runs
   - Performance regression detection
   - A/B testing framework

This roadmap provides a structured approach to building a robust RAG system, progressing from basic functionality to advanced observability and evaluation capabilities.