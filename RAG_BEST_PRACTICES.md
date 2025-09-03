# RAG Implementation Guide: Levels 1-4

This guide walks through Jason Liu's RAG complexity framework with practical Python examples for financial document processing. I've found this progression works well for building production systems that actually stay running.

## RAG System Evolution

Most RAG tutorials show you the "happy path" - embed some docs, throw them in a vector store, do similarity search, generate a response. Works great until you hit production, then everything breaks.

### What Goes Wrong

The basic RAG pattern fails because:

- **Chunking destroys context** - splitting mid-sentence or mid-table makes financial data useless
- **No visibility when things break** - good luck debugging why the system suddenly gives terrible answers
- **No way to measure quality** - you're flying blind on whether changes help or hurt
- **Brittle retrieval** - misses obviously relevant docs because of word choice differences
- **Inconsistent outputs** - same question, different answer every time

### The Level Progression

I've built enough of these systems to know you need to solve problems in order:

1. **Level 1** - Handle real documents without running out of memory
2. **Level 2** - Improve retrieval and response quality  
3. **Level 3** - Add monitoring so you can debug production issues
4. **Level 4** - Build evaluation to measure if you're getting better

Skipping levels just means you'll be back fixing earlier problems when the system is already in production.

## Table of Contents
1. [Level 1: The Basics](#level-1-the-basics)
2. [Level 2: More Structured Processing](#level-2-more-structured-processing)
3. [Level 3: Observability](#level-3-observability)
4. [Level 4: Evaluation](#level-4-evaluation)
5. [Implementation Roadmap](#implementation-roadmap)

## Level 1: The Basics

### Getting the Foundation Right

The goal here is processing real documents without falling over. Most demos use small text files that fit in memory. Financial docs are different - SEC filings can be hundreds of pages, and you might have thousands of them.

#### Memory Management with Generators

Loading all your documents into memory is a recipe for OOM errors. I learned this the hard way on a project with 10k+ PDFs. Generators let you process one document at a time:

```python
def traverse_documents(root_path: str) -> Generator[str, None, None]:
    """Process documents one at a time to avoid memory issues."""
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(('.pdf', '.txt', '.docx', '.md')):
                file_path = os.path.join(root, file)
                yield extract_text_from_file(file_path)
```

#### Chunking That Preserves Meaning

Don't split financial documents randomly. A quarterly revenue number without its time period is worse than useless - it's misleading. Respect sentence boundaries and table structures:

```python
def chunk_text_generator(text: str, chunk_size: int = 1000, overlap: int = 200) -> Generator[str, None, None]:
    """Chunk text without destroying financial context."""
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Don't split mid-sentence in financial docs
        if end < len(text) and not text[end].isspace():
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            boundary = max(last_period, last_newline)
            
            if boundary > start + chunk_size // 2:  # Keep reasonable chunk sizes
                end = start + boundary + 1
                chunk = text[start:end]
        
        yield chunk.strip()
        start = end - overlap
```

#### Batch API Calls Efficiently

OpenAI's API has rate limits. Batching reduces latency and costs, but you need error handling so one bad chunk doesn't kill your entire batch:

```python
class EmbeddingProcessor:
    def __init__(self, api_key: str, batch_size: int = 100):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.batch_size = batch_size
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings with fallback for API failures."""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            print(f"Embedding API failed: {e}")
            # Return zero vectors rather than crashing
            return [[0.0] * 1536 for _ in texts]
```

#### Simple Vector Storage

Start with SQLite. It's fast enough for prototypes and you can always migrate to something fancier later:

```python
class SimpleVectorDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Brute force similarity search - works fine for < 100k docs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT content, embedding FROM documents')
        
        results = []
        for content, embedding_str in cursor.fetchall():
            doc_embedding = json.loads(embedding_str)
            similarity = cosine_similarity(query_embedding, doc_embedding)
            results.append((content, similarity))
        
        conn.close()
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
```

## Level 2: More Structured Processing

### Fixing Retrieval Quality

Level 1 gets you a working system, but the retrieval quality is often terrible. Financial documents have structure that naive chunking destroys, and financial language has synonyms that embedding models miss.

#### Semantic Chunking

Group related sentences together instead of splitting arbitrarily. Financial reports have logical flow - keep related ideas together:

```python
class SemanticChunker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.8
    
    def semantic_chunk(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """Group semantically similar sentences together."""
        sentences = self.split_into_sentences(text)
        if not sentences:
            return [text]
        
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i]
            
            similarity = cosine_similarity(current_embedding, sentence_embedding)
            chunk_text = ' '.join(current_chunk + [sentence])
            
            if (similarity < self.similarity_threshold or 
                len(chunk_text) > max_chunk_size * 4):  # rough token estimate
                
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_embedding = sentence_embedding
            else:
                current_chunk.append(sentence)
                current_embedding = (current_embedding + sentence_embedding) / 2
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
```

#### Financial Document Structure

Preserve tables and section boundaries. Financial tables split across chunks become meaningless:

```python
class FinancialDocumentChunker:
    def preserve_table_structure(self, text: str) -> List[dict]:
        """Keep financial tables intact - they're useless when split."""
        import re
        
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
            
            # Keep entire table together
            table_text = match.group(0)
            chunks.append({
                'content': table_text,
                'chunk_type': 'financial_table',
                'preserve_formatting': True
            })
            
            last_end = match.end()
        
        # Process remaining text
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                chunks.extend(self.chunk_by_paragraphs(remaining_text))
        
        return chunks
```

#### Query Expansion

Financial language is full of synonyms. "Revenue" vs "sales" vs "top line" - expand queries to catch variations:

```python
class FinancialQueryProcessor:
    def __init__(self):
        # Built this list from years of financial document work
        self.financial_synonyms = {
            'revenue': ['sales', 'turnover', 'income', 'top line'],
            'profit': ['earnings', 'net income', 'bottom line'],
            'margin': ['profitability', 'profit margin'],
            'debt': ['liabilities', 'borrowings', 'leverage'],
            'cash flow': ['cash generation', 'operating cash flow', 'free cash flow']
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand with financial synonyms."""
        expanded_queries = [query]
        
        query_lower = query.lower()
        for term, synonyms in self.financial_synonyms.items():
            if term in query_lower:
                for synonym in synonyms[:2]:  # Don't go overboard
                    expanded_query = query.replace(term, synonym)
                    expanded_queries.append(expanded_query)
        
        return expanded_queries
```

#### Structured Responses with Citations

Users need to verify financial information. Include source citations and confidence scores:

```python
@dataclass
class StructuredResponse:
    answer: str
    citations: List[Citation]
    confidence_score: float
    retrieved_chunks: List[str]

class ResponseGenerator:
    def create_citation_prompt(self, query: str, context: str) -> str:
        """Prompt that forces citation behavior."""
        return f"""Based on these financial documents, answer: {query}

Context:
{context}

Requirements:
- Include specific numbers and figures
- Cite sources using [Source: document_name] format
- State clearly if information is missing
- Stick to facts from the documents

Answer:"""
```

## Level 3: Observability

### Making the Black Box Transparent

When your RAG system starts giving bad answers in production, you need to figure out why. Was it bad retrieval? Poor generation? API issues? Without observability, you're debugging blind.

#### Comprehensive Logging

Log everything in the pipeline so you can trace problems:

```python
class RAGObservability:
    def __init__(self):
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.WriteLoggerFactory(),
        )
        self.logger = structlog.get_logger()
    
    def log_query_pipeline(
        self,
        query: str,
        retrieval_latency: float,
        generation_latency: float,
        retrieved_docs: List[Tuple[str, float]],
        final_response: str,
        user_id: str = None
    ):
        """Log the complete query flow for debugging."""
        
        self.logger.info("RAG query completed", 
            query=query,
            user_id=user_id,
            retrieval_latency_ms=retrieval_latency * 1000,
            generation_latency_ms=generation_latency * 1000,
            documents_found=len(retrieved_docs),
            avg_similarity=sum(score for _, score in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
            response_length=len(final_response),
            contains_financial_data=self.contains_financial_numbers(final_response)
        )
```

#### Performance Monitoring

Track system resources and response times to catch degradation early:

```python
class PerformanceMonitor:
    def timing_decorator(self, operation_name: str):
        """Time operations and track failures."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
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
            return wrapper
        return decorator
```

#### User Feedback Collection

The best signal for RAG quality is user feedback. Make it easy to collect:

```python
def log_user_feedback(
    self,
    query_id: str,
    query: str,
    response: str,
    rating: int,
    feedback_text: str = None
):
    """Track user satisfaction for continuous improvement."""
    
    self.logger.info("User feedback",
        query_id=query_id,
        query=query,
        response_preview=response[:100],
        rating=rating,
        feedback_text=feedback_text,
        timestamp=datetime.utcnow().isoformat()
    )
```

## Level 4: Evaluation

### Measuring What Matters

You can't improve what you don't measure. Level 4 builds systematic evaluation so you know if changes actually help.

#### Synthetic Test Generation

Don't wait for user feedback. Generate test cases that cover your domain:

```python
class FinancialRAGEvaluator:
    def __init__(self):
        # Real patterns I've seen in financial queries
        self.question_templates = {
            'factual': [
                "What was the {metric} in {period}?",
                "How much {metric} did the company report for {period}?",
            ],
            'comparative': [
                "How did {metric} change from {period1} to {period2}?",
                "Compare the {metric} between {period1} and {period2}",
            ],
            'analytical': [
                "What factors affected {metric} in {period}?",
                "Why did {metric} {change_direction} in {period}?",
            ]
        }
        
        self.financial_metrics = [
            'revenue', 'net income', 'gross profit', 'cash flow', 'debt'
        ]
        
        self.time_periods = [
            'Q1 2024', 'Q2 2024', 'fiscal year 2024'
        ]
```

#### Multi-Dimensional Evaluation

Financial RAG needs more than semantic similarity. Check accuracy, citations, and mathematical consistency:

```python
class AutomatedEvaluator:
    def check_financial_consistency(
        self, 
        answer: str, 
        retrieved_chunks: List[str]
    ) -> float:
        """Verify financial figures match source documents."""
        answer_figures = self.extract_financial_figures(answer)
        
        if not answer_figures:
            return 1.0  # No figures to verify
        
        source_figures = []
        for chunk in retrieved_chunks:
            source_figures.extend(self.extract_financial_figures(chunk))
        
        if not source_figures:
            return 0.5  # Answer has figures but sources don't
        
        # Check if answer figures appear in sources
        verified_count = 0
        for answer_fig in answer_figures:
            if any(abs(answer_fig - source_fig) / max(answer_fig, source_fig) < 0.05 
                   for source_fig in source_figures):
                verified_count += 1
        
        return verified_count / len(answer_figures)
    
    def extract_financial_figures(self, text: str) -> List[float]:
        """Pull out numbers that look like financial data."""
        import re
        
        patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M)',
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|B)',
            r'(\d+(?:\.\d+)?)%',
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)',
        ]
        
        figures = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value_str = match.group(1).replace(',', '')
                    value = float(value_str)
                    
                    if 'million' in match.group(0).lower() or 'M' in match.group(0):
                        value *= 1_000_000
                    elif 'billion' in match.group(0).lower() or 'B' in match.group(0):
                        value *= 1_000_000_000
                    
                    figures.append(value)
                except ValueError:
                    continue
        
        return figures
```

## Implementation Roadmap

### Phase 1: Get Level 1 Working
1. **File processing pipeline** - handle your actual document formats
2. **Vector storage** - start with SQLite, migrate later if needed
3. **Basic query pipeline** - embed query, search, generate response

### Phase 2: Improve Quality (Level 2)
1. **Better chunking** - semantic chunking for your domain
2. **Query enhancement** - expand with domain synonyms
3. **Structured responses** - citations and confidence scores

### Phase 3: Add Observability (Level 3)
1. **Comprehensive logging** - track the full query pipeline
2. **Performance monitoring** - response times and error rates
3. **User feedback** - simple thumbs up/down to start

### Phase 4: Systematic Evaluation (Level 4)
1. **Test generation** - synthetic questions covering your use cases
2. **Automated evaluation** - accuracy, citations, consistency checks
3. **Continuous monitoring** - catch regressions before users do

Build these in order. Each level solves real problems you'll hit as the system scales.