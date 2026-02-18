import os
import json
import time
import sqlite3
import numpy as np # Used in SemanticCache for embedding comparison
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Literal

# OpenAI specific imports
from openai import OpenAI
from pydantic import BaseModel, Field, confloat, model_validator

# --- Global Constants ---
# Define synthetic model costs per 1M tokens (as per OpenAI pricing)
# Pricing as of current date, subject to change. Using values from the attachment where possible.
MODEL_COSTS = {
    "gpt-4o-mini": {
        "input_cost_per_1M_tokens": 0.15,
        "output_cost_per_1M_tokens": 0.60,
    },
    "gpt-4o": {
        "input_cost_per_1M_tokens": 5.00,
        "output_cost_per_1M_tokens": 15.00,
    },
    "none": { # For direct API lookups with no LLM
        "input_cost_per_1M_tokens": 0.0,
        "output_cost_per_1M_tokens": 0.0,
    },
    "error": { # For error logging cost attribution
        "input_cost_per_1M_tokens": 0.0,
        "output_cost_per_1M_tokens": 0.0,
    },
    "cache": { # For cache hits, minimal overhead
        "input_cost_per_1M_tokens": 0.0,
        "output_cost_per_1M_tokens": 0.0,
    },
}

# Define costs for different handler types (simulated overheads or specific model usage)
HANDLER_OVERHEAD_COSTS = {
    "data_lookup": 0.001, # Cost for external API call, minimal
    "document_qa": 0.005, # Cost for RAG retrieval and processing
    "research_agent": 0.010, # Cost for agent orchestration and tool usage
    "general_knowledge": 0.000, # Handled directly by cheap LLM, minimal overhead
    "routing": 0.000, # Routing cost is already token-based, no additional overhead per query
    "cache_hit": 0.0005, # Minimal cost for cache lookup and data retrieval
}

# Router System Prompt for LLM classification
ROUTER_SYSTEM = (
    "You are a strict router for financial research queries. "
    "Classify into exactly one category and output ONLY valid JSON matching the schema."
)

ROUTER_PROMPT = """Classify the question into one category:

Categories:
- data_lookup: simple factual data retrievable via a direct API/DB call; no document retrieval needed.
- document_qa: requires looking up information inside specific filings/earnings calls/reports (RAG).
- general_knowledge: definitional/conceptual finance questions answerable without external tools.
- research_agent: multi-step research requiring synthesis/comparison/analysis across sources/tools.

Question:
{query}

Return JSON with:
- category: one of ["data_lookup","document_qa","general_knowledge","research_agent"]
- confidence: number in [0,1]
- reasoning: short justification
"""

# Semantic embedding model to use for caching
SEMANTIC_EMBEDDING_MODEL = "text-embedding-3-small"

# --- Pydantic Models for Structured Output ---

class QueryCategoryEnum(str, Enum):
    data_lookup = "data_lookup"
    document_qa = "document_qa"
    general_knowledge = "general_knowledge"
    research_agent = "research_agent"
    error = "error" # Add an error category for routing failures

class QueryRoute(BaseModel):
    category: QueryCategoryEnum
    confidence: confloat(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)

    @model_validator(mode="after")
    def sanity(self):
        # Optional: discourage overconfident low-effort routing
        if self.confidence > 0.99 and len(self.reasoning.strip()) < 10:
            raise ValueError("If confidence is extremely high, provide non-trivial reasoning.")
        return self

# --- Compliance Logger Class ---

class ComplianceLogger:
    """
    Manages logging of all AI interactions for regulatory compliance and internal audit.
    Uses SQLite for persistent storage.
    """
    def __init__(self, db_path: str = 'copilot_compliance.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        """Creates the interactions table if it does not exist."""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                sources TEXT,
                category TEXT,
                model TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL,
                latency_sec REAL,
                cached INTEGER,
                feedback TEXT DEFAULT NULL
            )
        ''')
        self.conn.commit()

    def log(self, interaction_data: Dict[str, Any]):
        """
        Logs a single AI interaction to the database.

        Args:
            interaction_data (Dict[str, Any]): A dictionary containing interaction details.
                                               Must include: query, response, sources (json string),
                                               category, model, cost_usd, latency_sec, user_id, cached (bool).
                                               Optionally includes input_tokens, output_tokens.
        """
        try:
            timestamp = datetime.now().isoformat()

            # Ensure sources is a JSON string
            sources_str = json.dumps(interaction_data.get('sources', []))

            self.conn.execute('''
                INSERT INTO interactions (
                    timestamp, user_id, query, response, sources, category,
                    model, input_tokens, output_tokens, cost_usd, latency_sec, cached
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                interaction_data.get('user_id', 'unknown_user'),
                interaction_data['query'],
                interaction_data['response'],
                sources_str,
                interaction_data['category'],
                interaction_data['model_used'],
                interaction_data.get('input_tokens', 0),
                interaction_data.get('output_tokens', 0),
                interaction_data['cost_usd'],
                interaction_data['latency_sec'],
                1 if interaction_data.get('cached', False) else 0 # Store as integer
            ))
            self.conn.commit()
        except Exception as e:
            print(f"Error logging interaction: {e}")
            # print(f"Problematic data: {interaction_data}") # Keep this for debugging if needed

    def get_cost_summary(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Generates a cost summary for the last N days, broken down by day and category.
        """
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = self.conn.execute(f'''
            SELECT
                DATE(timestamp) as day,
                category,
                SUM(cost_usd) as total_cost,
                COUNT(*) as queries,
                AVG(latency_sec) as avg_latency
            FROM interactions
            WHERE timestamp >= ?
            GROUP BY day, category
            ORDER BY day DESC, category
        ''', (start_date,))
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def export_audit_report(self, user_id: Optional[str] = None, start_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Exports all interactions for compliance review, with optional filters.
        """
        query_parts = ["SELECT * FROM interactions WHERE 1=1"]
        params = []
        if user_id:
            query_parts.append(" AND user_id = ?")
            params.append(user_id)
        if start_date:
            query_parts.append(" AND timestamp >= ?")
            params.append(start_date)

        query = " ".join(query_parts)
        cursor = self.conn.execute(query, params)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self):
        """Closes the database connection."""
        self.conn.close()

# --- Semantic Cache Class ---

class SemanticCache:
    """
    A conceptual semantic cache for storing and retrieving query responses
    based on embedding similarity and time-based expiry.
    """
    def __init__(self, openai_client: OpenAI, embedding_model_name: str, similarity_threshold: float = 0.90, max_age_hours: int = 1):
        self.client = openai_client
        self.embedding_model_name = embedding_model_name
        self.similarity_threshold = similarity_threshold
        self.max_age_seconds = max_age_hours * 3600
        self.cache: List[Dict[str, Any]] = [] # Stores [{{embedding, query, response, timestamp, metadata}}]

    def _get_embedding(self, text: str) -> List[float]:
        """Generates an embedding for the given text using OpenAI's embedding model."""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.embedding_model_name
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for text: '{text[:50]}...' - {e}")
            return [] # Return empty list on failure

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Checks if a semantically similar query was recently answered and returns the cached entry.
        Returns None if no hit or if the entry is too old.
        """
        if not self.cache:
            return None

        query_emb = self._get_embedding(query)
        if not query_emb:
            return None # Cannot process query if embedding fails

        query_emb_np = np.array(query_emb)
        now = datetime.now().timestamp()

        # Prune old entries during retrieval to keep cache lean
        self.cache = [entry for entry in self.cache if now - entry['timestamp'] < self.max_age_seconds]

        best_match = None
        max_similarity = 0.0

        for entry in self.cache:
            entry_emb_np = np.array(entry['embedding'])
            # Calculate cosine similarity
            norm_query = np.linalg.norm(query_emb_np)
            norm_entry = np.linalg.norm(entry_emb_np)

            if norm_query == 0 or norm_entry == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_emb_np, entry_emb_np) / (norm_query * norm_entry)

            if similarity >= self.similarity_threshold and similarity > max_similarity:
                max_similarity = similarity
                best_match = entry

        if best_match:
            response_data = {
                "query": query,
                "response": best_match['response']['response'],
                "sources": best_match['response']['sources'],
                "category": best_match['response']['category'],
                "model_used": "cache",
                "latency_sec": round(np.random.uniform(0.05, 0.2), 2), # Very fast from cache
                "user_id": best_match['user_id'], # Use the user who put it in cache, will be overwritten by current user_id for logging
                "ai_generated": True,
                "disclaimer": "AI-generated content. Verify before use.",
                "cached": True,
                "cache_hit_similarity": float(max_similarity),
                "cost_usd": HANDLER_OVERHEAD_COSTS["cache_hit"], # Minimal cost for cache lookup
                "input_tokens": 0,
                "output_tokens": 0,
            }
            return response_data
        return None

    def put(self, query: str, response_data: Dict[str, Any], user_id: str = "analyst_01"):
        """Stores a query-response pair in the cache along with its embedding."""
        embedding = self._get_embedding(query)
        if not embedding:
            return # Don't cache if embedding fails

        now = datetime.now().timestamp()

        # Store a simplified version of the response data to avoid deep nesting in cache
        cached_response_content = {
            "response": response_data.get('response'),
            "sources": response_data.get('sources'),
            "category": response_data.get('category'),
        }

        self.cache.append({
            'embedding': embedding,
            'query': query,
            'response': cached_response_content,
            'timestamp': now,
            'user_id': user_id,
        })
        # Prune old entries after adding new one
        self.cache = [entry for entry in self.cache if now - entry['timestamp'] < self.max_age_seconds]

# --- Main Financial Copilot Class ---

class FinancialCopilot:
    """
    Orchestrates the financial research copilot functionalities including
    query routing, handling, semantic caching, and compliance logging.
    """
    def __init__(self, openai_api_key: str, db_path: str = 'copilot_compliance.db',
                 cache_max_age_hours: int = 1, cache_threshold: float = 0.90):
        self.client = OpenAI(api_key=openai_api_key)
        self.logger = ComplianceLogger(db_path=db_path)
        self.semantic_cache = SemanticCache(
            openai_client=self.client,
            embedding_model_name=SEMANTIC_EMBEDDING_MODEL,
            similarity_threshold=cache_threshold,
            max_age_hours=cache_max_age_hours
        )

    def _route_query(self, query: str) -> Dict[str, Any]:
        """
        Routes a financial research query to the appropriate handler using a lightweight LLM.
        """
        try:
            start_time = time.time()
            response = self.client.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM},
                    {"role": "user", "content": ROUTER_PROMPT.format(query=query)},
                ],
                response_format=QueryRoute,
                temperature=0.0,
                max_tokens=200,
            )
            end_time = time.time()
            latency_sec = round(end_time - start_time, 2)

            parsed_output: QueryRoute = response.choices[0].message.parsed
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            input_cost = (prompt_tokens / 1_000_000) * MODEL_COSTS["gpt-4o-mini"]["input_cost_per_1M_tokens"]
            output_cost = (completion_tokens / 1_000_000) * MODEL_COSTS["gpt-4o-mini"]["output_cost_per_1M_tokens"]
            routing_cost_usd = input_cost + output_cost

            return {
                "category": parsed_output.category.value,
                "confidence": float(parsed_output.confidence),
                "reasoning": parsed_output.reasoning,
                "routing_cost_usd": routing_cost_usd,
                "model_used_for_routing": "gpt-4o-mini",
                "routing_latency_sec": latency_sec,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            }

        except Exception as e:
            print(f"Error routing query '{query}': {e}")
            return {
                "category": QueryCategoryEnum.error.value,
                "confidence": 0.0,
                "reasoning": f"Routing failed: {e}",
                "routing_cost_usd": 0.0,
                "model_used_for_routing": "none",
                "routing_latency_sec": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

    def _handle_data_lookup(self, query: str) -> Dict[str, Any]:
        """Simulates a direct API call to a market data provider."""
        start_time = time.time()
        mock_response = f"Simulated: Data lookup for '{query}' successfully performed. [Source: yfinance/Bloomberg API]"
        mock_sources = [{"type": "market_data", "provider": "yfinance", "url": "https://finance.yahoo.com/lookup"}]
        cost_usd = HANDLER_OVERHEAD_COSTS["data_lookup"]
        latency_sec = round(time.time() - start_time + np.random.uniform(0.1, 0.5), 2)
        return {
            "response": mock_response,
            "sources": mock_sources,
            "model_used": "none",
            "cost_usd": cost_usd,
            "latency_sec": latency_sec,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def _rag_answer(self, query: str) -> Dict[str, Any]:
        """Simulates a RAG pipeline using a more expensive LLM (gpt-4o)."""
        start_time = time.time()
        prompt_tokens = 500 # Simulating input context + query
        completion_tokens = 200 # Simulating generated answer

        input_cost = (prompt_tokens / 1_000_000) * MODEL_COSTS["gpt-4o"]["input_cost_per_1M_tokens"]
        output_cost = (completion_tokens / 1_000_000) * MODEL_COSTS["gpt-4o"]["output_cost_per_1M_tokens"]
        llm_cost_usd = input_cost + output_cost

        mock_response = f"Simulated: Comprehensive answer for '{query}' based on internal documents. [Source: Internal 10-K filings, Earnings Call Transcripts]"
        mock_sources = [{"type": "document", "id": "doc_123", "page": "5"}, {"type": "document", "id": "earnings_q4", "timestamp": "2023-10-25"}]

        cost_usd = HANDLER_OVERHEAD_COSTS["document_qa"] + llm_cost_usd
        latency_sec = round(time.time() - start_time + np.random.uniform(1.0, 3.0), 2)
        return {
            "response": mock_response,
            "sources": mock_sources,
            "model_used": "gpt-4o",
            "cost_usd": cost_usd,
            "latency_sec": latency_sec,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        }

    def _run_agent(self, query: str) -> Dict[str, Any]:
        """Simulates an agentic workflow using a powerful LLM (gpt-4o) and tools."""
        start_time = time.time()
        prompt_tokens = 1000 # Simulating tool calls, scratchpad, reasoning
        completion_tokens = 400 # Simulating synthesized analysis

        input_cost = (prompt_tokens / 1_000_000) * MODEL_COSTS["gpt-4o"]["input_cost_per_1M_tokens"]
        output_cost = (completion_tokens / 1_000_000) * MODEL_COSTS["gpt-4o"]["output_cost_per_1M_tokens"]
        llm_cost_usd = input_cost + output_cost

        mock_response = f"Simulated: Detailed multi-step analysis for '{query}' using agentic workflow. [Trace: Data Fetch, Analysis, Synthesis]"
        mock_sources = [{"type": "agent_trace", "steps": 3, "tools_used": ["market_data", "news_sentiment"]}]

        cost_usd = HANDLER_OVERHEAD_COSTS["research_agent"] + llm_cost_usd
        latency_sec = round(time.time() - start_time + np.random.uniform(3.0, 7.0), 2)
        return {
            "response": mock_response,
            "sources": mock_sources,
            "model_used": "gpt-4o",
            "cost_usd": cost_usd,
            "latency_sec": latency_sec,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        }

    def _handle_general_knowledge(self, query: str) -> Dict[str, Any]:
        """Uses a general-purpose, cost-effective LLM (gpt-4o-mini) for general knowledge."""
        start_time = time.time()
        try:
            response_llm = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a CFA-qualified financial analyst. Provide concise, accurate answers."},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=200
            )
            llm_response_text = response_llm.choices[0].message.content

            prompt_tokens = response_llm.usage.prompt_tokens
            completion_tokens = response_llm.usage.completion_tokens

            input_cost = (prompt_tokens / 1_000_000) * MODEL_COSTS["gpt-4o-mini"]["input_cost_per_1M_tokens"]
            output_cost = (completion_tokens / 1_000_000) * MODEL_COSTS["gpt-4o-mini"]["output_cost_per_1M_tokens"]
            llm_cost_usd = input_cost + output_cost

            mock_sources = [{"type": "llm_knowledge", "model": "gpt-4o-mini"}]

            cost_usd = HANDLER_OVERHEAD_COSTS["general_knowledge"] + llm_cost_usd
            latency_sec = round(time.time() - start_time + np.random.uniform(0.5, 1.5), 2)
            return {
                "response": llm_response_text,
                "sources": mock_sources,
                "model_used": "gpt-4o-mini",
                "cost_usd": cost_usd,
                "latency_sec": latency_sec,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            }
        except Exception as e:
            print(f"Error handling general knowledge query '{query}': {e}")
            return {
                "response": "Error processing general knowledge query.",
                "sources": [],
                "model_used": "error",
                "cost_usd": 0.0,
                "latency_sec": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

    def handle_query(self, query: str, user_id: str = "analyst_01", simulate_failure: bool = False) -> Dict[str, Any]:
        """
        Main entry point for the research copilot: routes, processes, caches, logs, and returns response.
        Includes fallback handling.
        """
        total_start_time = time.time()

        # Initialize a default error response
        formatted_response = {
            "query": query,
            "response": "Service temporarily unavailable. Please try again or consult source documents directly.",
            "sources": [],
            "category": QueryCategoryEnum.error.value,
            "model_used": "none",
            "latency_sec": 0.0,
            "user_id": user_id,
            "ai_generated": False,
            "disclaimer": "AI-generated content. Verify before use.",
            "cached": False,
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        if simulate_failure:
            print(f"--- Simulating API failure for query: '{query}' ---")
            formatted_response["response"] = "Simulated API failure: Research copilot temporarily unavailable."
            formatted_response["model_used"] = "fallback"
            formatted_response["latency_sec"] = round(time.time() - total_start_time, 2)
            # Log the simulated failure
            self.logger.log(formatted_response)
            return formatted_response

        try:
            # 1. Check Cache First
            cached_result = self.semantic_cache.get(query)
            if cached_result:
                formatted_response = cached_result
                formatted_response["user_id"] = user_id # Ensure current user_id is logged for cache hit
                self.logger.log(formatted_response)
                return formatted_response

            # 2. Route the Query
            route_info = self._route_query(query)
            category = route_info['category']
            routing_cost = route_info['routing_cost_usd']
            routing_latency = route_info['routing_latency_sec']
            routing_input_tokens = route_info['input_tokens']
            routing_output_tokens = route_info['output_tokens']

            # If routing itself failed, return an error
            if category == QueryCategoryEnum.error.value:
                formatted_response.update({
                    "response": f"Routing failed: {route_info['reasoning']}",
                    "category": QueryCategoryEnum.error.value,
                    "model_used": route_info['model_used_for_routing'],
                    "latency_sec": round(time.time() - total_start_time, 2),
                    "cost_usd": routing_cost,
                    "input_tokens": routing_input_tokens,
                    "output_tokens": routing_output_tokens,
                    "ai_generated": False,
                })
                return formatted_response # Logged in finally block

            # 3. Handle the Query based on Category
            handler_result = None
            if category == QueryCategoryEnum.data_lookup.value:
                handler_result = self._handle_data_lookup(query)
            elif category == QueryCategoryEnum.document_qa.value:
                handler_result = self._rag_answer(query)
            elif category == QueryCategoryEnum.research_agent.value:
                handler_result = self._run_agent(query)
            elif category == QueryCategoryEnum.general_knowledge.value:
                handler_result = self._handle_general_knowledge(query)
            else:
                raise ValueError(f"Unsupported query category after routing: {category}")

            # Combine routing and handler results
            combined_cost = routing_cost + handler_result['cost_usd']
            combined_latency = routing_latency + handler_result['latency_sec']

            # Aggregate token usage for logging
            total_input_tokens = routing_input_tokens + handler_result.get('input_tokens', 0)
            total_output_tokens = routing_output_tokens + handler_result.get('output_tokens', 0)

            formatted_response = {
                "query": query,
                "response": handler_result['response'],
                "sources": handler_result['sources'],
                "category": category,
                "model_used": handler_result['model_used'],
                "latency_sec": round(combined_latency, 2),
                "user_id": user_id,
                "ai_generated": True,
                "disclaimer": "AI-generated content. Verify before use.",
                "cached": False, # Not cached on this run
                "cost_usd": combined_cost,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            }

            # 4. Cache the successful response
            self.semantic_cache.put(query, formatted_response, user_id=user_id)

        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            formatted_response.update({
                "response": f"ERROR: An unexpected error occurred: {str(e)}",
                "category": QueryCategoryEnum.error.value,
                "model_used": "error",
                "latency_sec": round(time.time() - total_start_time, 2),
                "cost_usd": 0.0,
                "ai_generated": False,
                "disclaimer": "Error. Response not AI-generated.",
            })
        finally:
            # 5. Log the interaction (even errors)
            self.logger.log(formatted_response)

        return formatted_response

    def get_cost_summary(self, days: int = 7) -> List[Dict[str, Any]]:
        """Delegates to the compliance logger to get a cost summary."""
        return self.logger.get_cost_summary(days)

    def export_audit_report(self, user_id: Optional[str] = None, start_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Delegates to the compliance logger to export an audit report."""
        return self.logger.export_audit_report(user_id, start_date)

    def close(self):
        """Closes the database connection for the compliance logger."""
        self.logger.close()

# Example of how this module would be used in an app.py:
#
# from .your_module_name import FinancialCopilot
# import os
#
# openai_key = os.environ.get("OPENAI_API_KEY")
# if not openai_key:
#     raise ValueError("OPENAI_API_KEY environment variable not set.")
#
# # Initialize the copilot once in your application startup
# financial_copilot = FinancialCopilot(openai_api_key=openai_key, db_path='my_copilot_app.db')
#
# # In your API endpoint or function that handles user queries:
# def get_financial_insights(user_query: str, session_user_id: str):
#     response = financial_copilot.handle_query(user_query, user_id=session_user_id)
#     return response
#
# # Example usage:
# if __name__ == "__main__":
#     # This part simulates app.py usage and is usually not in the module itself.
#     # For demonstration, we use a placeholder API key and simulate calls.
#     # In a real app, you'd load from environment variable.
#     try:
#         copilot_instance = FinancialCopilot(openai_api_key="YOUR_OPENAI_KEY", db_path="app_example.db")
#         print("FinancialCopilot initialized.")
#
#         query1 = "What was Google's Q1 2024 revenue?"
#         res1 = copilot_instance.handle_query(query1, "test_user_01")
#         print(f"\nQuery: '{query1}'")
#         print(f"Response: {res1['response'][:100]}...")
#         print(f"Cost: ${res1['cost_usd']:.6f}, Latency: {res1['latency_sec']:.2f}s, Cached: {res1['cached']}")
#
#         query2 = "How much did Google make in the first quarter of 2024?"
#         res2 = copilot_instance.handle_query(query2, "test_user_01") # Should be a cache hit
#         print(f"\nQuery: '{query2}'")
#         print(f"Response: {res2['response'][:100]}...")
#         print(f"Cost: ${res2['cost_usd']:.6f}, Latency: {res2['latency_sec']:.2f}s, Cached: {res2['cached']}")
#
#         query3 = "Explain the efficient market hypothesis."
#         res3 = copilot_instance.handle_query(query3, "test_user_02")
#         print(f"\nQuery: '{query3}'")
#         print(f"Response: {res3['response'][:100]}...")
#         print(f"Cost: ${res3['cost_usd']:.6f}, Latency: {res3['latency_sec']:.2f}s, Cached: {res3['cached']}")
#
#         print("\nCost Summary (last 7 days):")
#         summary = copilot_instance.get_cost_summary(days=7)
#         for entry in summary:
#             print(entry)
#
#         copilot_instance.close()
#         print("\nCopilot instance closed.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         print("Please ensure your OPENAI_API_KEY is correctly set in environment variables or passed to the FinancialCopilot constructor.")
