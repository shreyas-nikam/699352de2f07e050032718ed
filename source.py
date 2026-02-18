import os
import json
import time
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Literal
from enum import Enum

# OpenAI specific imports
from openai import OpenAI
from pydantic import BaseModel, Field, confloat, model_validator

# Import for structured output parsing (assuming 'instructor' or similar is used).
# If 'instructor' is not installed, the `parse` method will not be available.
# We'll provide a fallback, but structured routing will then rely on manual parsing
# or default to error handling.
try:
    from instructor import Instructor
    # Patch the OpenAI client to add structured output capabilities
    _patched_openai_client_class = Instructor.from_openai(OpenAI)
except ImportError:
    # Fallback to standard OpenAI client if instructor is not available.
    # Note: `client.chat.completions.parse` will not work with the standard client.
    # The routing logic will need to handle this gracefully (e.g., manual JSON parsing or error).
    _patched_openai_client_class = OpenAI
    print("Warning: 'instructor' library not found. Structured output parsing (client.chat.completions.parse) will not work.")
    print("Routing will fall back to basic error handling if the expected `parse` method is called.")


# --- Global Constants (Module-level, used across the system) ---

# Define synthetic model costs per 1M tokens (as per OpenAI pricing)
# Pricing as of current date, subject to change.
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
}

# Define costs for different handler types (simulated overheads or specific model usage)
HANDLER_OVERHEAD_COSTS = {
    "data_lookup": 0.001, # Cost for external API call, minimal
    "document_qa": 0.005, # Cost for RAG retrieval and processing
    "research_agent": 0.010, # Cost for agent orchestration and tool usage
    "general_knowledge": 0.000, # Handled directly by cheap LLM, minimal overhead
    "routing": 0.000, # Routing cost is already token-based, no additional overhead per query
    "cache_hit": 0.000, # No LLM call, minimal processing cost
}

# Define query categories
QUERY_CATEGORIES = [
    "data_lookup",
    "document_qa",
    "general_knowledge",
    "research_agent",
]

# System and prompt for the query router LLM
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

# Embedding model for semantic cache
SEMANTIC_EMBEDDING_MODEL = "text-embedding-3-small"


# --- Pydantic Models for Structured Output ---

class QueryCategory(BaseModel):
    """Pydantic model for general query category classification."""
    category: str = Field(..., description="One of the predefined query categories: data_lookup, document_qa, general_knowledge, research_agent.")
    confidence: float = Field(..., description="A confidence score for the classification, between 0.0 and 1.0.")
    reasoning: str = Field(..., description="Brief explanation for the classification.")

class QueryCategoryEnum(str, Enum):
    """Enum for valid query categories."""
    data_lookup = "data_lookup"
    document_qa = "document_qa"
    general_knowledge = "general_knowledge"
    research_agent = "research_agent"
    error = "error" # Added for routing failures

class QueryRoute(BaseModel):
    """Pydantic model for structured output from the LLM router."""
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
            # In a production app, you might log this error to a monitoring system
            # rather than printing to stdout.
            # print(f"Error logging interaction: {e}")
            # print(f"Problematic data: {interaction_data}")
            pass # Fail silently in log method

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
        query_str = "SELECT * FROM interactions WHERE 1=1"
        params = []
        if user_id:
            query_str += " AND user_id = ?"
            params.append(user_id)
        if start_date:
            query_str += " AND timestamp >= ?"
            params.append(start_date)

        cursor = self.conn.execute(query_str, params)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()


# --- Semantic Cache Class ---
class SemanticCache:
    """
    A conceptual semantic cache for storing and retrieving query responses
    based on embedding similarity and time-based expiry.
    """
    def __init__(self, openai_client: OpenAI, embedding_model_name: str, similarity_threshold: float = 0.95, max_age_hours: int = 24):
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
            # print(f"Error generating embedding for text: '{text[:50]}...' - {e}")
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
        if not self.cache: # If cache becomes empty after pruning
            return None

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
            # Reconstruct cache hit response for consistency with other handlers
            response_data = {
                "query": query,
                "response": best_match['response']['response'],
                "sources": best_match['response']['sources'],
                "category": best_match['response']['category'],
                "model_used": "cache",
                "latency_sec": round(np.random.uniform(0.05, 0.2), 2), # Very fast from cache
                "user_id": best_match['user_id'], # Original user_id who put it in cache
                "ai_generated": True,
                "disclaimer": "AI-generated content. Verify before use.",
                "cached": True,
                "cache_hit_similarity": float(max_similarity),
                "cost_usd": HANDLER_OVERHEAD_COSTS["cache_hit"], # Minimal cost for cache lookup
                "input_tokens": 0, # No LLM tokens for cache hit
                "output_tokens": 0, # No LLM tokens for cache hit
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
            'response': cached_response_content, # Store the necessary response data
            'timestamp': now,
            'user_id': user_id,
        })
        # Prune old entries after adding new one
        self.cache = [entry for entry in self.cache if now - entry['timestamp'] < self.max_age_seconds]


# --- Main Financial Research Copilot Class ---
class FinancialResearchCopilot:
    """
    Orchestrates the routing, processing, caching, and logging of financial research queries.
    """
    def __init__(self, api_key: Optional[str] = None, db_path: str = 'copilot_compliance.db',
                 cache_threshold: float = 0.90, cache_max_age_hours: int = 1):
        """
        Initializes the FinancialResearchCopilot.

        Args:
            api_key (Optional[str]): OpenAI API key. If None, it will attempt to read from OPENAI_API_KEY environment variable.
            db_path (str): Path to the SQLite database for compliance logging.
            cache_threshold (float): Cosine similarity threshold for cache hits (0.0 to 1.0).
            cache_max_age_hours (int): Maximum age (in hours) for cached entries.
        """
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY not found. Please set the environment variable or pass it to the constructor.")

        # Initialize the OpenAI client (potentially patched by instructor)
        self.client = _patched_openai_client_class(api_key=api_key)
        self.compliance_logger = ComplianceLogger(db_path=db_path)
        self.semantic_cache = SemanticCache(self.client, SEMANTIC_EMBEDDING_MODEL,
                                            similarity_threshold=cache_threshold,
                                            max_age_hours=cache_max_age_hours)

        # Map query categories to their respective handler methods
        self.handlers = {
            "data_lookup": self._handle_data_lookup,
            "document_qa": self._rag_answer,
            "research_agent": self._run_agent,
            "general_knowledge": self._handle_general_knowledge,
        }

    def _route_query(self, query: str) -> Dict[str, Any]:
        """
        Routes a financial research query to the appropriate handler using a lightweight LLM.
        Uses structured outputs by defining a Pydantic model.
        """
        try:
            start_time = time.time()

            # Attempt structured completion with `parse` method
            response = self.client.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM},
                    {"role": "user", "content": ROUTER_PROMPT.format(query=query)},
                ],
                response_format=QueryRoute, # Requires instructor.patch to work
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
            # print(f"Error routing query '{query}': {e}")
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
        # Simulate token usage for a RAG response with gpt-4o
        prompt_tokens = 500
        completion_tokens = 200

        input_cost = (prompt_tokens / 1_000_000) * MODEL_COSTS["gpt-4o"]["input_cost_per_1M_tokens"]
        output_cost = (completion_tokens / 1_000_000) * MODEL_COSTS["gpt-4o"]["output_cost_per_1M_tokens"]
        llm_cost_usd = input_cost + output_cost

        mock_response = f"Simulated: Comprehensive answer for '{query}' based on internal documents. [Source: Internal 10-K filings, Earnings Call Transcripts]"
        mock_sources = [{"type": "document", "id": "doc_123", "page": "5"}, {"type": "document", "id": "earnings_q4", "timestamp": "2023-10-25"}]

        cost_usd = HANDLER_OVERHEAD_COSTS["document_qa"] + llm_cost_usd
        latency_sec = round(time.time() - start_time + np.random.uniform(1.0, 3.0), 2) # RAG typically longer
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
        # Simulate higher token usage for agentic workflow with gpt-4o
        prompt_tokens = 1000
        completion_tokens = 400

        input_cost = (prompt_tokens / 1_000_000) * MODEL_COSTS["gpt-4o"]["input_cost_per_1M_tokens"]
        output_cost = (completion_tokens / 1_000_000) * MODEL_COSTS["gpt-4o"]["output_cost_per_1M_tokens"]
        llm_cost_usd = input_cost + output_cost

        mock_response = f"Simulated: Detailed multi-step analysis for '{query}' using agentic workflow. [Trace: Data Fetch, Analysis, Synthesis]"
        mock_sources = [{"type": "agent_trace", "steps": 3, "tools_used": ["market_data", "news_sentiment"]}]

        cost_usd = HANDLER_OVERHEAD_COSTS["research_agent"] + llm_cost_usd
        latency_sec = round(time.time() - start_time + np.random.uniform(3.0, 7.0), 2) # Agents are typically slowest
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
            # print(f"Error handling general knowledge query '{query}': {e}")
            return {
                "response": "Error processing general knowledge query.",
                "sources": [],
                "model_used": "error",
                "cost_usd": 0.0,
                "latency_sec": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

    def process_query(self, query: str, user_id: str = "analyst_01", simulate_failure: bool = False) -> Dict[str, Any]:
        """
        Main entry point for the research copilot: routes, processes, caches, logs, and returns response.
        Includes fallback handling.

        Args:
            query (str): The user's financial research query.
            user_id (str): Identifier for the user making the query.
            simulate_failure (bool): If True, simulates an API failure to test fallback.

        Returns:
            Dict[str, Any]: A dictionary containing the response, sources, cost, latency, etc.
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
            "ai_generated": False, # Assume not AI-generated if it's an error
            "disclaimer": "AI-generated content. Verify before use.",
            "cached": False,
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        if simulate_failure:
            # print(f"--- Simulating API failure for query: '{query}' ---")
            formatted_response["response"] = "Simulated API failure: Research copilot temporarily unavailable."
            formatted_response["model_used"] = "fallback"
            formatted_response["latency_sec"] = round(time.time() - total_start_time, 2)
            self.compliance_logger.log(formatted_response)
            return formatted_response

        try:
            # 1. Check Cache First
            cached_result = self.semantic_cache.get(query)
            if cached_result:
                formatted_response = cached_result # The get method already formats it
                formatted_response["user_id"] = user_id # Ensure current user_id is logged for the cache hit
                # Log cache hit
                self.compliance_logger.log(formatted_response)
                return formatted_response

            # 2. Route the Query
            route_info = self._route_query(query)
            category = route_info['category']
            routing_cost = route_info['routing_cost_usd']
            routing_latency = route_info['routing_latency_sec']
            routing_input_tokens = route_info['input_tokens']
            routing_output_tokens = route_info['output_tokens']

            # Check if routing resulted in an error
            if category == QueryCategoryEnum.error.value:
                raise Exception(f"Query routing failed: {route_info['reasoning']}")

            handler_result = None
            if category in self.handlers:
                handler_result = self.handlers[category](query)
            else:
                raise ValueError(f"Unknown query category or unhandled category: {category}")

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

            # 3. Cache the successful response
            self.semantic_cache.put(query, formatted_response, user_id=user_id)

        except Exception as e:
            # print(f"Error processing query '{query}': {e}")
            formatted_response.update({
                "response": f"ERROR: An unexpected error occurred: {str(e)}",
                "category": QueryCategoryEnum.error.value,
                "model_used": "error",
                "latency_sec": round(time.time() - total_start_time, 2),
                "cost_usd": 0.0, # No cost incurred for a failed query, or minimal logging cost
                "ai_generated": False,
                "disclaimer": "Error. Response not AI-generated.",
            })
        finally:
            # 4. Log the interaction (even errors)
            self.compliance_logger.log(formatted_response)

        return formatted_response

    def close(self):
        """Closes resources (e.g., database connection for the logger)."""
        self.compliance_logger.close()


# --- Reporting and Visualization Function ---
def generate_and_display_reports(compliance_logger: ComplianceLogger):
    """
    Generates and displays various performance and cost optimization reports
    based on the interactions logged by the ComplianceLogger.
    This function is intended for standalone analysis or during development/demo,
    not typically part of an imported application's core logic.

    Args:
        compliance_logger (ComplianceLogger): An instance of the ComplianceLogger
                                              containing logged interactions.
    """
    all_interactions_raw = compliance_logger.export_audit_report()
    df_interactions = pd.DataFrame(all_interactions_raw)

    if not df_interactions.empty:
        df_interactions['cost_usd'] = pd.to_numeric(df_interactions['cost_usd'], errors='coerce').fillna(0)
        df_interactions['latency_sec'] = pd.to_numeric(df_interactions['latency_sec'], errors='coerce').fillna(0)
        df_interactions['cached'] = pd.to_numeric(df_interactions['cached'], errors='coerce').fillna(0).astype(bool)
        df_interactions['timestamp'] = pd.to_datetime(df_interactions['timestamp'])
    else:
        print("No interactions logged. Cannot generate reports.")
        # Initialize an empty DataFrame with required columns if no data
        df_interactions = pd.DataFrame(columns=['timestamp', 'category', 'cost_usd', 'latency_sec', 'cached', 'model', 'query', 'response', 'user_id', 'input_tokens', 'output_tokens'])
        return # Exit if no data to report

    # --- Cost Optimization Report ---
    print("\n" + "="*60)
    print("              COST OPTIMIZATION REPORT - Apex Capital Management")
    print("="*60)

    total_routed_cost = df_interactions['cost_usd'].sum()
    total_queries_processed = len(df_interactions)
    total_cached_queries = df_interactions['cached'].sum()

    print(f"Total Queries Processed: {total_queries_processed}")
    print(f"Queries served from Cache: {total_cached_queries} ({total_cached_queries/total_queries_processed:.1%} hit rate)")
    print(f"Total Cost (Routed + Cached): ${total_routed_cost:.4f}")
    print(f"Average Cost per Query (Routed + Cached): ${total_routed_cost / total_queries_processed:.4f}")

    # Calculate "all-to-GPT-4o" baseline cost for comparison
    avg_prompt_tokens_gpt4o = 700
    avg_completion_tokens_gpt4o = 300
    gpt4o_cost_per_query = (avg_prompt_tokens_gpt4o / 1_000_000) * MODEL_COSTS["gpt-4o"]["input_cost_per_1M_tokens"] + \
                           (avg_completion_tokens_gpt4o / 1_000_000) * MODEL_COSTS["gpt-4o"]["output_cost_per_1M_tokens"] + \
                           HANDLER_OVERHEAD_COSTS["document_qa"] # Add some overhead

    total_gpt4o_baseline_cost = total_queries_processed * gpt4o_cost_per_query
    print(f"Total Cost (Baseline - All to GPT-4o): ${total_gpt4o_baseline_cost:.4f}")

    cost_savings = total_gpt4o_baseline_cost - total_routed_cost
    percentage_savings = (cost_savings / total_gpt4o_baseline_cost) * 100 if total_gpt4o_baseline_cost > 0 else 0
    print(f"Simulated Cost Savings: ${cost_savings:.4f} ({percentage_savings:.1f}%)")

    # Visualization 1: Comparison of Simulated API Costs
    costs_df = pd.DataFrame({
        'Scenario': ['Routed + Cached', 'Baseline (All to GPT-4o)'],
        'Total Cost ($)': [total_routed_cost, total_gpt4o_baseline_cost]
    })

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Scenario', y='Total Cost ($)', data=costs_df, palette='viridis')
    plt.title('Simulated API Costs: Routed + Cached vs. Baseline (All to GPT-4o)')
    plt.ylabel('Total Cost (USD)')
    plt.xlabel('Scenario')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Visualization 2: Daily API Spend by Query Category over a Simulated Week
    df_interactions['date'] = df_interactions['timestamp'].dt.date
    daily_spend = df_interactions.groupby(['date', 'category'])['cost_usd'].sum().unstack(fill_value=0)

    plt.figure(figsize=(12, 7))
    daily_spend.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Simulated Daily API Spend by Query Category')
    plt.ylabel('Cost (USD)')
    plt.xlabel('Date')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Visualization 3: Cache Hit Rate and Corresponding Cost Savings
    cache_hits_df = df_interactions[df_interactions['cached'] == True]
    cache_misses_df = df_interactions[df_interactions['cached'] == False]

    # Estimate the cost savings from cache: how much would it have cost if not cached?
    avg_non_cached_query_cost = cache_misses_df['cost_usd'].mean() if not cache_misses_df.empty else gpt4o_cost_per_query * 0.5 # Estimate if no misses happened
    estimated_cache_savings = total_cached_queries * avg_non_cached_query_cost

    cache_hit_rate = total_cached_queries / total_queries_processed if total_queries_processed > 0 else 0

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Value (Rate)', color=color)
    ax1.bar('Cache Hit Rate', cache_hit_rate, color=color, label='Cache Hit Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0,1)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Value (USD)', color=color)
    ax2.bar('Estimated Cost Savings from Cache', estimated_cache_savings, color=color, label='Estimated Cache Savings')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Cache Performance: Hit Rate and Estimated Cost Savings')
    plt.show()


    # Visualization 4: Histogram of Simulated Response Times by Query Category
    plt.figure(figsize=(12, 7))
    sns.histplot(data=df_interactions, x='latency_sec', hue='category', multiple='stack', bins=20, kde=True, palette='Spectral')
    plt.title('Distribution of Response Times by Query Category')
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Number of Queries')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


    # --- Compliance Audit Report Sample ---
    print("\n" + "="*60)
    print("           COMPLIANCE AUDIT REPORT SAMPLE - Apex Capital Management")
    print("="*60)

    # Display a sample of the audit log
    sample_audit_log = df_interactions.sample(min(5, len(df_interactions))).to_dict(orient='records')
    for i, entry in enumerate(sample_audit_log):
        print(f"\n--- Audit Entry {i+1} ---")
        print(f"Timestamp: {entry.get('timestamp')}")
        print(f"User ID: {entry.get('user_id')}")
        print(f"Query: {entry.get('query')[:100]}...")
        print(f"Response: {entry.get('response')[:150]}...")
        print(f"Category: {entry.get('category')}")
        print(f"Model Used: {entry.get('model')}")
        print(f"Cost (USD): ${entry.get('cost_usd'):.6f}")
        print(f"Latency (sec): {entry.get('latency_sec'):.2f}")
        print(f"Cached: {entry.get('cached')}")
        print(f"Sources: {entry.get('sources')}")
        print("-" * 20)

    print("\nCompliance logging is critical for financial firms using AI for research.")
    print("It enables supervisory review, error tracing, and regulatory examination.")


# --- Main execution block for simulation/testing (only runs when script is executed directly) ---
if __name__ == "__main__":
    # Define synthetic queries and distribution for simulation within the __main__ block
    # These are only for demonstrating the functionality and reports.
    SYNTHETIC_QUERIES = {
        "data_lookup": [
            "What was Google's Q1 2024 revenue?", "Current stock price of AAPL?", "S&P 500 YTD return?",
            "What is the market cap of Tesla?", "What was Microsoft's EPS for Q3 2023?",
            "Latest dividend yield for ExxonMobil?", "Current interest rate (Fed funds effective rate)?",
            "Gold price today?", "What is the historical revenue trend for Amazon (AMZN)?",
            "How much debt does NVIDIA (NVDA) have?",
        ],
        "document_qa": [
            "Summarize risk factors mentioned in Tesla's latest 10-K.",
            "What are the key takeaways from Apple's Q4 2023 earnings call transcript?",
            "Extract all mentions of 'supply chain disruptions' from the latest Intel annual report.",
            "Compare the capex plans of TSMC and Samsung from their recent financial statements.",
            "Identify competitive threats to Netflix from its most recent investor presentation.",
            "What are the regulatory risks for pharmaceutical companies according to Pfizer's latest filings?",
            "Summarize the growth strategy outlined in Salesforce's recent analyst day transcript.",
            "Find any mentions of 'AI investment' in Google's (GOOGL) 2023 annual report.",
            "What is the estimated market size for electric vehicles according to a recent industry report?",
            "Discuss ESG initiatives from JPMorgan's latest sustainability report.",
        ],
        "general_knowledge": [
            "Explain the Sharpe ratio.", "What is duration risk?", "Define EBITDA.",
            "What are the primary drivers of inflation?", "How does quantitative easing work?",
            "What is the efficient market hypothesis?", "Explain the concept of 'black swan' events in finance.",
            "What is a convertible bond?", "Describe the different types of derivatives.",
            "What is the role of a central bank?",
        ],
        "research_agent": [
            "Analyze Tesla's competitive position in the EV market.",
            "Compare valuation metrics for NVIDIA and AMD.",
            "Evaluate the investment case for Google (GOOGL) considering its cloud and AI segments.",
            "Research the impact of rising interest rates on the real estate sector.",
            "Conduct a sentiment analysis of recent news articles on Boeing (BA).",
            "Assess the long-term growth prospects of renewable energy companies.",
            "Perform a peer analysis of major payment processing companies like Visa and Mastercard.",
            "Investigate the regulatory landscape for cryptocurrencies in the US and Europe.",
            "Provide a comprehensive overview of the semiconductor industry outlook for 2024.",
            "Analyze the potential M&A targets in the biotech space, focusing on oncology.",
        ],
    }

    QUERY_DISTRIBUTION_PROFILE = {
        "data_lookup": 0.30,
        "general_knowledge": 0.25,
        "document_qa": 0.30,
        "research_agent": 0.15,
    }

    SEMANTIC_CACHE_TEST_PAIRS = [
        ("What was Google's Q1 2024 revenue?", "How much did Google make in the first quarter of 2024?"),
        ("Explain the Sharpe ratio.", "Can you define the Sharpe ratio for me?"),
        ("Analyze Tesla's competitive position in the EV market.", "What is Tesla's competitive landscape like in electric vehicles?"),
        ("Current stock price of AAPL?", "What's Apple's stock price right now?"),
        ("Summarize risk factors mentioned in Tesla's latest 10-K.", "Summarize the risks in Tesla's most recent 10-K filing."),
    ]


    print("Initializing Financial Research Copilot for testing...")
    # Initialize the copilot. The API key should be set in environment variables
    # or passed explicitly.
    try:
        # Use a placeholder API key if not set, for local testing without environment variable
        # For actual use, ensure OPENAI_API_KEY is set in your environment.
        copilot = FinancialResearchCopilot(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE"))
        print("Financial Research Copilot initialized successfully.")
    except ValueError as e:
        print(f"Error initializing copilot: {e}. Please ensure OPENAI_API_KEY is set.")
        exit(1) # Exit if API key is not available

    # --- Test routing with synthetic queries ---
    print("\n--- Testing Query Routing ---")
    test_queries_for_routing = [
        "What was Apple's Q4 revenue last quarter?",
        "Analyze JPMorgan's competitive position.",
        "What is a convertible bond?",
        "Current price of MSFT?",
        "Summarize risk factors mentioned in Tesla's latest 10-K.",
        "Explain the efficient market hypothesis.",
        "Evaluate the investment case for Google (GOOGL) considering its cloud and AI segments.",
        "Find any mentions of 'AI investment' in Google's (GOOGL) 2023 annual report."
    ]

    for q in test_queries_for_routing:
        route_info = copilot._route_query(q)
        print(f"Query: '{q}'")
        print(f"  -> Category: [{route_info['category']}] (Confidence: {route_info['confidence']:.2f})")
        print(f"  -> Reasoning: {route_info['reasoning']}")
        print(f"  -> Routing Cost: ${route_info['routing_cost_usd']:.6f} | Latency: {route_info['routing_latency_sec']:.2f}s")
        print("-" * 30)

    # --- Test handlers conceptually ---
    print("\n--- Testing Backend Handlers ---")
    sample_data_lookup_query = "What was Google's Q1 2024 revenue?"
    data_lookup_res = copilot._handle_data_lookup(sample_data_lookup_query)
    print(f"Data Lookup Handler for '{sample_data_lookup_query}':")
    print(f"  Response: {data_lookup_res['response'][:70]}...")
    print(f"  Cost: ${data_lookup_res['cost_usd']:.6f} | Latency: {data_lookup_res['latency_sec']:.2f}s | Model: {data_lookup_res['model_used']}")
    print("-" * 30)

    sample_general_knowledge_query = "Explain the Sharpe ratio."
    general_knowledge_res = copilot._handle_general_knowledge(sample_general_knowledge_query)
    print(f"General Knowledge Handler for '{sample_general_knowledge_query}':")
    print(f"  Response: {general_knowledge_res['response'][:70]}...")
    print(f"  Cost: ${general_knowledge_res['cost_usd']:.6f} | Latency: {general_knowledge_res['latency_sec']:.2f}s | Model: {general_knowledge_res['model_used']}")
    print("-" * 30)

    sample_document_qa_query = "Summarize risk factors mentioned in Tesla's latest 10-K."
    document_qa_res = copilot._rag_answer(sample_document_qa_query)
    print(f"Document Q&A Handler for '{sample_document_qa_query}':")
    print(f"  Response: {document_qa_res['response'][:70]}...")
    print(f"  Cost: ${document_qa_res['cost_usd']:.6f} | Latency: {document_qa_res['latency_sec']:.2f}s | Model: {document_qa_res['model_used']}")
    print("-" * 30)

    sample_research_agent_query = "Analyze Tesla's competitive position in the EV market."
    research_agent_res = copilot._run_agent(sample_research_agent_query)
    print(f"Research Agent Handler for '{sample_research_agent_query}':")
    print(f"  Response: {research_agent_res['response'][:70]}...")
    print(f"  Cost: ${research_agent_res['cost_usd']:.6f} | Latency: {research_agent_res['latency_sec']:.2f}s | Model: {research_agent_res['model_used']}")
    print("-" * 30)

    # --- Test semantic caching ---
    print("\n--- Testing Semantic Caching ---")
    query_original = "What was Google's Q1 2024 revenue?"
    query_similar = "How much did Google make in the first quarter of 2024?"
    query_unrelated = "Explain the Black-Scholes model."

    print(f"1. Processing original query: '{query_original}'")
    original_response_for_cache = copilot.process_query(query_original, user_id="analyst_01")
    print(f"   -> Cached original response. Cache size: {len(copilot.semantic_cache.cache)}")

    print(f"2. Retrieving similar query: '{query_similar}'")
    cached_hit = copilot.semantic_cache.get(query_similar)
    if cached_hit:
        print(f"   -> CACHE HIT! Response: {cached_hit['response'][:70]}... (Similarity: {cached_hit['cache_hit_similarity']:.2f})")
        print(f"   -> Cost: ${cached_hit['cost_usd']:.6f} | Latency: {cached_hit['latency_sec']:.2f}s | Model: {cached_hit['model_used']}")
    else:
        print("   -> Cache Miss.")

    print(f"3. Retrieving unrelated query: '{query_unrelated}'")
    cached_miss = copilot.semantic_cache.get(query_unrelated)
    if cached_miss:
        print(f"   -> CACHE HIT! Response: {cached_miss['response'][:70]}... (Similarity: {cached_miss['cache_hit_similarity']:.2f})")
    else:
        print("   -> Cache Miss. (Expected)")

    copilot.semantic_cache.cache = [] # Clear cache to simulate expiry for the next test
    print(f"4. Cache cleared to simulate expiry. Cache size: {len(copilot.semantic_cache.cache)}")


    # --- Test the full pipeline with a series of diverse queries ---
    print("\n--- Testing the Full Research Copilot Pipeline ---")
    test_pipeline_queries = [
        ("What was Google's Q1 2024 revenue?", "analyst_01"),
        ("Analyze Tesla's competitive position in the EV market.", "analyst_02"),
        ("Explain the Sharpe ratio.", "analyst_01"),
        ("Summarize risk factors mentioned in Tesla's latest 10-K.", "analyst_03"),
        ("How much did Google make in the first quarter of 2024?", "analyst_01"), # Similar to first query, testing cache hit
        ("What is the current price of NVDA?", "analyst_04"),
        ("What is duration risk?", "analyst_02"),
        ("Evaluate the investment case for Google (GOOGL) considering its cloud and AI segments.", "analyst_01"),
        ("Current stock price of AAPL?", "analyst_05"),
        ("Summarize the risks in Tesla's most recent 10-K filing.", "analyst_03"), # Similar to fourth query, testing cache hit
    ]

    for i, (query, user_id) in enumerate(test_pipeline_queries):
        print(f"\n--- Query {i+1} by {user_id}: '{query}' ---")
        response_data = copilot.process_query(query, user_id)
        print(f"  Final Response (Category: {response_data['category']}): {response_data['response'][:100]}...")
        print(f"  Cost: ${response_data['cost_usd']:.6f} | Latency: {response_data['latency_sec']:.2f}s | Model: {response_data['model_used']} | Cached: {response_data['cached']}")

    # --- Test fallback handling ---
    print("\n--- Testing Fallback Handling (Simulated Failure) ---")
    error_query = "What happens if the main LLM API is down?"
    error_response = copilot.process_query(error_query, "analyst_06", simulate_failure=True)
    print(f"  Final Response (Category: {error_response['category']}): {error_response['response'][:100]}...")
    print(f"  Cost: ${error_response['cost_usd']:.6f} | Latency: {error_response['latency_sec']:.2f}s | Model: {error_response['model_used']} | Cached: {error_response['cached']}")

    # --- Simulate a larger batch of queries for reporting ---
    print("\n--- Simulating Large Batch of Queries for Performance Analysis ---")
    num_simulated_queries = 20 # Can increase for more robust reporting data
    simulated_queries_data = []

    categories = list(QUERY_DISTRIBUTION_PROFILE.keys())
    probabilities = list(QUERY_DISTRIBUTION_PROFILE.values())

    for i in range(num_simulated_queries):
        chosen_category = np.random.choice(categories, p=probabilities)

        # To introduce more cache hits, we sometimes pick from semantic_cache_test_pairs
        if np.random.rand() < 0.2 and chosen_category in ["data_lookup", "general_knowledge", "document_qa"]:
            if chosen_category == "data_lookup":
                query_pair = SEMANTIC_CACHE_TEST_PAIRS[0] if np.random.rand() < 0.5 else SEMANTIC_CACHE_TEST_PAIRS[3]
            elif chosen_category == "general_knowledge":
                query_pair = SEMANTIC_CACHE_TEST_PAIRS[1]
            elif chosen_category == "document_qa":
                # Assuming SEMANTIC_CACHE_TEST_PAIRS has a doc_qa example
                query_pair = SEMANTIC_CACHE_TEST_PAIRS[2]
            else: # Fallback for categories not explicitly in test pairs
                query_pair = (np.random.choice(SYNTHETIC_QUERIES[chosen_category]),)
            query_to_use = query_pair[1] if len(query_pair) > 1 and np.random.rand() < 0.7 else query_pair[0] # Sometimes use the similar one
        else:
            query_to_use = np.random.choice(SYNTHETIC_QUERIES[chosen_category])

        user_id = f"analyst_{np.random.randint(1, 10):02d}" # Simulate 10 different analysts
        simulated_queries_data.append((query_to_use, user_id))

    all_responses = []
    for i, (query, user_id) in enumerate(simulated_queries_data):
        if i % 5 == 0: # Print less frequently for larger simulations
            print(f"Processing simulated query {i+1}/{num_simulated_queries}...")
        response_data = copilot.process_query(query, user_id, simulate_failure=(i == num_simulated_queries // 2 and num_simulated_queries > 0)) # Simulate one failure
        all_responses.append(response_data)

    print("\n--- Simulation Complete. Generating Reports ---")

    # Generate and display reports using the copilot's logger
    generate_and_display_reports(copilot.compliance_logger)

    # Close the copilot's resources
    copilot.close()
    print(f"\nFinancial Research Copilot resources closed.")
