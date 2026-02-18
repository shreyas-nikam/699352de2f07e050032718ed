import os
import json
import time
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel, Field, confloat, model_validator


# --- Global Constants (Configuration) ---

MODEL_COSTS = {
    "gpt-4o-mini": {
        "input_cost_per_1M_tokens": 0.15,
        "output_cost_per_1M_tokens": 0.60,
    },
    "gpt-4o": {
        "input_cost_per_1M_tokens": 5.00,
        "output_cost_per_1M_tokens": 15.00,
    },
    "none": {
        "input_cost_per_1M_tokens": 0.0,
        "output_cost_per_1M_tokens": 0.0,
    },
    "error": {
        "input_cost_per_1M_tokens": 0.0,
        "output_cost_per_1M_tokens": 0.0,
    },
    "cache": {
        "input_cost_per_1M_tokens": 0.0,
        "output_cost_per_1M_tokens": 0.0,
    },
}

HANDLER_OVERHEAD_COSTS = {
    "data_lookup": 0.001,
    "document_qa": 0.005,
    "research_agent": 0.010,
    "general_knowledge": 0.000,
    "routing": 0.000,
    "cache_hit": 0.0001,
}

QUERY_CATEGORIES = [
    "data_lookup",
    "document_qa",
    "general_knowledge",
    "research_agent",
]

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

SEMANTIC_EMBEDDING_MODEL = "text-embedding-3-small"


# --- Pydantic Models for LLM Structured Output ---

class QueryCategoryEnum(str, Enum):
    data_lookup = "data_lookup"
    document_qa = "document_qa"
    general_knowledge = "general_knowledge"
    research_agent = "research_agent"
    error = "error"

class QueryRoute(BaseModel):
    category: QueryCategoryEnum
    confidence: confloat(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)

    @model_validator(mode="after")
    def sanity(self):
        if self.confidence > 0.99 and len(self.reasoning.strip()) < 10:
            raise ValueError("If confidence is extremely high, provide non-trivial reasoning.")
        return self


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


# --- Core Functions: Query Routing and Handling ---

def route_query(query: str, openai_client: OpenAI) -> Dict[str, Any]:
    """
    Routes a financial research query to the appropriate handler using a lightweight LLM.
    Uses structured outputs by defining a Pydantic model.
    """
    try:
        start_time = time.time()

        response = openai_client.chat.completions.parse(
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
        # In a production module, error logging would go to a proper logging system
        # print(f"Error routing query '{query}': {e}")
        return {
            "category": "error",
            "confidence": 0.0,
            "reasoning": f"Routing failed: {e}",
            "routing_cost_usd": 0.0,
            "model_used_for_routing": "none",
            "routing_latency_sec": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def handle_data_lookup(query: str) -> Dict[str, Any]:
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

def rag_answer(query: str) -> Dict[str, Any]:
    """Simulates a RAG pipeline using a more expensive LLM (gpt-4o)."""
    start_time = time.time()
    prompt_tokens = 500
    completion_tokens = 200

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

def run_agent(query: str) -> Dict[str, Any]:
    """Simulates an agentic workflow using a powerful LLM (gpt-4o) and tools."""
    start_time = time.time()
    prompt_tokens = 1000
    completion_tokens = 400

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

def handle_general_knowledge(query: str, openai_client: OpenAI) -> Dict[str, Any]:
    """Uses a general-purpose, cost-effective LLM (gpt-4o-mini) for general knowledge."""
    start_time = time.time()
    try:
        response_llm = openai_client.chat.completions.create(
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


# --- Semantic Cache ---

class SemanticCache:
    """
    A conceptual semantic cache for storing and retrieving query responses
    based on embedding similarity and time-based expiry.
    """
    def __init__(self, embedding_model_name: str, similarity_threshold: float = 0.90, max_age_hours: int = 1, openai_client: OpenAI = None):
        if openai_client is None:
            raise ValueError("OpenAI client must be provided to SemanticCache.")
        self.openai_client = openai_client
        self.embedding_model_name = embedding_model_name
        self.similarity_threshold = similarity_threshold
        self.max_age_seconds = max_age_hours * 3600
        self.cache: List[Dict[str, Any]] = []

    def _get_embedding(self, text: str) -> List[float]:
        """Generates an embedding for the given text using OpenAI's embedding model."""
        try:
            response = self.openai_client.embeddings.create(
                input=[text],
                model=self.embedding_model_name
            )
            return response.data[0].embedding
        except Exception as e:
            # print(f"Error generating embedding for text: '{text[:50]}...' - {e}")
            return []

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Checks if a semantically similar query was recently answered and returns the cached entry.
        Returns None if no hit or if the entry is too old.
        """
        if not self.cache:
            return None

        query_emb = self._get_embedding(query)
        if not query_emb:
            return None

        query_emb_np = np.array(query_emb)
        now = datetime.now().timestamp()

        self.cache = [entry for entry in self.cache if now - entry['timestamp'] < self.max_age_seconds]

        best_match = None
        max_similarity = 0.0

        for entry in self.cache:
            entry_emb_np = np.array(entry['embedding'])
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
                "latency_sec": round(np.random.uniform(0.05, 0.2), 2),
                "user_id": best_match['user_id'],
                "ai_generated": True,
                "disclaimer": "AI-generated content. Verify before use. (Cached)",
                "cached": True,
                "cache_hit_similarity": float(max_similarity),
                "cost_usd": HANDLER_OVERHEAD_COSTS["cache_hit"],
                "input_tokens": 0,
                "output_tokens": 0,
            }
            return response_data
        return None

    def put(self, query: str, response_data: Dict[str, Any], user_id: str = "analyst_01"):
        """Stores a query-response pair in the cache along with its embedding."""
        embedding = self._get_embedding(query)
        if not embedding:
            return

        now = datetime.now().timestamp()

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
        self.cache = [entry for entry in self.cache if now - entry['timestamp'] < self.max_age_seconds]


# --- Compliance Logger ---

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
                interaction_data.get('category', 'unknown'),
                interaction_data.get('model_used', 'unknown'),
                interaction_data.get('input_tokens', 0),
                interaction_data.get('output_tokens', 0),
                interaction_data.get('cost_usd', 0.0),
                interaction_data.get('latency_sec', 0.0),
                1 if interaction_data.get('cached', False) else 0
            ))
            self.conn.commit()
        except Exception as e:
            # print(f"Error logging interaction: {e}. Data: {interaction_data}")
            pass # Fail gracefully in logging to not block main process

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
        query = "SELECT * FROM interactions WHERE 1=1"
        params = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        cursor = self.conn.execute(query, params)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self):
        """Closes the database connection."""
        self.conn.close()


# --- Main Entry Point for the Research Copilot ---

def handle_query(
    query: str,
    user_id: str,
    openai_client: OpenAI,
    semantic_cache: SemanticCache,
    compliance_logger: ComplianceLogger,
    simulate_failure: bool = False
) -> Dict[str, Any]:
    """
    Main entry point for the research copilot: routes, processes, caches, logs, and returns response.
    Includes fallback handling.
    """
    total_start_time = time.time()

    formatted_response = {
        "query": query,
        "response": "Service temporarily unavailable. Please try again or consult source documents directly.",
        "sources": [],
        "category": "error",
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
        # print(f"--- Simulating API failure for query: '{query}' ---")
        formatted_response["response"] = "Simulated API failure: Research copilot temporarily unavailable."
        formatted_response["model_used"] = "fallback"
        formatted_response["latency_sec"] = round(time.time() - total_start_time, 2)
        compliance_logger.log(formatted_response)
        return formatted_response

    try:
        # 1. Check Cache First
        cached_result = semantic_cache.get(query)
        if cached_result:
            formatted_response = cached_result
            formatted_response["user_id"] = user_id
            compliance_logger.log(formatted_response)
            return formatted_response

        # 2. Route the Query
        route_info = route_query(query, openai_client)
        category = route_info['category']
        routing_cost = route_info['routing_cost_usd']
        routing_latency = route_info['routing_latency_sec']
        routing_input_tokens = route_info['input_tokens']
        routing_output_tokens = route_info['output_tokens']

        handler_result = None
        if category == "data_lookup":
            handler_result = handle_data_lookup(query)
        elif category == "document_qa":
            handler_result = rag_answer(query)
        elif category == "research_agent":
            handler_result = run_agent(query)
        elif category == "general_knowledge":
            handler_result = handle_general_knowledge(query, openai_client)
        else:
            raise ValueError(f"Unknown query category: {category}")

        combined_cost = routing_cost + handler_result['cost_usd']
        combined_latency = routing_latency + handler_result['latency_sec']

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
            "cached": False,
            "cost_usd": combined_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        }

        # 3. Cache the successful response
        semantic_cache.put(query, formatted_response, user_id=user_id)

    except Exception as e:
        # print(f"Error processing query '{query}': {e}")
        formatted_response.update({
            "response": f"ERROR: An unexpected error occurred: {str(e)}",
            "category": "error",
            "model_used": "error",
            "latency_sec": round(time.time() - total_start_time, 2),
            "cost_usd": 0.0,
            "ai_generated": False,
            "disclaimer": "Error. Response not AI-generated.",
        })
    finally:
        # 4. Log the interaction (even errors)
        compliance_logger.log(formatted_response)

    return formatted_response


# --- Reporting Functions (can be put in a separate reporting module) ---
def generate_cost_optimization_report(df_interactions: pd.DataFrame):
    """Generates and displays cost optimization insights and visualizations."""
    if df_interactions.empty:
        print("No interactions logged. Cannot generate reports.")
        return

    df_interactions['cost_usd'] = pd.to_numeric(df_interactions['cost_usd'], errors='coerce').fillna(0)
    df_interactions['latency_sec'] = pd.to_numeric(df_interactions['latency_sec'], errors='coerce').fillna(0)
    df_interactions['cached'] = pd.to_numeric(df_interactions['cached'], errors='coerce').fillna(0).astype(bool)
    df_interactions['timestamp'] = pd.to_datetime(df_interactions['timestamp'])
    df_interactions['date'] = df_interactions['timestamp'].dt.date

    total_routed_cost = df_interactions['cost_usd'].sum()
    total_queries_processed = len(df_interactions)
    total_cached_queries = df_interactions['cached'].sum()

    print("\n" + "="*60)
    print("              COST OPTIMIZATION REPORT - Apex Capital Management")
    print("="*60)
    print(f"Total Queries Processed: {total_queries_processed}")
    print(f"Queries served from Cache: {total_cached_queries} ({total_cached_queries/total_queries_processed:.1%} hit rate)" if total_queries_processed > 0 else "N/A")
    print(f"Total Cost (Routed + Cached): ${total_routed_cost:.4f}")
    print(f"Average Cost per Query (Routed + Cached): ${total_routed_cost / total_queries_processed:.4f}" if total_queries_processed > 0 else "N/A")

    avg_prompt_tokens_gpt4o = 700
    avg_completion_tokens_gpt4o = 300
    gpt4o_cost_per_query = (avg_prompt_tokens_gpt4o / 1_000_000) * MODEL_COSTS["gpt-4o"]["input_cost_per_1M_tokens"] + \
                           (avg_completion_tokens_gpt4o / 1_000_000) * MODEL_COSTS["gpt-4o"]["output_cost_per_1M_tokens"] + \
                           HANDLER_OVERHEAD_COSTS["document_qa"]
    total_gpt4o_baseline_cost = total_queries_processed * gpt4o_cost_per_query
    print(f"Total Cost (Baseline - All to GPT-4o): ${total_gpt4o_baseline_cost:.4f}")

    cost_savings = total_gpt4o_baseline_cost - total_routed_cost
    percentage_savings = (cost_savings / total_gpt4o_baseline_cost) * 100 if total_gpt4o_baseline_cost > 0 else 0
    print(f"Simulated Cost Savings: ${cost_savings:.4f} ({percentage_savings:.1f}%)")

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

    cache_hits_df = df_interactions[df_interactions['cached'] == True]
    cache_misses_df = df_interactions[df_interactions['cached'] == False]

    avg_non_cached_query_cost = cache_misses_df['cost_usd'].mean() if not cache_misses_df.empty else gpt4o_cost_per_query * 0.5
    estimated_cache_savings = total_cached_queries * avg_non_cached_query_cost

    cache_hit_rate = total_cached_queries / total_queries_processed if total_queries_processed > 0 else 0

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Value (Rate)', color=color)
    bars1 = ax1.bar('Cache Hit Rate', cache_hit_rate, color=color, label='Cache Hit Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Value (USD)', color=color)
    bars2 = ax2.bar('Estimated Cost Savings from Cache', estimated_cache_savings, color=color, label='Estimated Cache Savings')
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    plt.title('Cache Performance: Hit Rate and Estimated Cost Savings')
    plt.show()

    plt.figure(figsize=(12, 7))
    sns.histplot(data=df_interactions, x='latency_sec', hue='category', multiple='stack', bins=20, kde=True, palette='Spectral')
    plt.title('Distribution of Response Times by Query Category')
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Number of Queries')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def generate_compliance_audit_report_sample(df_interactions: pd.DataFrame, num_samples: int = 5):
    """Displays a sample of the audit log."""
    if df_interactions.empty:
        print("No interactions logged. Cannot generate audit report sample.")
        return

    print("\n" + "="*60)
    print("           COMPLIANCE AUDIT REPORT SAMPLE - Apex Capital Management")
    print("="*60)

    sample_audit_log = df_interactions.sample(min(num_samples, len(df_interactions))).to_dict(orient='records')
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


# --- Initialization function for app.py ---

def initialize_copilot_components(
    openai_api_key: Optional[str] = None,
    db_path: str = 'copilot_compliance.db',
    cache_similarity_threshold: float = 0.90,
    cache_max_age_hours: int = 1
) -> Dict[str, Any]:
    """
    Initializes and returns all necessary components for the Research Copilot.
    This function should be called once when the application starts.
    """
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key is None or openai_api_key == "YOUR_OPENAI_KEY":
            raise ValueError("OPENAI_API_KEY must be provided either directly or via environment variable.")

    openai_client = OpenAI(api_key=openai_api_key)
    semantic_cache = SemanticCache(
        embedding_model_name=SEMANTIC_EMBEDDING_MODEL,
        similarity_threshold=cache_similarity_threshold,
        max_age_hours=cache_max_age_hours,
        openai_client=openai_client
    )
    compliance_logger = ComplianceLogger(db_path=db_path)

    return {
        "openai_client": openai_client,
        "semantic_cache": semantic_cache,
        "compliance_logger": compliance_logger
    }


# --- Simulation Runner (for testing/demonstration, not part of core module functions) ---

def run_simulation_and_reports(num_simulated_queries: int = 20, db_path: str = 'copilot_compliance.db'):
    """
    Runs a full simulation, processes queries, and generates reports.
    This function demonstrates the system's capabilities outside of a web app.
    """
    try:
        # Initialize components for simulation
        components = initialize_copilot_components(db_path=db_path)
        openai_client = components["openai_client"]
        semantic_cache = components["semantic_cache"]
        compliance_logger = components["compliance_logger"]

        print("\n--- Simulating Large Batch of Queries for Performance Analysis ---")
        simulated_queries_data = []

        categories = list(QUERY_DISTRIBUTION_PROFILE.keys())
        probabilities = list(QUERY_DISTRIBUTION_PROFILE.values())

        for i in range(num_simulated_queries):
            chosen_category = np.random.choice(categories, p=probabilities)

            query_to_use = np.random.choice(SYNTHETIC_QUERIES[chosen_category])
            # Small chance to pick a semantically similar query for cache testing
            if np.random.rand() < 0.2:
                for original_q, similar_q in SEMANTIC_CACHE_TEST_PAIRS:
                    if original_q in SYNTHETIC_QUERIES.get(chosen_category, []) or similar_q in SYNTHETIC_QUERIES.get(chosen_category, []):
                        if np.random.rand() < 0.5: # 50% chance to use the similar one
                            query_to_use = similar_q
                        else:
                            query_to_use = original_q
                        break


            user_id = f"analyst_{np.random.randint(1, 10):02d}"
            simulated_queries_data.append((query_to_use, user_id))

        all_responses = []
        for i, (query, user_id) in enumerate(simulated_queries_data):
            if i % 10 == 0:
                print(f"Processing query {i+1}/{num_simulated_queries}...")
            response_data = handle_query(
                query, user_id, openai_client, semantic_cache, compliance_logger,
                simulate_failure=(i == num_simulated_queries // 2 and num_simulated_queries > 0)
            )
            all_responses.append(response_data)

        print("\n--- Simulation Complete. Generating Reports ---")

        all_interactions_raw = compliance_logger.export_audit_report()
        df_interactions = pd.DataFrame(all_interactions_raw)

        generate_cost_optimization_report(df_interactions)
        generate_compliance_audit_report_sample(df_interactions)

    except Exception as e:
        print(f"An error occurred during simulation: {e}")
    finally:
        if 'compliance_logger' in locals() and compliance_logger:
            compliance_logger.close()
            print(f"\nComplianceLogger database '{db_path}' connection closed.")


if __name__ == '__main__':
    # This block runs only when the script is executed directly, not when imported.
    # It demonstrates the full simulation and reporting as in the original notebook.
    print("Running simulation and generating reports...")
    run_simulation_and_reports(num_simulated_queries=50) # Adjust count for faster/longer simulation
