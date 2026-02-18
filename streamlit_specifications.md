
## Streamlit Application Specification: Intelligent AI Research Copilot

### 1. Application Overview

**Purpose of the Application**

This Streamlit application serves as a **research copilot frontend** for financial analysts and CFA Charterholders at Apex Capital Management. Its primary purpose is to demonstrate a **cost-optimized and compliant AI research workflow**. It showcases how to intelligently route financial research queries to the most appropriate and cost-effective backend handler (e.g., direct API lookup, lightweight LLM, RAG pipeline, or multi-step agent), integrate semantic caching to reduce redundant LLM calls, and maintain a comprehensive compliance audit log for regulatory adherence. The application aims to provide a tangible example of how a firm can leverage AI for research while meticulously managing API costs and ensuring ethical and compliant AI deployment.

**High-Level Story Flow of the Application**

The application guides the user (an Apex Capital Management analyst or a Head of Research) through the following workflow:

1.  **Introduction and Context:** The user first encounters an overview page explaining the challenges of LLM costs and compliance in financial research, setting the stage for the intelligent copilot.
2.  **Interactive Research Copilot:** The core interaction begins on the "Research Copilot" page. Here, the user inputs a financial research query (e.g., "Current price of AAPL?", "Explain the Sharpe ratio.", "Analyze Tesla's competitive position").
3.  **Intelligent Processing:** Upon submission, the application, behind the scenes, performs the following:
    *   **Semantic Cache Check:** It first checks if a semantically similar query was recently answered. If so, it returns the cached response for speed and cost efficiency.
    *   **Query Routing:** If not cached, a lightweight LLM (e.g., `gpt-4o-mini`) classifies the query into "Data Lookup", "Document Q&A", "General Knowledge", or "Multi-step Research Agent".
    *   **Specialized Handling:** Based on the classification, the query is dispatched to the most appropriate backend handler (e.g., direct API call for data, RAG for documents, a general LLM for knowledge, or a powerful agentic LLM for complex tasks).
    *   **Fallback Handling:** The system gracefully handles potential LLM API failures, returning a user-friendly error message.
    *   **Compliance Logging:** Every interaction, regardless of outcome or source (cached, LLM, API), is meticulously logged into an SQLite database with detailed metadata (query, response, sources, model, cost, latency, tokens, user ID, timestamp).
4.  **Response Display:** The user receives the AI-generated response, along with insights into which category it was routed, which model was used, the cost incurred, and the latency. A clear disclaimer for AI-generated content is always included.
5.  **Performance & Compliance Dashboards:** The user can then navigate to dedicated dashboards to visualize:
    *   **Cost Efficiency:** A "Cost Monitoring Dashboard" displays simulated API spend comparisons (routed vs. baseline), daily spend by category, cache hit rates, and latency distributions, quantifying the cost savings.
    *   **Auditability:** A "Compliance Audit Log" presents a sample of recorded interactions, demonstrating the system's adherence to regulatory record-keeping requirements.

This flow effectively demonstrates how a sophisticated AI system can be built for financial professionals, focusing on practical deployment concerns like cost management, efficiency, and regulatory compliance, rather than just raw LLM capability.

### 2. Code Requirements

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import json

# Import all functions and global objects from source.py
from source import *

# Ensure logger is connected (it reconnects implicitly on __init__ if needed,
# and is globally instantiated when source.py is imported, but we make sure here)
# The `logger = ComplianceLogger()` and `semantic_cache = SemanticCache(...)` are already global
# objects from `source.py` when imported.

# --- Global Configuration and Session State Initialization ---

# Initialize session state variables
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Application Overview'
if 'user_id' not in st.session_state:
    st.session_state.user_id = "analyst_01" # Default user ID for persona
if 'query_history' not in st.session_state:
    st.session_state.query_history = [] # Stores list of dicts for past queries/responses
if 'simulate_failure' not in st.session_state:
    st.session_state.simulate_failure = False
if 'audit_filter_user_id' not in st.session_state:
    st.session_state.audit_filter_user_id = ""
if 'audit_filter_start_date' not in st.session_state:
    st.session_state.audit_filter_start_date = None
if 'cost_dashboard_days' not in st.session_state:
    st.session_state.cost_dashboard_days = 7

# --- Application Structure and Flow ---

st.sidebar.title("Navigation")
st.session_state.current_page = st.sidebar.selectbox(
    "Go to",
    ['Application Overview', 'Research Copilot', 'Cost Monitoring Dashboard', 'Compliance Audit Log']
)

# --- Page: Application Overview ---
if st.session_state.current_page == 'Application Overview':
    st.title("Intelligent AI Research Copilot for Apex Capital Management")
    
    st.markdown(f"As the Head of Research at `Apex Capital Management`, I'm constantly seeking ways to enhance our research efficiency while judiciously managing costs. The rapid adoption of AI research copilots in our firm presents both immense opportunities and a significant challenge: escalating API costs from powerful Large Language Models (LLMs). My team needs to ensure we're using the right tool for the right job, much like we assign human analysts tasks based on their complexity and specialization.")
    st.markdown(f"This application demonstrates the development of an \"Intelligent AI Query Router\" designed to optimize LLM resource allocation. It classifies incoming financial research queries and directs them to the most cost-effective and appropriate backend handler (e.g., a simple API call, a Retrieval Augmented Generation (RAG) pipeline, or a more complex agentic workflow). Beyond cost efficiency, this system incorporates crucial components for real-world deployment: robust compliance logging, semantic caching to reduce redundant calls, and basic fallback handling for improved resilience.")
    st.markdown(f"Our goal is to build a system that acts as a central nervous system for our AI research tools, ensuring that powerful, expensive LLMs like `gpt-4o` are reserved for truly complex analytical tasks, while simpler queries are handled by faster, cheaper methods like `gpt-4o-mini` or direct data lookups. This approach promises significant API cost reductions, improved latency for our analysts, and a solid foundation for regulatory compliance.")
    st.header("Key Components:")
    st.markdown(f"- **Intelligent Query Router**: Classifies financial queries to select the optimal processing method.")
    st.markdown(f"- **Diverse Backend Handlers**: Tailored processing for data lookups, document Q&A, general knowledge, and multi-step research.")
    st.markdown(f"- **Semantic Caching**: Reuses previous answers for semantically similar queries to save costs and reduce latency.")
    st.markdown(f"- **Compliance Logging**: Records every interaction for audit, regulatory, and supervisory review (CFA Standard V(C) – Record Retention).")
    st.markdown(f"- **Fallback Handling**: Ensures graceful degradation of service during API failures.")
    st.markdown(f"- **Cost Monitoring**: Provides dashboards to track and visualize API spend, demonstrating ROI.")

# --- Page: Research Copilot ---
elif st.session_state.current_page == 'Research Copilot':
    st.title("Engage Your Copilot: Smart Financial Research")
    st.markdown(f"As an analyst, simply type your financial research query below. The copilot will intelligently route it to the most efficient backend, whether it's a quick data lookup, a deep dive into documents, a general knowledge query, or a complex multi-step research task. Every interaction is optimized for cost and speed, and fully logged for compliance.")

    st.subheader("Your Query:")
    user_input_col, user_id_col = st.columns([3, 1])
    with user_id_col:
        st.session_state.user_id = st.text_input("Your User ID", st.session_state.user_id, key="current_user_id_input")
    with user_input_col:
        user_query = st.text_area("Enter your financial research query:", key="research_query_input", height=100)
    
    st.session_state.simulate_failure = st.checkbox("Simulate API Failure", value=st.session_state.simulate_failure)

    if st.button("Ask Copilot", key="ask_copilot_button"):
        if user_query:
            with st.spinner("Processing your request..."):
                response_data = handle_query(user_query, st.session_state.user_id, st.session_state.simulate_failure)
                st.session_state.query_history.append(response_data)
        else:
            st.warning("Please enter a query.")

    st.subheader("Recent Interactions:")
    if not st.session_state.query_history:
        st.info("No queries submitted yet. Your interactions will appear here.")
    
    # Display queries in reverse chronological order
    for i, res_data in enumerate(reversed(st.session_state.query_history)):
        st.markdown(f"---")
        st.markdown(f"**Query {len(st.session_state.query_history) - i}:** `{res_data['query']}` (by `{res_data['user_id']}`)")
        st.markdown(f"**Response:** {res_data['response']}")
        
        st.markdown(f"**Details:**")
        st.markdown(f"- **Category:** `{res_data['category'].replace('_', ' ').title()}`")
        if res_data.get('cached'):
            st.markdown(f"- **Source:** `Cache Hit` (Similarity: `{res_data.get('cache_hit_similarity', 'N/A'):.2f}`) ")
        else:
            st.markdown(f"- **Model Used:** `{res_data['model_used']}`")
        st.markdown(f"- **Cost:** `${res_data['cost_usd']:.6f}`")
        st.markdown(f"- **Latency:** `{res_data['latency_sec']:.2f}` seconds")
        if res_data['sources']:
            st.markdown(f"- **Sources:** {json.dumps(res_data['sources'])}")

        st.markdown(f"")
        st.markdown(f"*{res_data['disclaimer']}*")

    st.subheader("How the Router Saves Costs:")
    st.markdown(f"The intelligent query router uses a lightweight, cost-effective LLM (`gpt-4o-mini`) to classify your query. This prevents sending simple queries to more powerful, expensive LLMs like `gpt-4o` unnecessarily. The routing cost for a single query is calculated based on token usage and model prices:")
    st.markdown(r"$$ \text{{Routing Cost}} = \left( \frac{{\text{{Prompt Tokens}}}}{{10^6}} \times \text{{Prompt Cost/1M Tokens}} \right) + \left( \frac{{\text{{Completion Tokens}}}}{{10^6}} \times \text{{Completion Cost/1M Tokens}} \right) $$")
    st.markdown(r"where $\text{{Prompt Tokens}}$ and $\text{{Completion Tokens}}$ are the number of input and output tokens respectively, and $\text{{Prompt Cost/1M Tokens}}$ and $\text{{Completion Cost/1M Tokens}}$ are the per-million-token costs for the `gpt-4o-mini` model.")
    st.markdown(f"This two-tier architecture ensures that powerful, expensive LLMs are reserved for truly complex analytical tasks, while simpler queries are handled by faster, cheaper methods or direct data lookups.")

# --- Page: Cost Monitoring Dashboard ---
elif st.session_state.current_page == 'Cost Monitoring Dashboard':
    st.title("Cost Efficiency Dashboard")
    st.markdown(f"As the Head of Research, quantifying API cost savings is crucial for demonstrating the ROI of our AI initiatives. This dashboard provides a clear overview of our simulated API spend, showing how our intelligent routing and caching mechanisms deliver significant financial benefits. This aligns with effective **Technology Management** principles for CFA professionals.")
    
    st.session_state.cost_dashboard_days = st.slider("Select period for cost summary (days)", 1, 90, st.session_state.cost_dashboard_days)

    # Re-fetch data from the logger (source of truth) for up-to-date metrics
    all_interactions_raw = logger.export_audit_report()
    df_interactions = pd.DataFrame(all_interactions_raw)

    if df_interactions.empty:
        st.info("No interactions logged yet. Please use the 'Research Copilot' page to generate some data.")
    else:
        # Data preparation as in source.py
        df_interactions['cost_usd'] = pd.to_numeric(df_interactions['cost_usd'], errors='coerce').fillna(0)
        df_interactions['latency_sec'] = pd.to_numeric(df_interactions['latency_sec'], errors='coerce').fillna(0)
        df_interactions['cached'] = pd.to_numeric(df_interactions['cached'], errors='coerce').fillna(0).astype(bool)
        df_interactions['timestamp'] = pd.to_datetime(df_interactions['timestamp'])
        df_interactions['date'] = df_interactions['timestamp'].dt.date

        # Filter for the selected number of days
        start_date_filter = datetime.now() - timedelta(days=st.session_state.cost_dashboard_days)
        df_filtered_interactions = df_interactions[df_interactions['timestamp'] >= start_date_filter]

        if df_filtered_interactions.empty:
            st.info(f"No interactions found for the last {st.session_state.cost_dashboard_days} days.")
        else:
            total_routed_cost = df_filtered_interactions['cost_usd'].sum()
            total_queries_processed = len(df_filtered_interactions)
            total_cached_queries = df_filtered_interactions['cached'].sum()

            # Calculate "all-to-GPT-4o" baseline cost (replicated logic from source.py)
            avg_prompt_tokens_gpt4o = 700
            avg_completion_tokens_gpt4o = 300
            gpt4o_cost_per_query = (avg_prompt_tokens_gpt4o / 1_000_000) * MODEL_COSTS["gpt-4o"]["input_cost_per_1M_tokens"] + \
                                   (avg_completion_tokens_gpt4o / 1_000_000) * MODEL_COSTS["gpt-4o"]["output_cost_per_1M_tokens"] + \
                                   HANDLER_OVERHEAD_COSTS["document_qa"] # Add some overhead
            
            total_gpt4o_baseline_cost = total_queries_processed * gpt4o_cost_per_query
            cost_savings = total_gpt4o_baseline_cost - total_routed_cost
            percentage_savings = (cost_savings / total_gpt4o_baseline_cost) * 100 if total_gpt4o_baseline_cost > 0 else 0

            st.subheader("Overall Cost Summary:")
            st.markdown(f"- **Total Queries Processed:** `{total_queries_processed}`")
            st.markdown(f"- **Queries served from Cache:** `{total_cached_queries}` ({total_cached_queries/total_queries_processed:.1%} hit rate)")
            st.markdown(f"- **Total Cost (Routed + Cached):** `${total_routed_cost:.4f}`")
            st.markdown(f"- **Average Cost per Query (Routed + Cached):** `${total_routed_cost / total_queries_processed:.4f}`")
            st.markdown(f"- **Total Cost (Baseline - All to GPT-4o):** `${total_gpt4o_baseline_cost:.4f}`")
            st.markdown(f"- **Simulated Cost Savings:** `${cost_savings:.4f}` ({percentage_savings:.1f}%)")


            # Visualization 1: Comparison of Simulated API Costs
            st.subheader("API Cost Comparison: Routed vs. Baseline")
            costs_df = pd.DataFrame({
                'Scenario': ['Routed + Cached', 'Baseline (All to GPT-4o)'],
                'Total Cost ($)': [total_routed_cost, total_gpt4o_baseline_cost]
            })
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Scenario', y='Total Cost ($)', data=costs_df, palette='viridis', ax=ax1)
            ax1.set_title('Simulated API Costs: Routed + Cached vs. Baseline (All to GPT-4o)')
            ax1.set_ylabel('Total Cost (USD)')
            ax1.set_xlabel('Scenario')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig1)
            st.markdown(f"This chart clearly illustrates the financial advantage of our intelligent routing and caching strategy over a naive approach of sending all queries to a powerful LLM like GPT-4o.")

            # Visualization 2: Daily API Spend by Query Category
            st.subheader("Simulated Daily API Spend by Query Category")
            daily_spend = df_filtered_interactions.groupby(['date', 'category'])['cost_usd'].sum().unstack(fill_value=0)
            fig2, ax2 = plt.subplots(figsize=(12, 7))
            daily_spend.plot(kind='bar', stacked=True, colormap='viridis', ax=ax2)
            ax2.set_title('Simulated Daily API Spend by Query Category')
            ax2.set_ylabel('Cost (USD)')
            ax2.set_xlabel('Date')
            ax2.tick_params(axis='x', rotation=45, ha='right')
            ax2.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig2)
            st.markdown(f"Tracking daily spend by category helps us understand where our LLM resources are being consumed and allows for fine-tuning our routing and model selection policies.")


            # Visualization 3: Cache Performance: Hit Rate and Estimated Cost Savings
            st.subheader("Cache Performance: Hit Rate and Estimated Cost Savings")
            cache_hits_df = df_filtered_interactions[df_filtered_interactions['cached'] == True]
            cache_misses_df = df_filtered_interactions[df_filtered_interactions['cached'] == False]
            avg_non_cached_query_cost = cache_misses_df['cost_usd'].mean() if not cache_misses_df.empty else gpt4o_cost_per_query * 0.5
            estimated_cache_savings = total_cached_queries * avg_non_cached_query_cost

            fig3, ax3_1 = plt.subplots(figsize=(10, 6))
            color = 'tab:blue'
            ax3_1.set_xlabel('Metric')
            ax3_1.set_ylabel('Value (Rate)', color=color)
            ax3_1.bar('Cache Hit Rate', total_cached_queries / total_queries_processed, color=color, label='Cache Hit Rate')
            ax3_1.tick_params(axis='y', labelcolor=color)
            ax3_1.set_ylim(0,1)

            ax3_2 = ax3_1.twinx()
            color = 'tab:green'
            ax3_2.set_ylabel('Value (USD)', color=color)
            ax3_2.bar('Estimated Cost Savings from Cache', estimated_cache_savings, color=color, label='Estimated Cache Savings')
            ax3_2.tick_params(axis='y', labelcolor=color)

            fig3.tight_layout()
            plt.title('Cache Performance: Hit Rate and Estimated Cost Savings')
            st.pyplot(fig3)
            st.markdown(f"Semantic caching plays a vital role in reducing API calls for repetitive or semantically similar queries, leading to direct cost savings and improved response times.")


            # Visualization 4: Histogram of Simulated Response Times by Query Category
            st.subheader("Distribution of Response Times by Query Category")
            fig4, ax4 = plt.subplots(figsize=(12, 7))
            sns.histplot(data=df_filtered_interactions, x='latency_sec', hue='category', multiple='stack', bins=20, kde=True, palette='Spectral', ax=ax4)
            ax4.set_title('Distribution of Response Times by Query Category')
            ax4.set_xlabel('Latency (seconds)')
            ax4.set_ylabel('Number of Queries')
            ax4.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig4)
            st.markdown(f"This histogram shows the expected latency variations across different query categories. Data lookups are typically fastest, while complex research agent tasks take longer due to multi-step operations.")


# --- Page: Compliance Audit Log ---
elif st.session_state.current_page == 'Compliance Audit Log':
    st.title("Compliance Audit Trail")
    st.markdown(f"For financial firms, compliance logging is not optional – it is a regulatory requirement. As the Head of Research, ensuring every AI interaction is meticulously recorded is paramount. This audit trail supports **CFA Standard V(C) – Record Retention** and **Ethics Standard I(B)**, enabling supervisory review, error tracing, and regulatory examination.")
    st.markdown(f"Below is a sample of all logged interactions. You can filter by user ID or a start date to review specific records.")

    st.subheader("Filter Audit Records:")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.audit_filter_user_id = st.text_input("Filter by User ID (e.g., analyst_01)", st.session_state.audit_filter_user_id, key="audit_user_id_filter")
    with col2:
        st.session_state.audit_filter_start_date = st.date_input("Filter from Start Date", st.session_state.audit_filter_start_date, key="audit_start_date_filter")

    if st.button("Generate Audit Report", key="generate_audit_button"):
        filter_start_date_str = st.session_state.audit_filter_start_date.isoformat() if st.session_state.audit_filter_start_date else None
        audit_records = logger.export_audit_report(
            user_id=st.session_state.audit_filter_user_id if st.session_state.audit_filter_user_id else None,
            start_date=filter_start_date_str
        )
        
        if audit_records:
            df_audit = pd.DataFrame(audit_records)
            st.subheader("Audit Records:")
            st.dataframe(df_audit)
        else:
            st.info("No audit records found matching your criteria.")
    else:
        # Initial display of all records or a default set
        audit_records = logger.export_audit_report()
        if audit_records:
            df_audit = pd.DataFrame(audit_records)
            st.subheader("All Audit Records (initial view):")
            st.dataframe(df_audit)
        else:
            st.info("No audit records found yet.")

```
