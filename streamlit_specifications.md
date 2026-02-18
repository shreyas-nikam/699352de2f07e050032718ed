
# Streamlit Application Specification: AI Research Copilot for Apex Capital Management

## 1. Application Overview

The **AI Research Copilot for Apex Capital Management** Streamlit application serves as a sophisticated front-end for financial analysts, enabling them to efficiently query a vast array of financial information while ensuring cost optimization and regulatory compliance. As the Head of Research, the primary goal is to provide a robust AI tool that leverages Large Language Models (LLMs) intelligently, routing queries to the most appropriate and cost-effective backend handlers, implementing semantic caching for speed and efficiency, and maintaining an immutable compliance log for every interaction.

**High-Level Story Flow:**

1.  **Analyst Onboarding:** A CFA Charterholder (e.g., "analyst\_01") accesses the copilot and provides their OpenAI API key to enable LLM-powered features. This key is used to configure the underlying LLM client.
2.  **Interactive Querying:** The analyst submits a financial research query (e.g., "What was Google's Q1 revenue?" or "Analyze Tesla's competitive position"). They can also specify their user ID.
3.  **Intelligent Routing & Caching:**
    *   First, the system checks a **semantic cache** for a similar previous query. If a high-similarity match is found, the cached (and faster, cheaper) response is returned immediately.
    *   If a cache miss, the application's core "intelligent router" (using a lightweight LLM like `gpt-4o-mini`) classifies the query into predefined categories: "Data Lookup", "Document Q&A", "General Knowledge", or "Multi-step Research Agent".
4.  **Optimized Response Generation:** Based on the classified category, the query is dispatched to the most appropriate and cost-effective backend handler:
    *   "Data Lookup" queries go to a direct API call (no LLM).
    *   "General Knowledge" queries are answered by a small general LLM (`gpt-4o-mini`).
    *   "Document Q&A" queries trigger a Retrieval Augmented Generation (RAG) pipeline (potentially using `gpt-4o`).
    *   "Multi-step Research Agent" queries activate an agentic workflow with tools (potentially using `gpt-4o`).
    *   **Fallback handling** is integrated to return a user-friendly error message (e.g., "service unavailable") instead of crashing in case of primary LLM API failures.
5.  **Compliance Logging:** Every interaction—including the query text, response text, source citations, model used, token count, cost, latency, timestamp, user ID, category, and an AI-generated content flag—is meticulously recorded in an SQLite database, fulfilling critical compliance requirements.
6.  **Performance & Cost Monitoring:** The Head of Research (or other stakeholders) can navigate to a dedicated "Cost & Compliance Dashboard". Here, they can trigger a simulation of multiple queries to generate data for analysis. The dashboard then visualizes aggregated API costs (comparing the intelligent routing strategy to a "send-everything-to-GPT-4o" baseline), daily spend by query category, cache hit rates, latency distributions, and presents a sample of the compliance audit log. This demonstrates the system's Return on Investment (ROI) and adherence to regulatory standards.

This comprehensive workflow simulates a real-world financial research environment, showcasing how advanced AI can be deployed responsibly and efficiently within an investment firm, aligning with the needs of CFA Charterholders and Investment Professionals.

## 2. Code Requirements

### Imports

The `app.py` Streamlit application will begin with the following import statements:

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os

# All application logic and classes are imported directly from source.py
# This includes global client, MODEL_COSTS, HANDLER_OVERHEAD_COSTS, etc.
from source import (
    MODEL_COSTS, HANDLER_OVERHEAD_COSTS, QUERY_CATEGORIES, SYNTHETIC_QUERIES,
    QUERY_DISTRIBUTION_PROFILE, SEMANTIC_CACHE_TEST_PAIRS, QueryCategory,
    ROUTER_PROMPT, route_query,
    handle_data_lookup, rag_answer, run_agent, handle_general_knowledge,
    SemanticCache, ComplianceLogger, handle_query, client # client is the global OpenAI instance from source.py
)

# Set matplotlib style for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid')
```

### `st.session_state` Design

The following `st.session_state` keys will be initialized, updated, and read to manage application state across reruns and different pages:

*   `st.session_state.openai_api_key`:
    *   **Initialized**: `""` (empty string).
    *   **Updated**: Via `st.sidebar.text_input` when the user enters their key.
    *   **Read**: Used to configure the global `client` object imported from `source.py`.
*   `st.session_state.is_api_key_set`:
    *   **Initialized**: `False`.
    *   **Updated**: `True` if `st.session_state.openai_api_key` is non-empty and API key validation is successful; `False` otherwise.
    *   **Read**: Controls the visibility and interactivity of most application widgets, ensuring functionality is only available when a valid API key is present.
*   `st.session_state.current_page`:
    *   **Initialized**: `"Ask Copilot"`.
    *   **Updated**: By the `st.sidebar.selectbox` widget for page navigation.
    *   **Read**: Determines which content block is rendered on the main page, simulating a multi-page experience.
*   `st.session_state.user_id`:
    *   **Initialized**: `"analyst_01"`.
    *   **Updated**: Via a `st.text_input` widget on the "Ask Copilot" page.
    *   **Read**: Passed as an argument to `handle_query` for compliance logging purposes.
*   `st.session_state.chat_history`:
    *   **Initialized**: `[]` (an empty list). Each entry is a dictionary containing the detailed response from `handle_query` (e.g., `query`, `response`, `category`, `model_used`, `cost_usd`, `latency_sec`, `sources`, `cached`, `disclaimer`).
    *   **Updated**: Appended with the result of each `handle_query` call on the "Ask Copilot" page.
    *   **Read**: Displayed as a chronological log of user interactions on the "Ask Copilot" page.
*   `st.session_state.semantic_cache_instance`:
    *   **Initialized**: `None`.
    *   **Updated**: Once `st.session_state.is_api_key_set` is `True`, an instance of `SemanticCache` (from `source.py`) is created and stored here. This instance will implicitly use the global `client` from `source.py`, which has been configured with the user's API key.
    *   **Read**: Used internally by `handle_query` for cache lookup and storage.
*   `st.session_state.compliance_logger_instance`:
    *   **Initialized**: `None`.
    *   **Updated**: Once `st.session_state.is_api_key_set` is `True`, an instance of `ComplianceLogger` (from `source.py`) is created and stored here.
    *   **Read**: Used internally by `handle_query` for logging, and by the "Cost & Compliance Dashboard" to fetch audit reports and cost summaries.
*   `st.session_state.simulated_queries_processed`:
    *   **Initialized**: `False`.
    *   **Updated**: Set to `True` after the "Simulate Batch Queries" button is pressed on the dashboard and the simulation loop completes.
    *   **Read**: Prevents re-running the potentially long simulation on every dashboard page reload.
*   `st.session_state.audit_report_data`:
    *   **Initialized**: `[]` (an empty list of dictionaries).
    *   **Updated**: Populated by calling `st.session_state.compliance_logger_instance.export_audit_report()` after the batch simulation on the "Cost & Compliance Dashboard" or when the dashboard is accessed if data already exists in the SQLite database.
    *   **Read**: Used to populate the dataframes and charts on the "Cost & Compliance Dashboard" to display aggregated metrics and audit logs.

### UI Interactions and Function Calls

The Streamlit application will interact with functions and classes from `source.py` at the following points:

1.  **OpenAI API Key Configuration (Sidebar):**
    *   **Widget**: `st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key_widget_input")`
    *   **Function Call/Logic**:
        *   When `st.session_state.openai_api_key_widget_input` is updated, `st.session_state.openai_api_key` is set.
        *   If the API key is non-empty:
            *   `client.api_key = st.session_state.openai_api_key` (This crucial step configures the global `client` object imported from `source.py`).
            *   `st.session_state.semantic_cache_instance = SemanticCache(embedding_model_name="text-embedding-3-small", similarity_threshold=0.90, max_age_hours=1)`
            *   `st.session_state.compliance_logger_instance = ComplianceLogger()`
            *   `st.session_state.is_api_key_set = True`
            *   On valid key, `st.sidebar.success("OpenAI API Key set successfully!")` and `st.experimental_rerun()` to enable disabled widgets.
        *   If the API key is cleared:
            *   `st.session_state.is_api_key_set = False` and associated session state variables are reset.
            *   `st.sidebar.warning("OpenAI API Key removed. Functionality disabled.")` and `st.experimental_rerun()`.

2.  **Page Navigation (Sidebar):**
    *   **Widget**: `st.sidebar.selectbox("Navigate", ["Ask Copilot", "Cost & Compliance Dashboard"], key="page_selector")`
    *   **Function Call/Logic**: Updates `st.session_state.current_page` which controls conditional rendering of page content.

3.  **"Ask Copilot" Page:**
    *   **User ID Input**: `st.text_input("Your User ID", value=st.session_state.user_id, key="user_id_input", disabled=not st.session_state.is_api_key_set)`
        *   **Function Call/Logic**: Updates `st.session_state.user_id` on change.
    *   **Query Input**: `st.text_area("Enter your financial research query:", height=100, key="query_input", disabled=not st.session_state.is_api_key_set)`
    *   **Submit Query Button**: `st.button("Submit Query", key="submit_query_button", disabled=not st.session_state.is_api_key_set)`
        *   **Function Call**: If clicked, `response_data = handle_query(query=st.session_state.query_input, user_id=st.session_state.user_id)` (using the `handle_query` function from `source.py`).
        *   **`st.session_state` Update**: `st.session_state.chat_history.append(response_data)`.
    *   **Simulate LLM API Failure Button**: `st.button("Simulate LLM API Failure", key="simulate_failure_button", disabled=not st.session_state.is_api_key_set)`
        *   **Function Call**: If clicked, `response_data = handle_query(query=st.session_state.query_input, user_id=st.session_state.user_id, simulate_failure=True)`.
        *   **`st.session_state` Update**: `st.session_state.chat_history.append(response_data)`.

4.  **"Cost & Compliance Dashboard" Page:**
    *   **Simulate Batch Queries Button**: `st.button("Simulate Batch Queries for Dashboard Analysis", key="simulate_batch_button", disabled=not st.session_state.is_api_key_set)`
        *   **Function Call/Logic**: If clicked and `st.session_state.simulated_queries_processed` is `False`:
            *   A loop runs `num_simulated_queries` (e.g., 200) times. In each iteration:
                *   A query and user ID are randomly generated based on `QUERY_DISTRIBUTION_PROFILE` and `SYNTHETIC_QUERIES`/`SEMANTIC_CACHE_TEST_PAIRS` (as per `source.py` simulation logic).
                *   `handle_query(query, user_id, simulate_failure=...)` is called. This automatically logs interactions via `st.session_state.compliance_logger_instance`.
            *   `st.session_state.simulated_queries_processed = True`.
            *   `st.session_state.audit_report_data = st.session_state.compliance_logger_instance.export_audit_report()` to fetch all logged data.
        *   **Dashboard Data Generation**: After simulation or on dashboard load if data exists:
            *   `df_interactions = pd.DataFrame(st.session_state.audit_report_data)`
            *   Visualizations use `df_interactions` and constants from `source.py` like `MODEL_COSTS`, `HANDLER_OVERHEAD_COSTS`.
            *   **Function Calls for Data**:
                *   `st.session_state.compliance_logger_instance.get_cost_summary()` (used internally for calculating daily spend).
                *   `st.session_state.compliance_logger_instance.export_audit_report()` (to populate `st.session_state.audit_report_data`).

### Markdown Content

#### Global Markdown (App Title and Sidebar)

```python
st.set_page_config(layout="wide", page_title="AI Research Copilot")

st.markdown("# AI Research Copilot for Apex Capital Management")
st.markdown("---")

st.sidebar.title("Configuration & Navigation")
st.sidebar.markdown("### OpenAI API Key")

# Initialize session state for API key and related components if not already present
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
if "is_api_key_set" not in st.session_state:
    st.session_state.is_api_key_set = False
if "user_id" not in st.session_state:
    st.session_state.user_id = "analyst_01"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "semantic_cache_instance" not in st.session_state:
    st.session_state.semantic_cache_instance = None
if "compliance_logger_instance" not in st.session_state:
    st.session_state.compliance_logger_instance = None
if "simulated_queries_processed" not in st.session_state:
    st.session_state.simulated_queries_processed = False
if "audit_report_data" not in st.session_state:
    st.session_state.audit_report_data = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Ask Copilot"


# API key input widget
openai_api_key_input = st.sidebar.text_input("Enter your OpenAI API Key", type="password", value=st.session_state.openai_api_key, key="openai_api_key_widget_input")

# Logic to initialize/update global client and session state based on API key input
if openai_api_key_input != st.session_state.openai_api_key:
    st.session_state.openai_api_key = openai_api_key_input
    if openai_api_key_input:
        try:
            # Configure the global client imported from source.py
            client.api_key = st.session_state.openai_api_key
            # Initialize other persistent components that rely on the configured client
            st.session_state.semantic_cache_instance = SemanticCache(
                embedding_model_name="text-embedding-3-small", 
                similarity_threshold=0.90, 
                max_age_hours=1
            )
            st.session_state.compliance_logger_instance = ComplianceLogger()
            st.session_state.is_api_key_set = True
            st.sidebar.success("OpenAI API Key set successfully!")
            st.experimental_rerun() # Rerun to enable widgets
        except Exception as e:
            st.session_state.is_api_key_set = False
            st.sidebar.error(f"Invalid OpenAI API Key: {e}")
    else:
        st.session_state.is_api_key_set = False
        st.session_state.semantic_cache_instance = None
        st.session_state.compliance_logger_instance = None
        st.sidebar.warning("OpenAI API Key removed. Functionality disabled.")
        st.experimental_rerun()
elif st.session_state.openai_api_key and not st.session_state.is_api_key_set:
    # This block handles the case where key was set via env var initially but not validated yet
    try:
        client.api_key = st.session_state.openai_api_key
        st.session_state.semantic_cache_instance = SemanticCache(
            embedding_model_name="text-embedding-3-small", 
            similarity_threshold=0.90, 
            max_age_hours=1
        )
        st.session_state.compliance_logger_instance = ComplianceLogger()
        st.session_state.is_api_key_set = True
        st.sidebar.success("OpenAI API Key loaded (from environment or previous input).")
        st.experimental_rerun()
    except Exception as e:
        st.session_state.is_api_key_set = False
        st.sidebar.error(f"Error with OpenAI API Key: {e}")


st.sidebar.markdown("---")
st.session_state.current_page = st.sidebar.selectbox(
    "Navigate",
    ["Ask Copilot", "Cost & Compliance Dashboard"],
    key="page_selector"
)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed for CFA Charterholders and Investment Professionals.")
```

#### "Ask Copilot" Page Markdown

```python
if st.session_state.current_page == "Ask Copilot":
    st.markdown(f"## Ask the Research Copilot")
    st.markdown(f"As a **CFA Charterholder and Investment Professional** at Apex Capital Management, you can use this copilot to quickly get insights into financial queries. The system intelligently routes your questions to the most efficient AI handler, optimizes for cost, and logs every interaction for compliance.")

    st.markdown(f"### Your Analyst Profile")
    st.markdown(f"Enter your User ID. This will be logged for compliance and audit purposes.")
    st.session_state.user_id = st.text_input("Your User ID", value=st.session_state.user_id, key="user_id_input", disabled=not st.session_state.is_api_key_set)

    st.markdown(f"### Submit a Research Query")
    st.markdown(f"Ask any financial research question. The copilot will determine the best way to answer it (e.g., data lookup, document Q&A, general knowledge, or multi-step agent).")
    
    query_input_text = st.text_area("Enter your financial research query:", height=100, key="query_input", disabled=not st.session_state.is_api_key_set)
    
    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.button("Submit Query", key="submit_query_button", disabled=not st.session_state.is_api_key_set)
    with col2:
        simulate_failure_button = st.button("Simulate LLM API Failure", key="simulate_failure_button", disabled=not st.session_state.is_api_key_set)

    if submit_button and query_input_text:
        with st.spinner("Processing query..."):
            response_data = handle_query(query=query_input_text, user_id=st.session_state.user_id)
            st.session_state.chat_history.append(response_data)
    
    if simulate_failure_button and query_input_text:
        with st.spinner("Simulating API failure..."):
            response_data = handle_query(query=query_input_text, user_id=st.session_state.user_id, simulate_failure=True)
            st.session_state.chat_history.append(response_data)

    st.markdown(f"---")
    st.markdown(f"### Interaction History")

    if not st.session_state.chat_history:
        st.info("No interactions yet. Ask a question or simulate a failure!")
    else:
        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"**Query {len(st.session_state.chat_history) - i}:** `{entry['query']}` (Category: `{entry['category'].replace('_', ' ').title()}`)", expanded=False):
                st.markdown(f"**User:** `{entry['user_id']}`")
                st.markdown(f"**Response:** {entry['response']}")
                if entry.get('disclaimer'):
                    st.warning(entry['disclaimer'])
                st.markdown(f"**Processing Details:**")
                st.markdown(f"- **Category Detected:** `{entry['category'].replace('_', ' ').title()}`")
                st.markdown(f"- **Model Used:** `{entry['model_used']}`")
                st.markdown(f"- **Cost (USD):** `${entry['cost_usd']:.6f}`")
                st.markdown(f"- **Latency (sec):** `{entry['latency_sec']:.2f}s`")
                st.markdown(f"- **Cached:** {'✅ Yes' if entry.get('cached') else '❌ No'}")
                if entry.get('sources'):
                    st.markdown(f"- **Sources:**")
                    for source in entry['sources']:
                        st.markdown(f"  - `{json.dumps(source)}`")
```

#### "Cost & Compliance Dashboard" Page Markdown

```python
elif st.session_state.current_page == "Cost & Compliance Dashboard":
    st.markdown(f"## Cost & Compliance Dashboard")
    st.markdown(f"As the **Head of Research at Apex Capital Management**, this dashboard provides critical insights into the operational efficiency and regulatory adherence of our AI Research Copilot. Monitor costs, assess cache effectiveness, and review compliance logs to ensure responsible AI deployment.")

    if not st.session_state.is_api_key_set:
        st.warning("Please enter your OpenAI API Key in the sidebar to enable dashboard features.")
    else:
        st.markdown(f"### Generate Data for Analysis")
        st.markdown(f"To see meaningful trends and cost savings, we need to simulate a batch of diverse financial queries. This process logs interactions and populates the database for the visualizations below.")
        
        simulate_batch_button = st.button("Simulate Batch Queries for Dashboard Analysis", key="simulate_batch_button", disabled=st.session_state.simulated_queries_processed)

        if simulate_batch_button and not st.session_state.simulated_queries_processed:
            st.session_state.simulated_queries_processed = True
            st.session_state.audit_report_data = [] # Reset data before simulation
            with st.spinner("Simulating queries... This may take a moment."):
                num_simulated_queries = 200 # As per source.py
                categories = list(QUERY_DISTRIBUTION_PROFILE.keys())
                probabilities = list(QUERY_DISTRIBUTION_PROFILE.values())

                for i in range(num_simulated_queries):
                    chosen_category = np.random.choice(categories, p=probabilities)
                    
                    # Logic to pick a query, potentially similar for cache hits
                    if np.random.rand() < 0.2 and chosen_category in ["data_lookup", "general_knowledge", "document_qa", "research_agent"]:
                        if chosen_category == "data_lookup":
                            query_pair = np.random.choice([SEMANTIC_CACHE_TEST_PAIRS[0], SEMANTIC_CACHE_TEST_PAIRS[3]])
                        elif chosen_category == "general_knowledge":
                            query_pair = SEMANTIC_CACHE_TEST_PAIRS[1]
                        elif chosen_category == "document_qa":
                            query_pair = SEMANTIC_CACHE_TEST_PAIRS[4]
                        elif chosen_category == "research_agent":
                            query_pair = SEMANTIC_CACHE_TEST_PAIRS[2]
                        else: # Fallback for categories not in SEMANTIC_CACHE_TEST_PAIRS if needed
                            query_pair = (np.random.choice(SYNTHETIC_QUERIES[chosen_category]),)
                        
                        query_to_use = query_pair[1] if len(query_pair) > 1 and np.random.rand() < 0.7 else query_pair[0]
                    else:
                        query_to_use = np.random.choice(SYNTHETIC_QUERIES[chosen_category])
                    
                    user_id = f"analyst_{np.random.randint(1, 10):02d}" # Simulate 10 different analysts
                    
                    # handle_query logs directly to compliance_logger_instance
                    response_data = handle_query(query_to_use, user_id, simulate_failure=(i == num_simulated_queries // 2 and num_simulated_queries > 0)) # Simulate one failure
            
            # After simulation, fetch all data from the logger
            st.session_state.audit_report_data = st.session_state.compliance_logger_instance.export_audit_report()
            st.success(f"Simulated {num_simulated_queries} queries. Data is ready for analysis.")
            # Clear current chat history after simulation for a fresh start on dashboard
            st.session_state.chat_history = [] 

        # Ensure audit_report_data is loaded if simulation has run
        if st.session_state.simulated_queries_processed and not st.session_state.audit_report_data:
             st.session_state.audit_report_data = st.session_state.compliance_logger_instance.export_audit_report()

        if not st.session_state.audit_report_data:
            st.info("No data available. Please simulate a batch of queries first to populate the dashboard.")
        else:
            df_interactions = pd.DataFrame(st.session_state.audit_report_data)
            df_interactions['cost_usd'] = pd.to_numeric(df_interactions['cost_usd'], errors='coerce').fillna(0)
            df_interactions['latency_sec'] = pd.to_numeric(df_interactions['latency_sec'], errors='coerce').fillna(0)
            df_interactions['cached'] = pd.to_numeric(df_interactions['cached'], errors='coerce').fillna(0).astype(bool)
            df_interactions['timestamp'] = pd.to_datetime(df_interactions['timestamp'])

            if not df_interactions.empty:
                st.markdown("### 1. Cost Optimization Analysis")
                st.markdown(f"Understanding the financial impact of AI deployment is crucial. This section compares our intelligent routing strategy against a baseline of sending all queries to an expensive model like `GPT-4o`.")

                total_routed_cost = df_interactions['cost_usd'].sum()
                total_queries_processed = len(df_interactions)
                total_cached_queries = df_interactions['cached'].sum()

                # Calculate "all-to-GPT-4o" baseline cost (as per source.py)
                avg_prompt_tokens_gpt4o = 700
                avg_completion_tokens_gpt4o = 300
                gpt4o_cost_per_query = (avg_prompt_tokens_gpt4o / 1_000_000) * MODEL_COSTS["gpt-4o"]["input_cost_per_1M_tokens"] + \
                                       (avg_completion_tokens_gpt4o / 1_000_000) * MODEL_COSTS["gpt-4o"]["output_cost_per_1M_tokens"] + \
                                       HANDLER_OVERHEAD_COSTS["document_qa"] # Add some overhead
                
                total_gpt4o_baseline_cost = total_queries_processed * gpt4o_cost_per_query
                cost_savings = total_gpt4o_baseline_cost - total_routed_cost
                percentage_savings = (cost_savings / total_gpt4o_baseline_cost) * 100 if total_gpt4o_baseline_cost > 0 else 0

                st.markdown(f"**Total Queries Processed:** `{total_queries_processed}`")
                st.markdown(f"**Queries served from Cache:** `{total_cached_queries}` (`{total_cached_queries/total_queries_processed:.1%}` hit rate)")
                st.markdown(f"**Total Cost (Routed + Cached):** `${total_routed_cost:.4f}`")
                st.markdown(f"**Total Cost (Baseline - All to GPT-4o):** `${total_gpt4o_baseline_cost:.4f}`")
                st.markdown(f"**Simulated Cost Savings:** `${cost_savings:.4f}` (`{percentage_savings:.1f}%`)")
                
                st.markdown(r"$$ \text{{Routing Cost}} = \left( \frac{{\text{{Prompt Tokens}}}}{{10^6}} \times \text{{Prompt Cost/1M Tokens}} \right) + \left( \frac{{\text{{Completion Tokens}}}}{{10^6}} \times \text{{Completion Cost/1M Tokens}} \right) $$")
                st.markdown(r"where $\text{{Prompt Tokens}}$ are the tokens sent in the query, $\text{{Completion Tokens}}$ are the tokens in the AI's response, and $\text{{Cost/1M Tokens}}$ are the predefined costs for the specific LLM model.")
                
                # Chart 1: Comparison of Simulated API Costs
                costs_df = pd.DataFrame({
                    'Scenario': ['Routed + Cached', 'Baseline (All to GPT-4o)'],
                    'Total Cost ($)': [total_routed_cost, total_gpt4o_baseline_cost]
                })
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                sns.barplot(x='Scenario', y='Total Cost ($)', data=costs_df, palette='viridis', ax=ax1)
                ax1.set_title('Simulated API Costs: Routed + Cached vs. Baseline (All to GPT-4o)')
                ax1.set_ylabel('Total Cost (USD)')
                ax1.set_xlabel('Scenario')
                st.pyplot(fig1)

                st.markdown("### 2. Daily API Spend by Query Category")
                st.markdown(f"This visualization helps in monitoring where our AI budget is being allocated, broken down by category. This allows for proactive cost management and resource optimization decisions.")
                
                df_interactions['date'] = df_interactions['timestamp'].dt.date
                daily_spend = df_interactions.groupby(['date', 'category'])['cost_usd'].sum().unstack(fill_value=0)
                
                fig2, ax2 = plt.subplots(figsize=(12, 7))
                daily_spend.plot(kind='bar', stacked=True, colormap='viridis', ax=ax2)
                ax2.set_title('Simulated Daily API Spend by Query Category')
                ax2.set_ylabel('Cost (USD)')
                ax2.set_xlabel('Date')
                ax2.tick_params(axis='x', rotation=45, ha='right')
                ax2.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig2)

                st.markdown("### 3. Cache Performance: Hit Rate and Estimated Cost Savings")
                st.markdown(f"Semantic caching significantly reduces redundant LLM calls, improving both speed and cost efficiency. This section highlights the cache's performance.")
                
                st.markdown(r"$$ \text{{similarity}}(\mathbf{{A}}, \mathbf{{B}}) = \frac{{\mathbf{{A}} \cdot \mathbf{{B}}}}{{|\!|\mathbf{{A}}|\!| \cdot |\!|\mathbf{{B}}|\!|}} $$")
                st.markdown(r"where $\mathbf{{A}}$ and $\mathbf{{B}}$ are the embedding vectors of two queries, $\mathbf{{A}} \cdot \mathbf{{B}}$ is their dot product, and $|\!|\mathbf{{A}}|\!|$ and $|\!|\mathbf{{B}}|\!|$ are their Euclidean norms (magnitudes). A similarity score close to 1 indicates high semantic resemblance, triggering a cache hit.")

                cache_hits_df = df_interactions[df_interactions['cached'] == True]
                cache_misses_df = df_interactions[df_interactions['cached'] == False]
                
                avg_non_cached_query_cost = cache_misses_df['cost_usd'].mean() if not cache_misses_df.empty else gpt4o_cost_per_query * 0.5
                estimated_cache_savings = total_cached_queries * avg_non_cached_query_cost
                
                cache_hit_rate = total_cached_queries / total_queries_processed if total_queries_processed > 0 else 0
                
                cache_summary_df = pd.DataFrame({
                    'Metric': ['Cache Hit Rate', 'Estimated Cost Savings from Cache'],
                    'Value': [cache_hit_rate, estimated_cache_savings]
                })

                fig3, ax3_1 = plt.subplots(figsize=(10, 6))
                color = 'tab:blue'
                ax3_1.set_xlabel('Metric')
                ax3_1.set_ylabel('Value (Rate)', color=color)
                ax3_1.bar(cache_summary_df['Metric'][0], cache_summary_df['Value'][0], color=color, label='Cache Hit Rate')
                ax3_1.tick_params(axis='y', labelcolor=color)
                ax3_1.set_ylim(0,1)

                ax3_2 = ax3_1.twinx()
                color = 'tab:green'
                ax3_2.set_ylabel('Value (USD)', color=color)
                ax3_2.bar(cache_summary_df['Metric'][1], cache_summary_df['Value'][1], color=color, label='Estimated Cache Savings')
                ax3_2.tick_params(axis='y', labelcolor=color)
                fig3.tight_layout()
                ax3_1.set_title('Cache Performance: Hit Rate and Estimated Cost Savings')
                st.pyplot(fig3)


                st.markdown("### 4. Distribution of Response Times by Query Category")
                st.markdown(f"Latency is a key factor in user experience. This histogram illustrates the distribution of response times, allowing us to identify bottlenecks and optimize for faster insights for our analysts.")
                
                fig4, ax4 = plt.subplots(figsize=(12, 7))
                sns.histplot(data=df_interactions, x='latency_sec', hue='category', multiple='stack', bins=20, kde=True, palette='Spectral', ax=ax4)
                ax4.set_title('Distribution of Response Times by Query Category')
                ax4.set_xlabel('Latency (seconds)')
                ax4.set_ylabel('Number of Queries')
                ax4.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig4)


                st.markdown("### 5. Compliance Audit Log Sample")
                st.markdown(f"For **regulatory compliance (CFA Standard V(C) – Record Retention)**, every interaction with the AI Copilot is logged. This sample demonstrates the detailed audit trail available for supervisory review and regulatory examination.")
                
                if not df_interactions.empty:
                    sample_audit_log_df = df_interactions[['timestamp', 'user_id', 'query', 'category', 'model', 'cost_usd', 'latency_sec', 'cached']].sample(min(5, len(df_interactions)))
                    st.dataframe(sample_audit_log_df, use_container_width=True)
                else:
                    st.info("No audit log entries to display yet.")

                st.markdown(f"---")
                st.markdown(f"**Disclaimer for AI-generated Content:** All AI-generated responses should be verified before use. The system prioritizes efficiency and compliance, but human oversight remains critical for financial analysis.")
            else:
                st.info("Simulate batch queries to populate the dashboard with data.")

st.sidebar.markdown("---")
st.sidebar.markdown("_AI for Financial Professionals_")
st.sidebar.markdown("© QuantUniversity")
```
