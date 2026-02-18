import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os

from source import (
    MODEL_COSTS, HANDLER_OVERHEAD_COSTS, QUERY_CATEGORIES, SYNTHETIC_QUERIES,
    QUERY_DISTRIBUTION_PROFILE, SEMANTIC_CACHE_TEST_PAIRS, QueryCategory,
    ROUTER_PROMPT, route_query,
    handle_data_lookup, rag_answer, run_agent, handle_general_knowledge,
    SemanticCache, ComplianceLogger, handle_query, client
)

# Set matplotlib style for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid')

st.set_page_config(page_title="QuLab: Lab 34: Building a Research Copilot", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 34: Building a Research Copilot")
st.divider()

# Initialize session state
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

st.sidebar.title("Configuration & Navigation")
st.sidebar.markdown(f"### OpenAI API Key")

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
            st.rerun() 
        except Exception as e:
            st.session_state.is_api_key_set = False
            st.sidebar.error(f"Invalid OpenAI API Key: {e}")
    else:
        st.session_state.is_api_key_set = False
        st.session_state.semantic_cache_instance = None
        st.session_state.compliance_logger_instance = None
        st.sidebar.warning("OpenAI API Key removed. Functionality disabled.")
        st.rerun()
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
        st.rerun()
    except Exception as e:
        st.session_state.is_api_key_set = False
        st.sidebar.error(f"Error with OpenAI API Key: {e}")

st.sidebar.markdown("---")

# Navigation
page_options = ["Ask Copilot", "Cost & Compliance Dashboard"]
try:
    current_index = page_options.index(st.session_state.current_page)
except ValueError:
    current_index = 0

st.session_state.current_page = st.sidebar.selectbox(
    "Navigate",
    page_options,
    index=current_index,
    key="page_selector"
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"Developed for CFA Charterholders and Investment Professionals.")

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
                        else:
                            query_pair = (np.random.choice(SYNTHETIC_QUERIES[chosen_category]),)
                        
                        query_to_use = query_pair[1] if len(query_pair) > 1 and np.random.rand() < 0.7 else query_pair[0]
                    else:
                        query_to_use = np.random.choice(SYNTHETIC_QUERIES[chosen_category])
                    
                    user_id = f"analyst_{np.random.randint(1, 10):02d}" # Simulate 10 different analysts
                    
                    # handle_query logs directly to compliance_logger_instance
                    handle_query(query_to_use, user_id, simulate_failure=(i == num_simulated_queries // 2 and num_simulated_queries > 0))
            
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
                st.markdown(f"### 1. Cost Optimization Analysis")
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
                
                st.markdown(r"$$ \text{Routing Cost} = \left( \frac{\text{Prompt Tokens}}{10^6} \times \text{Prompt Cost/1M Tokens} \right) + \left( \frac{\text{Completion Tokens}}{10^6} \times \text{Completion Cost/1M Tokens} \right) $$")
                st.markdown(r"where $\text{Prompt Tokens}$ are the tokens sent in the query, $\text{Completion Tokens}$ are the tokens in the AI's response, and $\text{Cost/1M Tokens}$ are the predefined costs for the specific LLM model.")
                
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

                st.markdown(f"### 2. Daily API Spend by Query Category")
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

                st.markdown(f"### 3. Cache Performance: Hit Rate and Estimated Cost Savings")
                st.markdown(f"Semantic caching significantly reduces redundant LLM calls, improving both speed and cost efficiency. This section highlights the cache's performance.")
                
                st.markdown(r"$$ \text{similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{|\!|\mathbf{A}|\!| \cdot |\!|\mathbf{B}|\!|} $$")
                st.markdown(r"where $\mathbf{A}$ and $\mathbf{B}$ are the embedding vectors of two queries, $\mathbf{A} \cdot \mathbf{B}$ is their dot product, and $|\!|\mathbf{A}|\!|$ and $|\!|\mathbf{B}|\!|$ are their Euclidean norms (magnitudes). A similarity score close to 1 indicates high semantic resemblance, triggering a cache hit.")

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


                st.markdown(f"### 4. Distribution of Response Times by Query Category")
                st.markdown(f"Latency is a key factor in user experience. This histogram illustrates the distribution of response times, allowing us to identify bottlenecks and optimize for faster insights for our analysts.")
                
                fig4, ax4 = plt.subplots(figsize=(12, 7))
                sns.histplot(data=df_interactions, x='latency_sec', hue='category', multiple='stack', bins=20, kde=True, palette='Spectral', ax=ax4)
                ax4.set_title('Distribution of Response Times by Query Category')
                ax4.set_xlabel('Latency (seconds)')
                ax4.set_ylabel('Number of Queries')
                ax4.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig4)


                st.markdown(f"### 5. Compliance Audit Log Sample")
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