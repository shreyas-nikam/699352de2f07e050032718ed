# QuLab: Lab 34: Building a Research Copilot

## Project Title and Description

**QuLab: Lab 34: Building a Research Copilot** is an advanced Streamlit application designed for **CFA Charterholders and Investment Professionals** at Apex Capital Management. This project demonstrates the construction of an intelligent AI-powered research copilot that efficiently processes financial queries, optimizes for cost, ensures regulatory compliance, and provides critical operational insights through a dedicated dashboard.

The application serves as a sophisticated front-end for various AI models and handlers, intelligently routing incoming queries (e.g., data lookups, document Q&A, general knowledge questions, multi-step agent tasks) to the most appropriate and cost-effective solution. It incorporates semantic caching to reduce redundant LLM calls and features a robust compliance logger to maintain a detailed audit trail of every interaction, adhering to financial industry standards like CFA Standard V(C) – Record Retention.

For the **Head of Research**, a comprehensive dashboard provides real-time insights into API costs, cache performance, query distribution, and response times, enabling data-driven decisions for resource allocation and operational efficiency.

## Features

This research copilot offers a rich set of functionalities tailored for financial professionals:

*   **Intelligent Query Routing**: Automatically directs user queries to specialized AI handlers (Data Lookup, RAG (Retrieval Augmented Generation), Research Agent, General Knowledge) based on query intent, optimizing for accuracy and efficiency.
*   **Cost Optimization**: Employs a tiered model strategy, leveraging less expensive LLMs for simpler tasks and reserving powerful models like GPT-4o for complex agentic workflows, significantly reducing API costs.
*   **Semantic Caching**: Implements a semantic cache using `text-embedding-3-small` to store and retrieve semantically similar query responses, drastically cutting down on redundant LLM calls, improving response times, and saving costs.
*   **Compliance Logging**: Records a detailed audit trail of every user interaction, including user ID, query, response, handler category, model used, cost, latency, and cache status, crucial for regulatory compliance.
*   **Interactive Chat Interface**: A user-friendly Streamlit interface allows analysts to submit queries and review their interaction history.
*   **OpenAI API Key Management**: Securely input and manage your OpenAI API key directly within the sidebar.
*   **Simulated LLM API Failure**: A dedicated button to simulate an LLM API failure, demonstrating the system's resilience and error handling.
*   **Comprehensive Cost & Compliance Dashboard**:
    *   **Cost Optimization Analysis**: Compares actual routed costs against a GPT-4o baseline, highlighting significant cost savings.
    *   **Daily API Spend by Category**: Visualizes cost allocation across different query categories over time.
    *   **Cache Performance Metrics**: Displays cache hit rate and estimated cost savings attributed to the semantic cache.
    *   **Response Time Distribution**: Histograms illustrating latency across query categories for performance monitoring.
    *   **Audit Log Sample**: Presents a sample of the compliance audit log, demonstrating adherence to record-keeping standards.
*   **Batch Query Simulation**: Ability to simulate a batch of diverse synthetic queries to generate robust data for dashboard analysis.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   An OpenAI API Key. You can obtain one from [OpenAI](https://platform.openai.com/account/api-keys).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd QuLab-Research-Copilot # Adjust if your clone directory is different
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    openai
    tiktoken
    scikit-learn # For cosine similarity if not handled by OpenAI embeddings directly
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your OpenAI API Key:**
    You can set your OpenAI API key in two ways:
    *   **Environment Variable (Recommended):** Set the `OPENAI_API_KEY` environment variable before running the application.
        ```bash
        export OPENAI_API_KEY="your_openai_api_key_here" # For Linux/macOS
        # set OPENAI_API_KEY="your_openai_api_key_here" # For Windows
        ```
    *   **Direct Input in App:** Enter your API key directly into the "Enter your OpenAI API Key" text box in the Streamlit sidebar.

    The application will prioritize the environment variable if set, otherwise, it will prompt for input in the UI.

## Usage

1.  **Run the Streamlit application:**
    Ensure your virtual environment is active and navigate to the project root where `app.py` is located.
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

2.  **Configure API Key:**
    If you haven't set the `OPENAI_API_KEY` environment variable, enter your OpenAI API key in the sidebar input field. A success message will confirm its activation.

3.  **Ask the Copilot:**
    *   Navigate to the "Ask Copilot" page (default).
    *   Enter your `User ID` (e.g., `analyst_01`).
    *   Type your financial research query into the text area.
    *   Click "Submit Query" to get a response.
    *   Observe the response, processing details, cost, latency, and whether it was served from the cache in the "Interaction History" section.
    *   You can also click "Simulate LLM API Failure" to see how the system handles errors.

4.  **Explore the Dashboard:**
    *   Navigate to the "Cost & Compliance Dashboard" page using the sidebar selector.
    *   Click "Simulate Batch Queries for Dashboard Analysis" to generate data. This will run 200 synthetic queries to populate the compliance logger and provide meaningful data for the visualizations.
    *   Once the simulation is complete, analyze the interactive charts and tables displaying cost optimization, daily spend, cache performance, response times, and a sample of the compliance audit log.

## Project Structure

The project is organized as follows:

```
.
├── app.py                     # Main Streamlit application file
├── source.py                  # Backend logic, AI handlers, SemanticCache, ComplianceLogger, constants
├── requirements.txt           # Python dependencies
└── README.md                  # This README file
```

*   **`app.py`**: Contains the Streamlit UI components, session state management, page navigation, and calls to the backend logic defined in `source.py`.
*   **`source.py`**: This file encapsulates the core intelligence of the research copilot. It defines:
    *   `MODEL_COSTS`, `HANDLER_OVERHEAD_COSTS`, `QUERY_CATEGORIES`, `SYNTHETIC_QUERIES`, `QUERY_DISTRIBUTION_PROFILE`, `SEMANTIC_CACHE_TEST_PAIRS` (constants for configuration and simulation).
    *   `QueryCategory` enum for categorizing queries.
    *   `ROUTER_PROMPT`, `route_query` (for intelligent routing).
    *   `handle_data_lookup`, `rag_answer`, `run_agent`, `handle_general_knowledge` (specific AI handlers).
    *   `SemanticCache` class (manages the semantic cache).
    *   `ComplianceLogger` class (logs all interactions for audit).
    *   `handle_query` (the central function orchestrating query processing, routing, caching, and logging).
    *   `client` (OpenAI client instance).

## Technology Stack

*   **Frontend Framework**: [Streamlit](https://streamlit.io/)
*   **Backend Language**: Python 3.8+
*   **Large Language Model (LLM) Provider**: [OpenAI API](https://openai.com/) (GPT models, Embedding models)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
*   **Numerical Computing**: [NumPy](https://numpy.org/)
*   **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
*   **Serialization**: `json`
*   **System Operations**: `os`
*   **Date/Time Handling**: `datetime`
*   **Tokenization**: [Tiktoken](https://github.com/openai/tiktoken)
*   **Machine Learning Utilities**: [scikit-learn](https://scikit-learn.org/) (potentially for similarity calculations if not exclusively using OpenAI embeddings for this)

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name` or `bugfix/your-bug-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to Python best practices and includes appropriate documentation and tests where applicable.

## License

This project is licensed under the MIT License - see the `LICENSE` file (if present, otherwise assume standard open-source license like MIT) for details.

## Contact

For any questions, suggestions, or collaboration opportunities, please reach out to:

*   **QuantUniversity**
*   **Website:** [www.quantuniversity.com](https://www.quantuniversity.com/)
*   **Email:** info@quantuniversity.com

---
_Developed for CFA Charterholders and Investment Professionals. AI for Financial Professionals._
_© QuantUniversity_
