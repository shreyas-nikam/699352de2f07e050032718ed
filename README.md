# QuLab: Lab 34: Building a Research Copilot

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab: Lab 34: Building a Research Copilot** is a sophisticated Streamlit application designed as an AI-powered research assistant for financial professionals, specifically CFA Charterholders and Investment Professionals at firms like Apex Capital Management.

This copilot intelligently routes financial research queries to the most appropriate and cost-effective AI handlers (e.g., data lookup, RAG, multi-step agent, or general knowledge LLM). It incorporates a semantic caching layer for performance and cost optimization, and meticulously logs every interaction for comprehensive compliance and audit purposes, adhering to industry standards like **CFA Standard V(C) – Record Retention**.

The application features an interactive chat interface for direct query submission and a dedicated dashboard for monitoring operational costs, cache performance, and reviewing audit trails. It empowers research heads and compliance officers to oversee the responsible and efficient deployment of AI within financial analysis workflows.

## Features

This Research Copilot offers a robust set of features tailored for financial research and compliance:

*   **Intelligent Query Routing**: Automatically categorizes incoming queries (e.g., `data_lookup`, `document_qa`, `research_agent`, `general_knowledge`) and dispatches them to specialized AI handlers to ensure optimal response quality and resource utilization.
*   **Cost Optimization**: Employs a tiered model strategy, utilizing cheaper, faster models where appropriate and only escalating to more expensive, powerful models when necessary. The system tracks token usage and associated costs for full transparency.
*   **Semantic Caching**: Integrates a semantic cache that stores embeddings of past queries and their responses. Subsequent semantically similar queries are served from the cache, drastically reducing LLM API calls, improving response times, and cutting costs.
*   **Comprehensive Compliance Logging**: Every user interaction, query, response, handler chosen, model used, cost incurred, and latency is meticulously logged. This audit trail is crucial for regulatory adherence and internal review.
*   **Interactive Chat Interface**: Provides a user-friendly Streamlit interface for analysts to submit queries and review their interaction history.
*   **Simulated LLM API Failure**: Allows users to test the application's resilience by simulating an LLM API error, demonstrating how the system handles such scenarios.
*   **Cost & Compliance Dashboard**: A dedicated dashboard for management to:
    *   View total costs and compare them against a "no-routing" baseline (e.g., all queries to GPT-4o) to highlight cost savings.
    *   Monitor daily API spend broken down by query category.
    *   Analyze semantic cache hit rates and estimated cost savings from caching.
    *   Visualize the distribution of response times by query category to identify performance bottlenecks.
    *   Access a sample of the detailed, immutable audit log for compliance review.
*   **Dynamic Configuration**: OpenAI API key can be securely entered via the sidebar or loaded from environment variables.
*   **User Identity Management**: Allows setting a `user_id` for each interaction, crucial for personalized logging and accountability.

## Getting Started

Follow these instructions to get the Research Copilot up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   An OpenAI API Key. You can obtain one from the [OpenAI Platform](https://platform.openai.com/).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/quLab-research-copilot.git
    cd quLab-research-copilot
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    (If `requirements.txt` is not provided, you can generate one or install manually):
    ```bash
    pip install streamlit pandas numpy matplotlib seaborn openai scikit-learn tiktoken
    ```

4.  **Set your OpenAI API Key:**

    It is highly recommended to set your OpenAI API key as an environment variable for security:

    ```bash
    # On macOS/Linux
    export OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY"

    # On Windows (Command Prompt)
    set OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY"

    # On Windows (PowerShell)
    $env:OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY"
    ```
    Alternatively, you can enter it directly in the Streamlit application's sidebar.

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    This will open the application in your default web browser.

2.  **Configure API Key:**
    *   If you set the `OPENAI_API_KEY` environment variable, it should be automatically detected.
    *   Otherwise, enter your OpenAI API Key in the "Enter your OpenAI API Key" text box in the sidebar. A "success" message will confirm validation.

3.  **Set Your User ID:**
    *   On the "Ask Copilot" page, enter your desired `User ID` (e.g., `analyst_01`). This ID will be logged with every query for compliance.

4.  **Ask a Research Query:**
    *   In the "Enter your financial research query:" text area, type your question (e.g., "What is the current P/E ratio of Apple Inc.?", "Summarize the key findings of the latest Fed meeting minutes.", "Perform a SWOT analysis for Tesla.").
    *   Click "Submit Query". The copilot will process your request, routing it to the appropriate handler, and display the response in the "Interaction History".

5.  **Simulate LLM API Failure:**
    *   Enter a query and click "Simulate LLM API Failure" to see how the application handles errors gracefully.

6.  **Navigate to the Dashboard:**
    *   Use the "Navigate" dropdown in the sidebar and select "Cost & Compliance Dashboard".

7.  **Generate Dashboard Data:**
    *   On the dashboard page, click "Simulate Batch Queries for Dashboard Analysis". This will run a batch of synthetic queries to populate the cost and compliance metrics, allowing you to visualize the application's performance and savings.

## Project Structure

```
quLab-research-copilot/
├── app.py                     # Main Streamlit application file
├── source.py                  # Backend logic: LLM integration, handlers, semantic cache, compliance logger, costs, prompts, etc.
├── requirements.txt           # Python dependencies
└── README.md                  # Project README file
# ├── data/                      # (Optional) Directory for RAG documents, lookup data, etc.
# └── .env                     # (Optional) For environment variables if using python-dotenv
```

*   `app.py`: Contains the Streamlit frontend UI, session state management, and orchestrates calls to the backend logic defined in `source.py`.
*   `source.py`: Encapsulates all the core intelligence of the copilot, including LLM client initialization, cost definitions, query routing logic, specialized handlers for different query types, the semantic cache implementation, and the compliance logger.

## Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Backend Logic**: Python
*   **AI/ML Framework**: [OpenAI API](https://openai.com/api) (GPT models for generation and embeddings)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
*   **Semantic Search**: Utilizes `text-embedding-3-small` for embeddings and cosine similarity (likely from `scikit-learn`) for cache lookups.
*   **Tokenization**: [Tiktoken](https://github.com/openai/tiktoken) (often used for accurate token counting with OpenAI models)

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if applicable, otherwise state "Proprietary" or "For Educational Use Only").

## Contact

For questions or feedback, please contact the QuantUniversity team:

*   **Website**: [QuantUniversity](https://www.quantuniversity.com/)
*   **Email**: info@quantuniversity.com
*   **LinkedIn**: [QuantUniversity LinkedIn](https://www.linkedin.com/company/quantuniversity)