# CyberBot

CyberBot is a cybersecurity chatbot built using Meta LLaMA LLM and VectorStore. It generates embeddings and uses LLM reasoning to answer cybersecurity-related questions, providing the source of the information. The project leverages Retrieval-Augmented Generation (RAG) and Chainlit for creating a ChatGPT-like UI.

## Features

- **Meta LLaMA LLM**: Utilizes Meta LLaMA for language model capabilities.
- **VectorStore**: Stores and retrieves embeddings for efficient querying.
- **RAG**: Combines retrieval and generation for accurate and context-aware responses.
- **Chainlit UI**: Provides an interactive ChatGPT-like user interface.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Dash10107/CyberBot.git
    cd CyberBot
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Chatbot

To run the chatbot locally, use the following command:
```bash
chainlit run model.py
```

### Changing Embeddings

If you need to change the embeddings, run the `ingest.py` script:
```bash
python ingest.py
```


