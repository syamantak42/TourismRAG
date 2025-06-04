##  Tourism RAG Agent (Proof of Concept)

This is a proof-of-concept (PoC) implementation of a **RAG-based travel and tourism assistant**. It collects travel information from [Wikivoyage](https://en.wikivoyage.org/) and uses it to answer user questions via a local LLM setup.

###  Included Scripts

1. **`collect_from_wikivoyage.py`**
   Scrapes Wikivoyage starting from a specified country page, recursively following internal links up to a user-defined depth. Outputs structured travel data.

2. **`TourismRAG_LMStudio.py`**
   Runs a minimal **Flask app** serving a local RAG pipeline. Both the LLM and embedding model are powered by **LMStudio**.

3. **`TourismRAG_LMStudio.py`**
   Runs a minimal **Flask app** serving a local RAG pipeline. The LLM is hosted locally through by **Ollama**, the embedding models are local as well.

### Data Folder Contents

* **`wikivoyage_Spain.json`**
  A JSON dictionary of travel content scraped from Wikivoyage for Spain.

* **`faiss_Spain.index`**
  A prebuilt FAISS vector database created from the above JSON for efficient semantic search (built with the embedding model **"text-embedding-nomic-embed-text-v1.5"**).
* **`faiss_Spain.index`**
  Another prebuilt FAISS vector database created from the same JSON but a different embedding model (**"BAAI/bge-small-en"**)

To run and access the local LLM install LMStudio from https://lmstudio.ai/ and download and run your favourite LLM locally
To run models through ollama download & install ollama from https://ollama.com/, pull the model and run it from ollama
