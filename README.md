##  Tourism RAG Agent (Proof of Concept)

This is a proof-of-concept (PoC) implementation of a **RAG-based travel and tourism assistant**. It collects travel information from [Wikivoyage](https://en.wikivoyage.org/) and uses it to answer user questions via a local LLM setup.

###  Included Scripts

1. **`collect_from_wikivoyage.py`**
   Scrapes Wikivoyage starting from a specified country page, recursively following internal links up to a user-defined depth. Outputs structured travel data.

2. **`TourismRAG_LMStudio.py`**
   Runs a minimal **Flask app** serving a local RAG pipeline. Both the LLM and embedding model are powered by **LMStudio** (no external API needed).

### Data Folder Contents

* **`wikivoyage_Spain.json`**
  A JSON dictionary of travel content scraped from Wikivoyage for Spain.

* **`faiss_Spain.index`**
  A prebuilt FAISS vector database created from the above JSON for efficient semantic search.
