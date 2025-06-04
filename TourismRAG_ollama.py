import faiss
import numpy as np
import json
import os
import sys
import requests
from flask import Flask, request, jsonify, render_template_string, session
from flask_session import Session
from sentence_transformers import  SentenceTransformer, util

import gc

# Load a fast bi-encoder (small and CPU-friendly)
bi_encoder = SentenceTransformer("BAAI/bge-small-en")  # 'all-MiniLM-L6-v2'

def rerank(query, passages):
    query_emb = bi_encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    doc_embs = bi_encoder.encode(passages, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.dot_score(query_emb, doc_embs)[0].cpu().tolist()
    return scores

# ----- Embedding API -----
embedding_model = SentenceTransformer("BAAI/bge-small-en")  # 'all-MiniLM-L6-v2'

def get_embeddings(texts):
    return embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)



def split_chunks(chunks, min_len=500, max_len=5000, overlap=500):
    new_chunks = []
    part_num = 1
    i = 0

    # Merge short chunks forward
    while i < len(chunks):
        text = chunks[i]['text'].replace('\n', ' ').strip()
        chunk_id = str(chunks[i]['chunk_id'])
        link = chunks[i]['wiki_url']
        i += 1

        # If text is too short, keep adding next chunks
        while len(text) < min_len and i < len(chunks):
            next_text = chunks[i]['text'].replace('\n', ' ').strip()
            text += ' ' + next_text
            i += 1

        # If too long, split it into overlapping parts
        if len(text) > max_len:
            start = 0
            while start < len(text):
                end = start + max_len
                chunk_text = text[start:end].strip()
                if len(chunk_text) >= min_len:
                    new_chunks.append({
                        'chunk_id': f"{chunk_id}_{part_num}",
                        'text': chunk_text,
                        'wiki_url': link
                    })
                    part_num += 1
                start += (max_len - overlap)

        # If in acceptable range, save as-is
        elif len(text) >= min_len:
            new_chunks.append({
                'chunk_id': f"{chunk_id}_{part_num}",
                'text': text.strip(),
                'wiki_url': link
            })
            part_num += 1

    return new_chunks



# ----- Answer pipeline -----
def get_answer(user_query):

    model = "gemma3:4b"
    temperature = 0.7
    max_tokens = 400

    q_emb = get_embeddings([user_query])
    D, I = index.search(q_emb, k=5)
    candidates = [(i, id2text[i], id2link[i]) for i in I[0] if i in id2text]
    texts_only = [text for _, text, _ in candidates]
    scores = rerank(user_query, texts_only)
    ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])[:3]

    context = "\n\n".join(text for _, (_, text, _) in ranked)
    urls = [url for _, (_, _, url) in ranked]
    link_text = "\n".join(f"- {url}" for url in urls)

    prompt = f"""You are a helpful travel consultant, 
            highly knowledgeable on various places of interest in {country}.
            Below is a user query, accompanied by som relevant context. 
            Answer the following user query precisely, using the provided context.
            Your answer must be highly detailed, thorough and informative. 
            Provide all the information that the user may need to plan a trip.


Context:
{context}

Question: {user_query}
Answer:"""
    print(context)
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False  # disables streaming for easier parsing
            }
        )
        output = response.json().get("response", "").strip()
    except Exception as e:
        return f"Error querying Ollama LLM: {e}"

    return f"{output}\nFor more info, consult:\n{link_text}"


# ----- Flask app -----
app = Flask(__name__)
@app.template_filter('nl2br')
def nl2br(s):
    return s.replace('\n', '<br>\n')

app.secret_key = "1234567887654321"  # Needed for session
app.config["SESSION_TYPE"] = "filesystem"

reset_on_start = True  # global flag

@app.before_request
def clear_session_on_start():
    global reset_on_start
    if reset_on_start:
        session.clear()
        reset_on_start = False

Session(app)

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>Explore {{ country }}</title>
    <style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(to right, #fffde7, #e3f2fd);
        max-width: 900px;
        margin: auto;
        padding: 2em;
    }
    h1 {
        color: #e53935;
    }
    h2 {
        color: #3949ab;
    }
    textarea, button {
        width: 100%;
        padding: 1em;
        margin-top: 1em;
        font-size: 1em;
        border-radius: 8px;
        border: 1px solid #ccc;
    }
    button {
        background-color: #3949ab;
        color: white;
        font-weight: bold;
        cursor: pointer;
    }
    pre {
        white-space: pre-wrap;
        background: #ffffff;
        padding: 1em;
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    .conversation {
        margin-top: 2em;
        background: #fafafa;
        border-left: 4px solid #3949ab;
        padding: 1em;
    }
    .user {
        font-weight: bold;
        color: #1e88e5;
        margin-top: 1em;
    }
    .assistant-label {
        font-weight: bold;
        color: #6a1b9a;
        margin-top: 0.5em;
    }
    .assistant-message {
        white-space: pre-wrap;
        line-height: 1.5;
        margin-bottom: 1.5em;
        color: #333;
    }
    a {
        color: #1e88e5;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>

    <script>
        window.onload = () => {
            const textarea = document.querySelector("textarea");
            textarea.addEventListener("keydown", function (e) {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    this.form.submit();
                }
            });
        };
    </script>
</head>
<body>
    <h1>Discover {{ country }}</h1>
    <form method="POST">
        <textarea name="query" rows="4" placeholder="Ask anything about the country...">{{ query or "" }}</textarea>
        <button type="submit">Get Answer</button>
    </form>

    {% if history %}
    <div class="conversation">
        {% for item in history %}
            <p><span class="user">You:</span> {{ item.query }}</p>
            <p><span class="assistant-label">Assistant:</span></p>
            <div class="assistant-message">{{ item.answer | nl2br | safe }}</div>
            <hr>
        {% endfor %}
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index_page():
    if "history" not in session:
        session["history"] = []

    query = ""
    answer = ""
    if request.method == "POST":
        query = request.form.get("query", "")
        answer = get_answer(query)
        session["history"].append({"query": query, "answer": answer})
        session.modified = True  # To ensure session saves the update

    return render_template_string(HTML_TEMPLATE, query=query, answer=answer,
                                  history=session["history"], country=country)

@app.route("/ask", methods=["POST"])
def api_endpoint():
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "Query missing"}), 400
    answer = get_answer(user_query)
    return jsonify({"answer": answer})



if __name__ == "__main__":

    # ----- Load data -----
    country = sys.argv[1] if len(sys.argv) > 1 else "Spain"
    json_path = os.path.join("data", f"wikivoyage_{country}.json")

    print("Loading wikivoyage data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print("old chunks:")
    texts = [c['text'] for c in chunks]
    lengths = np.array([len(text) for text in texts])
    print("min length: ", lengths.min())
    print("max length: ", lengths.max())
    print("mean length: ", lengths.mean())
    new_chunks = split_chunks(chunks, max_len=5000, min_len=500)
    print("extracting elements...")
    texts = [c['text'] for c in new_chunks]
    ids = [int(c['chunk_id']) for c in new_chunks]
    links = [c['wiki_url'] for c in new_chunks]
    id2text = dict(zip(ids, texts))
    id2link = dict(zip(ids, links))
    print("new chunks:")
    lengths = np.array([len(text) for text in texts])
    print("min length: ", lengths.min())
    print("max length: ", lengths.max())
    print("mean length: ", lengths.mean())
    
    # ----- Reranker -----
    #reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ----- Load/ Create FAISS index -----
    faiss_path = os.path.join("data", f"faiss_{country}_2.index")


    if os.path.exists(faiss_path):
        print("Loading FAISS index from disk...")
        index = faiss.read_index(faiss_path)
    else:
        print("Creating embeddings (only once) ...")
        embeddings = get_embeddings(texts)
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
        index.add_with_ids(np.array(embeddings), np.array(ids))
        print("Saving FAISS database to disk...")
        faiss.write_index(index, faiss_path)
        del embeddings
        gc.collect()

    print("Starting up App...")
    app.run()
