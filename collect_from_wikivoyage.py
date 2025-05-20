import os, sys, json, threading, requests
from urllib.parse import quote
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://en.wikivoyage.org/w/api.php"
BASE = "https://en.wikivoyage.org/wiki/"

TARGET_SECS = {
    "regions", "cities", "other destinations",
    "cities and towns", "districts", "related pages"
}
MAX_DEPTH = 1 # How many links to open recursively within target sections
WORKERS   = 16 # Number of workers for parallelization

visited    = set()
visit_lock = threading.Lock()
chunks     = []
chunk_lock = threading.Lock()

def get_sections(page):
    try:
        resp = requests.get(API, params={
            "action":"parse","page":page,
            "prop":"sections","format":"json"
        }, timeout=10).json()
        secs = resp.get("parse", {}).get("sections", [])
        return [(s["index"], s["line"]) for s in secs]
    except Exception as e:
        print(f"    get_sections failed for '{page}': {e}")
        return []

def get_section_html(page, idx):
    try:
        resp = requests.get(API, params={
            "action":"parse","page":page,
            "prop":"text","section":idx,"format":"json"
        }, timeout=10).json()
        return resp.get("parse", {}).get("text", {}).get("*", "")
    except Exception as e:
        print(f"   get_section_html failed for '{page}' sec {idx}: {e}")
        return ""

def get_section_links(page, idx):
    try:
        resp = requests.get(API, params={
            "action":"parse","page":page,
            "prop":"links","section":idx,
            "format":"json","pllimit":"max","plnamespace":0
        }, timeout=10).json()
        links = resp.get("parse", {}).get("links", [])
        titles = [l["*"] for l in links if ":" not in l["*"]]
        return titles
    except Exception as e:
        print(f"   get_section_links failed for '{page}' sec {idx}: {e}")
        return []

def clean(html):
    return BeautifulSoup(html, "html.parser").get_text("\n", strip=True)

def save_chunk(page, title, text):
    with chunk_lock:
        chunks.append({
            "chunk_id": f"{len(chunks)+1:05d}",
            "wiki_url": BASE + quote(page.replace(" ","_")) + "#" + quote(title.replace(" ","_")),
            "text": text
        })

def crawl(page, depth):
    with visit_lock:
        if page in visited:
            return []
        visited.add(page)

    print(f"\n Crawling: {page}")
    # 1) get list of sections
    secs = get_sections(page)

    # 2) always grab "Introduction" (section 0)
    intro = clean(get_section_html(page, 0))
    if intro:
        save_chunk(page, "Introduction", intro)

    to_follow = set()
    for idx, sec in secs:
        html = get_section_html(page, idx)
        txt  = clean(html)
        if txt:
            save_chunk(page, sec, txt)
        if sec.lower() in TARGET_SECS:
            for link in get_section_links(page, idx):
                to_follow.add(link)

    print(f"  Chunks so far: {len(chunks)}")
    if depth >= MAX_DEPTH:
        return []
    return [(lnk, depth+1) for lnk in to_follow]

def main(start):
    queue = [(start, 0)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        while queue:
            futures = {ex.submit(crawl, p, d): (p,d) for p,d in queue}
            queue = []
            for f in as_completed(futures):
                queue.extend(f.result())

if __name__ == "__main__":
    country = sys.argv[1] if len(sys.argv)>1 else "India"
    main(country)
    print(f"\n Done: collected {len(chunks)} chunks.")
    
    os.makedirs("data", exist_ok=True)

    # save into data/
    out_path = os.path.join("data", f"wikivoyage_{country}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {out_path}")

    
