# -*- coding: utf-8 -*-
!pip -q install pandas openpyxl numpy faiss-cpu sentence-transformers google-generativeai

import os, hashlib, pickle, numpy as np, pandas as pd, faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

os.environ["GOOGLE_API_KEY"] = "****************"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
LLM_MODEL = "models/gemini-1.5-flash"
EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

CACHE_PATH = "/content/embed_cache.pkl"
EMBED_CACHE = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        EMBED_CACHE = pickle.load(f)

def _key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def embed_texts_local(texts: list[str]) -> np.ndarray:
    """Batch encode with local model + cache."""
    to_encode, order = [], []
    vecs = [None]*len(texts)
    for i, t in enumerate(texts):
        k = _key(t or " ")
        if k in EMBED_CACHE:
            vecs[i] = EMBED_CACHE[k]
        else:
            to_encode.append(t or " ")
            order.append(i)
    if to_encode:
        embs = EMB_MODEL.encode(to_encode, batch_size=256, normalize_embeddings=True) 
        for i, v in zip(order, embs):
            vecs[i] = v.astype("float32")
            EMBED_CACHE[_key(texts[i] or " ")] = vecs[i]
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(EMBED_CACHE, f)
    return np.stack(vecs).astype("float32")

def load_excel_as_docs(xlsx_path: str):
    sheets = pd.read_excel(xlsx_path, sheet_name=None, header=0)
    docs = []
    for sname, df in sheets.items():
        if df.empty: continue
        df = df.reset_index(drop=True).copy()
        row_headers = df.iloc[:, 0].astype(str)
        col_headers = df.columns.astype(str)

        for r in range(len(df)):
            for c in range(1, df.shape[1]):
                val = df.iat[r, c]
                if pd.isna(val): continue
                context = f"{sname} | {row_headers.iloc[r]} | {col_headers[c]}"
                docs.append({
                    "uid": f"{sname}!R{r+2}C{c+1}",
                    "sheet": sname,
                    "row": r+2, "col": c+1,
                    "value": str(val),
                    "context": context,
                })
    return docs

class FastIndex:
    def __init__(self):
        self.meta = []
        self.index = None
        self.dim = None

    def build(self, docs):
        self.meta = docs
        texts = [d["context"] for d in docs]
        vecs = embed_texts_local(texts)          
        self.dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)
        print(f"Indexed {len(docs)} items (dim={self.dim}).")

    def search(self, query, k=5):
        qv = embed_texts_local([query])[0].reshape(1, -1) 
        sims, idx = self.index.search(qv, k)
        sims, idx = sims.ravel(), idx.ravel()
        return [(float(sims[i]), self.meta[idx[i]]) for i in range(len(idx))]


def answer_with_llm(query, hits):
    rows = [f"- {d['sheet']} [{d['row']},{d['col']}] | {d['value']} | {d['context']}" for _, d in hits[:5]]
    context_block = "\n".join(rows)
    prompt = f"""Answer the question using ONLY these spreadsheet cells.

Question: {query}

Cells:
{context_block}

Return a concise answer. If insufficient info, say so.
"""
    model = genai.GenerativeModel(LLM_MODEL)
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", "") or "").strip()

from google.colab import files
uploaded = files.upload()  
XLSX_PATH = list(uploaded.keys())[0]

docs = load_excel_as_docs(XLSX_PATH)
idx  = FastIndex(); idx.build(docs)

for q in [
    "What is total revenue?",
    "Show operating margin or any margin values.",
    "Budget vs actual differences for revenue or opex.",
]:
    hits = idx.search(q, k=5)
    print("\n🔎", q)
    for s, d in hits:
        print(f"  • {d['uid']} | {d['context']} | value={d['value']} (sim={s:.3f})")
    print("\n💡 Answer:")
    print(answer_with_llm(q, hits))
