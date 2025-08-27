# 📊 Semantic Search over Excel with FAISS + Gemini LLM

This project implements a **semantic search system** for Excel spreadsheets.  
You can ask **natural language questions** like *“What is total revenue?”* and the system will retrieve relevant spreadsheet cells, then synthesize a concise answer using **Google Gemini LLM**.

---

## 🚀 Features
- **Excel ingestion** → Converts spreadsheets into structured documents with metadata (`sheet | row | column`).
- **Local Embeddings** → Uses `sentence-transformers/all-MiniLM-L6-v2` for fast CPU/GPU embeddings.
- **Vector Search (FAISS)** → Performs cosine similarity search on normalized embeddings.
- **Caching** → SHA-1–based cache stored in memory + disk (`embed_cache.pkl`) for instant re-runs.
- **LLM Integration (Gemini 1.5 Flash)** → Generates concise answers grounded only in retrieved spreadsheet cells.
- **Batch Encoding** → Uses batch size of 256 for efficient embedding computation.

---

## 📂 Project Structure
```
.
├── semantic_search.py        # Main script
├── requirements.txt          # Dependencies
├── Sales Dashboard.xlsx      # Example input file
├── embed_cache.pkl           # Auto-generated embedding cache
└── README.md                 # Documentation
```

---

## ⚙️ Installation

### 1. Install dependencies
```bash
pip install pandas openpyxl numpy faiss-cpu sentence-transformers google-generativeai
```

### 2. Configure Gemini API key
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### 3. Add your Excel file
Put your workbook (e.g., `Sales Dashboard.xlsx`) in the project directory.

---

## ▶️ Usage

Run the script in Python (Colab or locally):

```python
XLSX_PATH = "Sales Dashboard.xlsx"
docs = load_excel_as_docs(XLSX_PATH)

idx = FastIndex()
idx.build(docs)

queries = [
    "What is total revenue?",
    "Show operating margin or any margin values.",
    "Budget vs actual differences for revenue or opex."
]

for q in queries:
    hits = idx.search(q, k=5)
    print("\n🔎", q)
    for s, d in hits:
        print(f"  • {d['uid']} | {d['context']} | value={d['value']} (sim={s:.3f})")
    print("\n💡 Answer:", answer_with_llm(q, hits))
```

**Example Output:**
```
🔎 What is total revenue?
  • Finance!R4C3 | Finance | 2024 | Revenue | value=200000 (sim=0.923)

💡 Answer:
Total revenue is 200,000.
```

---

## 🛠️ Technical Architecture
- **Excel Loader (`load_excel_as_docs`)** → Parses spreadsheets into `{uid, sheet, row, col, value, context}` documents.
- **Embedder (`embed_texts_local`)** → Converts contexts into normalized vectors with caching.
- **Cache** → Uses SHA-1 keys to avoid recomputation across runs.
- **Index (`FastIndex`)** → FAISS `IndexFlatIP` for similarity search.
- **LLM Answering (`answer_with_llm`)** → Synthesizes final answer using Gemini.

---

## 📈 Performance Notes
- **Batch size = 256** for faster embedding throughput.
- **Disk cache** enables near-instant re-runs when embeddings are reused.
- **FlatIP index** is exact and fast for small/medium data; for very large spreadsheets, approximate FAISS indexes (IVF, HNSW) are recommended.

---

## ⚠️ Limitations
- Large spreadsheets may require approximate search for scalability.
- Business-specific abbreviations are handled via context strings but not fine-tuned embeddings.
- LLM hallucination risk mitigated by strict prompting but still possible.

---

## ✅ Future Enhancements
- Support **multi-file search** across multiple spreadsheets.
- Implement **approximate FAISS indexes** for scaling to millions of rows.
- Add a **web-based UI** for interactive queries.
- Fine-tune embeddings for specific business terminology.

---

## 📜 License
MIT License – free for personal and commercial use.

---

## 🙌 Acknowledgments
- [SentenceTransformers](https://www.sbert.net/)  
- [FAISS](https://github.com/facebookresearch/faiss)  
- [Google Generative AI](https://ai.google.dev/)  
