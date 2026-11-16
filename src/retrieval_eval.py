"""Evaluation script comparing retrieval methods:
1. Lexical TF-IDF baseline (starter/baseline.py, budget 800)
2. Dense retrieval (BAAI/bge-small-en-v1.5 + ChromaDB Day 6, budget 800)
3. Hybrid Retrieval (Dense + Lexical RRF, budget 800)
4. Improvement 1: Hybrid RRF with Context Budget 1600
5. Improvement 2 (Chosen Strategy): Hybrid RRF with Expanded Budget 2400 (allows full 3 paragraph chunks in context)

Outputs trace and summaries for Dev and Test splits.
"""
import argparse
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path
import chromadb
from fastembed import TextEmbedding


def tokenize(text):
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def hit(chunk, gold):
    return any(
        chunk.get('source_id') == g.get('source_id') and
        chunk.get('page') == g.get('page') and
        g.get('evidence', '') in chunk.get('text', '')
        for g in gold
    )


def pack_context(results, budget=800):
    blocks, ids, seen = [], [], set()
    for c in results:
        cid = c['id']
        if cid in seen:
            continue
        seen.add(cid)
        block = f"[{cid} | {c.get('source', '')} | page {c.get('page', '')}]\n{c.get('text', '')}"
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(cid)
    return '\n\n'.join(blocks), ids


class TfidfRetriever:
    def __init__(self, chunks):
        self.chunks = {c['id']: c for c in chunks}
        self.chunk_list = chunks
        counts = [Counter(tokenize(c['text'])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts):
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1
        return {term: value / norm for term, value in vector.items()}

    def search(self, question, top_k=10, filters=None):
        query = self._vector(Counter(tokenize(question)))
        results = []
        for chunk, vector in zip(self.chunk_list, self.vectors):
            if filters and filters.get('section') and chunk.get('metadata', {}).get('section') != filters['section']:
                continue
            score = sum(value * vector.get(term, 0) for term, value in query.items())
            if score > 0:
                results.append({**chunk, 'score': score, 'strategy': 'tfidf'})
        return sorted(results, key=lambda c: (-c['score'], c['id']))[:top_k]


class DenseRetriever:
    def __init__(self, chroma_path, collection_name='strategy_B', embed_model='BAAI/bge-small-en-v1.5'):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.col = self.client.get_collection(collection_name)
        self.embedder = TextEmbedding(model_name=embed_model)

    def search(self, question, top_k=10, filters=None):
        q_vec = list(self.embedder.embed([question]))[0].tolist()
        where_cond = {'section': filters['section']} if filters and filters.get('section') else None
        res = self.col.query(query_embeddings=[q_vec], n_results=top_k, where=where_cond)
        
        results = []
        if res and res['ids'] and res['ids'][0]:
            for idx in range(len(res['ids'][0])):
                cid = res['ids'][0][idx]
                meta = res['metadatas'][0][idx]
                doc = res['documents'][0][idx]
                dist = res['distances'][0][idx]
                score = 1.0 - dist
                results.append({
                    'id': cid,
                    'source_id': meta.get('document_id', 'llm_survey_2024'),
                    'source': meta.get('source', 'llm_survey.pdf'),
                    'page': meta.get('page'),
                    'text': doc,
                    'metadata': meta,
                    'score': score,
                    'strategy': 'dense'
                })
        return results


class HybridRrfRetriever:
    def __init__(self, dense_retriever: DenseRetriever, tfidf_retriever: TfidfRetriever, rrf_k=60):
        self.dense = dense_retriever
        self.tfidf = tfidf_retriever
        self.rrf_k = rrf_k

    def search(self, question, top_k=10, filters=None):
        dense_hits = self.dense.search(question, top_k=top_k * 3, filters=filters)
        tfidf_hits = self.tfidf.search(question, top_k=top_k * 3, filters=filters)
        
        scores = Counter()
        doc_map = {}
        for rank, hit_item in enumerate(dense_hits):
            cid = hit_item['id']
            scores[cid] += 1.0 / (self.rrf_k + rank + 1)
            doc_map[cid] = hit_item
            
        for rank, hit_item in enumerate(tfidf_hits):
            cid = hit_item['id']
            scores[cid] += 1.0 / (self.rrf_k + rank + 1)
            if cid not in doc_map:
                doc_map[cid] = hit_item
                
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        results = []
        for cid in sorted_ids:
            chunk = doc_map[cid].copy()
            chunk['score'] = scores[cid]
            chunk['strategy'] = 'hybrid_rrf'
            results.append(chunk)
        return results


def run_experiment(retriever, questions, strategy_name, budget=800, top_k=3, use_filter=False):
    traces = []
    times = []
    for q in questions:
        filters = q.get('filter') if use_filter else None
        start = time.perf_counter()
        results = retriever.search(q['question'], top_k=top_k, filters=filters)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        
        context, ids = pack_context(results, budget=budget)
        gold = q.get('gold', [])
        marked = [{**c, 'hit': hit(c, gold)} for c in results]
        
        hit1 = bool(marked and marked[0]['hit'])
        hit3 = any(c['hit'] for c in marked[:3])
        context_hit = any(c['hit'] and c['id'] in ids for c in marked)
        
        traces.append({
            'query_id': q['id'],
            'split': q['split'],
            'question': q['question'],
            'strategy': strategy_name,
            'top_k': top_k,
            'budget_chars': budget,
            'gold': gold,
            'results': marked,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': hit1,
            'hit3': hit3,
            'context_hit': context_hit
        })
        
    n = len(traces)
    summary = {
        'strategy': strategy_name,
        'split': questions[0]['split'] if questions else 'unknown',
        'n': n,
        'hit1': sum(t['hit1'] for t in traces),
        'hit1_str': f"{sum(t['hit1'] for t in traces)}/{n} ({sum(t['hit1'] for t in traces)/n*100:.1f}%)" if n else "0/0",
        'hit3': sum(t['hit3'] for t in traces),
        'hit3_str': f"{sum(t['hit3'] for t in traces)}/{n} ({sum(t['hit3'] for t in traces)/n*100:.1f}%)" if n else "0/0",
        'context_hit': sum(t['context_hit'] for t in traces),
        'context_hit_str': f"{sum(t['context_hit'] for t in traces)}/{n} ({sum(t['context_hit'] for t in traces)/n*100:.1f}%)" if n else "0/0",
        'avg_context_chars': statistics.mean(t['context_chars'] for t in traces) if traces else 0,
        'median_ms': statistics.median(times) if times else 0,
        'service_cost_usd': 0.0
    }
    return traces, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple retrieval strategies for Day 07")
    parser.add_argument("--chunks", default="data/chunks.json")
    parser.add_argument("--questions", default="data/retrieval_questions.json")
    parser.add_argument("--chroma", default="/Users/lethanhtunggmail.com/Vinuni-collab/K5-L3-Day06-DataIngestionEngineering/results/chromadb")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.chunks) as f:
        chunks = json.load(f)
    with open(args.questions) as f:
        questions = json.load(f)

    dev_qs = [q for q in questions if q['split'] == 'dev']
    test_qs = [q for q in questions if q['split'] == 'test']

    print(f"Loaded {len(chunks)} chunks, {len(dev_qs)} dev questions, {len(test_qs)} test questions.")

    tfidf_retriever = TfidfRetriever(chunks)
    dense_retriever = DenseRetriever(chroma_path=args.chroma, collection_name='strategy_B')
    hybrid_retriever = HybridRrfRetriever(dense_retriever, tfidf_retriever, rrf_k=60)

    # Formal Experimentation Table conforming to BENCHMARK.md & GUIDE.md:
    # 1. Baseline: TF-IDF (budget 800)
    # 2. Method 2: Dense (budget 800)
    # 3. Method 3: Hybrid RRF (budget 800)
    # 4. Improvement 1: Hybrid RRF with Context Budget 1600
    # 5. Improvement 2 (Chosen): Hybrid RRF with Context Budget 2400 (top_k=5)
    
    experiments = [
        ("dev", "tfidf_b800", tfidf_retriever, dev_qs, 800, 3, False),
        ("dev", "dense_b800", dense_retriever, dev_qs, 800, 3, False),
        ("dev", "hybrid_rrf_b800", hybrid_retriever, dev_qs, 800, 3, False),
        ("dev", "hybrid_rrf_b1600", hybrid_retriever, dev_qs, 1600, 3, False),
        ("dev", "hybrid_rrf_top5_b2400", hybrid_retriever, dev_qs, 2400, 5, False),
        # Held-out Test evaluation after locking configurations
        ("test", "tfidf_b800", tfidf_retriever, test_qs, 800, 3, False),
        ("test", "hybrid_rrf_top5_b2400", hybrid_retriever, test_qs, 2400, 5, False),
    ]

    all_summaries = []
    for split, name, ret, qs, budget, top_k, use_filter in experiments:
        print(f"\n--- Running {name} on {split} ---")
        traces, summary = run_experiment(ret, qs, strategy_name=name, budget=budget, top_k=top_k, use_filter=use_filter)
        print(f"Summary: Hit@1={summary['hit1_str']}, Hit@3={summary['hit3_str']}, Context Evidence={summary['context_hit_str']}, Median time={summary['median_ms']:.2f}ms")
        
        all_summaries.append(summary)
        trace_file = out_dir / f"retrieval_trace_{split}_{name}.json"
        with open(trace_file, "w") as f:
            json.dump(traces, f, indent=2)
            
    summary_file = out_dir / "retrieval_benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n=== ALL RETRIEVAL EXPERIMENTS COMPLETED ===")


if __name__ == "__main__":
    main()
