# Day 07 — Agentic RAG & Retrieval Context Engineering

**Học viên:** Lê Thanh Tùng  
**MSSV:** 2A202603819  
**Lớp:** L3B  
**Repo:** [https://github.com/griin-03/K5-L3B-DAY07-2A202603819-LeThanhTung-RetrievalContext](https://github.com/griin-03/K5-L3B-DAY07-2A202603819-LeThanhTung-RetrievalContext)

Dự án phát triển hệ thống Agentic RAG kế thừa kho tri thức và vector store từ Day 6 (arXiv:2307.06435 - 35 trang nội dung kỹ thuật, 234 chunks), hiện thực 3 phương pháp retrieval (Lexical TF-IDF, Dense ChromaDB, Hybrid RRF), triển khai 2 cấu hình cải thiện (Context Budget 1600 & Metadata Filter), và xây dựng agent có khả năng tra cứu, trả lời có citation và từ chối khi thiếu bằng chứng (Abstention).

---

## 1. Cấu trúc thư mục

```text
K5-L3-Day07-Retrieval-Context-Engineering/
├── STUDENT.md                              # Thông tin học viên và repo
├── README.md                               # Hướng dẫn setup, chạy và cấu trúc dự án
├── data/
│   ├── chunks.json                        # 234 chunks kế thừa từ Day 6
│   ├── retrieval_questions.json           # 20 câu hỏi có gold evidence phục vụ so sánh ranking
│   └── agent_cases.json                   # 24 câu hỏi benchmark (20 có evidence + 4 thiếu evidence)
├── src/
│   ├── retrieval_eval.py                  # Đo lường so sánh Lexical, Dense, Hybrid RRF, Budget
│   └── agent_eval.py                      # Triển khai Agent Loop, Controller và xuất trace
├── starter/
│   ├── agentic.py                         # Khung Agent Loop chính thức
│   ├── baseline.py                        # TF-IDF Lexical Baseline
│   └── agent_metrics.py                   # Script tổng hợp nhãn đánh giá độc lập
├── results/
│   ├── retrieval_benchmark_summary.json   # Bảng tổng hợp các phương pháp retrieval
│   ├── retrieval_trace_*.json             # Chi tiết trace top-3 của các phương pháp
│   ├── agent_traces_*.json                # Trace từng lượt chạy và tool call của Agent
│   ├── reviews_*.json                     # Nhãn audit thủ công Grounded Answer & Citation
│   └── agent_metrics_*.json               # Kết quả tổng hợp chỉ số Agent
└── report/
    └── individual/
        └── 2A202603819_LeThanhTung.md     # Báo cáo cá nhân chi tiết theo mẫu
```

---

## 2. Hướng dẫn thiết lập và chạy

### Kích hoạt môi trường và chạy unit tests:
```bash
# Kích hoạt môi trường ảo Python 3.10
source ../K5-L3-Day06-DataIngestionEngineering/.venv/bin/activate

# Kiểm tra unit tests
python3 -m unittest discover -s tests -v
```

### Chạy so sánh các phương pháp Retrieval:
```bash
PYTHONPATH=. python3 src/retrieval_eval.py
```

### Chạy và đánh giá Agent:
```bash
# Chạy agent trên cả dev và test:
PYTHONPATH=.:starter python3 -c "
from src.agent_eval import HeuristicReasoningController, run_agent_workflow
from src.retrieval_eval import HybridRrfRetriever, DenseRetriever, TfidfRetriever
import json

with open('data/chunks.json') as f: chunks = json.load(f)
with open('data/agent_cases.json') as f: cases = json.load(f)
tfidf = TfidfRetriever(chunks)
dense = DenseRetriever('../K5-L3-Day06-DataIngestionEngineering/results/chromadb')
hybrid = HybridRrfRetriever(dense, tfidf, rrf_k=60)
controller = HeuristicReasoningController()

for split in ['dev', 'test']:
    sc = [c for c in cases if c['split'] == split]
    traces = run_agent_workflow(sc, hybrid, controller, budget=1600)
    with open(f'results/agent_traces_{split}.json', 'w') as f: json.dump(traces, f, indent=2)
"

# Tổng hợp chỉ số agent bằng script chuẩn:
PYTHONPATH=.:starter python3 starter/agent_metrics.py \
  --cases data/agent_cases.json \
  --traces results/agent_traces_test.json \
  --reviews results/reviews_test.json \
  --split test \
  --output results/agent_metrics_test.json
```

---

## 3. Tóm tắt kết quả Benchmark

### Bảng 1: So sánh Retrieval & Context

| Cấu hình | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian TB |
|---|---|---|---|---|---|
| **Baseline TF-IDF (b800)** | dev | 10/15 (66.7%) | 14/15 (93.3%) | 6/15 (40.0%) | 0.30ms |
| **Dense BGE-small (b800)** | dev | 7/15 (46.7%) | 10/15 (66.7%) | 4/15 (26.7%) | 5.41ms |
| **Hybrid RRF (b800)** | dev | 10/15 (66.7%) | 13/15 (86.7%) | 6/15 (40.0%) | 5.72ms |
| **Cải thiện 1: Hybrid RRF (b1600)** | dev | 10/15 (66.7%) | 13/15 (86.7%) | **10/15 (66.7%)** | 5.82ms |
| **Cải thiện 2: Filter + B1600** | dev | 10/15 (66.7%) | 13/15 (86.7%) | **10/15 (66.7%)** | 5.72ms |
| **Baseline TF-IDF (b800)** | test | 1/5 (20.0%) | 4/5 (80.0%) | 0/5 (0.0%) | 0.29ms |
| **Cấu hình chọn (Hybrid B1600)** | test | 1/5 (20.0%) | 2/5 (40.0%) | **2/5 (40.0%)** | 5.62ms |

### Bảng 2: Chỉ số đánh giá Agent trên tập Test

- **Tool use:** 7/7 (100%)
- **Số lượt gọi trung bình:** 1.00 call/query
- **Grounded Answer:** [0: 2, 1: 1, 2: 2] (Điểm trung bình: **1.00 / 2.0**)
- **Citation hợp lệ:** 2/5 (40.0%)
- **Tỷ lệ câu có citation:** 5/5 (100%)
- **Abstention đúng (câu thiếu bằng chứng):** 2/2 (100%)
- **Từ chối nhầm (false abstention):** 0/5 (0.0%)

Chi tiết phân tích sâu về các ví dụ lỗi, đánh đổi và giới hạn xem tại:
👉 [Báo cáo cá nhân Day 7](report/individual/2A202603819_LeThanhTung.md)
