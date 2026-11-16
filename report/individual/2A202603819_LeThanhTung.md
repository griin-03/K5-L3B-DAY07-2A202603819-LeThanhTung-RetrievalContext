# Báo cáo Day 7 — Agentic RAG & Retrieval Context Engineering

**Học viên:** Lê Thanh Tùng  
**MSSV:** 2A202603819  
**Lớp:** L3B  
**Repo bài nộp:** https://github.com/griin-03/K5-L3B-DAY07-2A202603819-LeThanhTung-RetrievalContext  

---

## 1. Dữ liệu và benchmark

- **Nguồn & Quyền sử dụng:** Bài báo khoa học tổng quan *"A Comprehensive Overview of Large Language Models"* (Naveed et al., arXiv: [2307.06435](https://arxiv.org/abs/2307.06435)), phân phối công khai theo giấy phép Open Access không độc quyền của arXiv.
- **Số trang & Số chunk:** 35 trang nội dung kỹ thuật cốt lõi (Sections 1 đến 8), gồm **234 chunks** theo chiến lược paragraph-aware chunking (tối đa 800 ký tự/chunk, giữ trọn ý ngữ nghĩa và heading).
- **Vector store & Embedding model:**
  - Embedding model: `BAAI/bge-small-en-v1.5` (FastEmbed ONNX runtime), số chiều **384**, chuẩn hóa L2, khoảng cách Cosine.
  - Vector store: **ChromaDB v1.5.9** Persistent Store (`results/chromadb`) kế thừa trực tiếp từ Day 6 (collection `strategy_B`).
- **Agent model & Cách chạy:** 
  - Reasoning Controller tích hợp vào Agent Loop `starter/agentic.py` với khả năng lập kế hoạch truy xuất, kiểm tra độ bao phủ ngữ cảnh, tổng hợp câu trả lời kèm citation `[chunk-id]` và cơ chế từ chối đáng tin cậy (Abstention).
  - Môi trường chạy: Cục bộ trên macOS ARM64 với Python 3.10 virtual environment, chi phí API: **$0.00**.
- **Phân chia benchmark:** 
  - Tổng cộng **24 câu hỏi** chuẩn:
    - **Tập Dev (17 câu):** 15 câu có đáp án và bằng chứng kiểm chứng (`Q01`–`Q15`) + 2 câu thiếu bằng chứng (`Q21`, `Q22`).
    - **Tập Test (7 câu):** 5 câu có đáp án và bằng chứng kiểm chứng (`Q16`–`Q20`) + 2 câu thiếu bằng chứng (`Q23`, `Q24`).
- **Tiêu chí xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Câu có evidence:* Đoạn trích bằng chứng `gold.evidence` phải tồn tại nguyên văn trong văn bản của trang vật lý nguồn `gold.page`. Chấm Grounded Answer: `0` = sai hoặc thiếu bằng chứng; `1` = đúng một phần hoặc bằng chứng gián tiếp; `2` = đúng, đủ và trích dẫn chuẩn xác.
  - *Câu thiếu evidence:* Là các câu hỏi gần chủ đề (AI, doanh thu, hạ tầng phần cứng tản nhiệt, bảo mật) nhưng hoàn toàn không có trong nội dung khảo sát kỹ thuật của tài liệu. Yêu cầu agent phải nhận biết thiếu thông tin và từ chối trả lời (`status: "abstain"`), không sinh hallucination.

---

## 2. Retrieval và thử nghiệm

### Bảng so sánh các phương pháp và cấu hình cải thiện

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline (TF-IDF)** | Lexical TF-IDF, top-3, no filter, budget 800 chars | Từ khóa chính xác giúp tìm đúng đoạn chứa thuật ngữ hiếm nhưng trượt ngữ cảnh câu dài | dev | 10/15 (66.7%) | 14/15 (93.3%) | 6/15 (40.0%) | 0.31ms / $0.00 |
| **Phương pháp 2 (Dense)** | Dense BGE-small, top-3, no filter, budget 800 chars | Vector embedding nắm bắt ngữ nghĩa tổng quát nhưng dễ phân tán trên câu hỏi ngắn | dev | 7/15 (46.7%) | 10/15 (66.7%) | 4/15 (26.7%) | 5.22ms / $0.00 |
| **Phương pháp 3 (Hybrid RRF)** | Dense + TF-IDF, RRF (k=60), top-3, budget 800 chars | Hợp nhất thứ hạng Reciprocal Rank Fusion kết hợp ưu điểm từ khóa và ngữ nghĩa | dev | 11/15 (73.3%) | 13/15 (86.7%) | 6/15 (40.0%) | 5.80ms / $0.00 |
| **Cải thiện 1 (Budget 1600)** | Hybrid RRF, top-3, no filter, **budget 1600 chars** | Tăng context budget từ 800 lên 1600 ký tự giúp đóng gói đủ 2 chunks chứa evidence | dev | 11/15 (73.3%) | 13/15 (86.7%) | 11/15 (73.3%) | 5.71ms / $0.00 |
| **Cải thiện 2 / Chọn (Top5 + B2400)** | Hybrid RRF, **top-5 candidate**, **budget 2400 chars** | Mở rộng ứng viên top-5 và ngân sách 2400 ký tự giúp đưa trọn vẹn cả 3 top chunks vào prompt | dev | **11/15 (73.3%)** | **13/15 (86.7%)** | **13/15 (86.7%)** | 5.96ms / $0.00 |
| **Baseline (TF-IDF)** | Lexical TF-IDF, top-3, no filter, budget 800 chars | Kiểm thử độc lập trên tập test giữ nguyên baseline | test | 1/5 (20.0%) | 4/5 (80.0%) | 0/5 (0.0%) | 0.30ms / $0.00 |
| **Cấu hình chọn (Top5 + B2400)** | Hybrid RRF, top-5, budget 2400 chars | Cấu hình đã khóa sau dev: kiểm thử độc lập trên test đạt kết quả vượt trội | test | 1/5 (20.0%) | **4/5 (80.0%)** | **4/5 (80.0%)** | 5.72ms / $0.00 |

### Phân tích 2 lỗi trên tập Dev và lý do chọn cấu hình
1. **Lỗi 1 (Hiện tượng rơi rụng bằng chứng do Context Budget quá hẹp ở baseline 800 chars):**
   - *Hiện tượng:* Với baseline TF-IDF và Dense ở ngân sách 800 ký tự, mặc dù Evidence Hit@3 đạt từ 66.7% đến 93.3%, nhưng Context Evidence Rate chỉ đạt từ 26.7% đến 40.0%. Đặc biệt trên tập test, baseline TF-IDF có Context Evidence Rate rớt xuống 0/5 (0.0%).
   - *Nguyên nhân:* Các đoạn văn paragraph chunk trung bình dài 750–800 ký tự. Khi chunk top-1 chiếm hết 750 ký tự, chunk top-2 và top-3 chứa gold evidence bị cắt bỏ hoàn toàn khỏi context cuối gửi cho Agent.
   - *Khắc phục:* Mở rộng ngân sách lên 2400 ký tự (Cải thiện 2) cho phép đóng gói trọn vẹn cả 3 đoạn văn có điểm xếp hạng cao nhất, đưa Context Evidence Rate trên tập test từ 0.0% lên **80.0%** (4/5 câu).
2. **Lỗi 2 (Nhiễu từ khóa đa nghĩa và bất cân xứng giữa Dense và Lexical):**
   - *Hiện tượng:* Câu hỏi `Q07` (JAX library) và `Q06` (Megatron-LM) bị cạnh tranh bởi các bảng danh mục ở đầu tài liệu.
   - *Khắc phục:* Thuật toán hợp nhất Reciprocal Rank Fusion (RRF với hằng số $k=60$) giúp cân bằng điểm số, ngăn chặn trường hợp một bộ truy xuất cho điểm số quá cao lấn át hoàn toàn bộ còn lại.

- **Lý do chọn cấu hình:** Cấu hình **Hybrid RRF (top-5, budget 2400 ký tự)** được chọn vì tối ưu hóa đồng thời cả độ bao quát của bằng chứng trong context (Context Evidence Rate đạt 86.7% trên dev, 80.0% trên test) và tốc độ phản hồi cực nhanh (5.72ms trên chip M-series).

---

## 3. Agent và đánh giá

### Thiết kế Agent
- **Tool sử dụng:** Tool tìm kiếm lai `search(query, filters, top_k)` tích hợp bộ điều phối Hybrid RRF (Dense + TF-IDF).
- **Số lượt gọi & Dừng:** Giới hạn tối đa `max_searches = 3`. Nếu sau 2 lượt truy xuất ngữ cảnh tích lũy vẫn không có sự tương đồng hoặc phát hiện câu hỏi ngoài phạm vi, controller chủ động dừng và trả về trạng thái từ chối (`abstain`).
- **Cách chọn context & Xử lý lỗi:** Lọc trùng lặp chunk ID, ưu tiên chunk có rank cao nhất, đóng gói context kèm định dạng chuẩn `[chunk_id | source | page]`.

### Bảng tổng hợp chỉ số Agent (theo `agent_metrics.py`)

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer (0 / 1 / 2) và TB | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| **Agent Hybrid (Dev)** | 17/17 (100%) | 1.00 calls/q | [0: 2, 1: 2, 2: 11] → **1.60 / 2.0** | 11/15 (73.3%) | 15/15 (100%) | 2/2 (100%) | 0/15 (0.0%) |
| **Agent Hybrid (Test)** | 7/7 (100%) | 1.00 calls/q | [0: 1, 1: 1, 2: 3] → **1.40 / 2.0** | 3/5 (60.0%) | 5/5 (100%) | 2/2 (100%) | 0/5 (0.0%) |

### Chi tiết đánh giá từng câu trên tập Test (Audit Log)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| **Q16** | Có đáp án (PAWS dataset) | `search("PAWS dataset designed to evaluate...")` → Chunks `p31_7`, `p32_0`, `p28_3` | Trả lời đầy đủ về cross-lingual paraphrase identification kèm citation `[para_llm_survey_2024_p32_0]` | **2 / 2**: Đúng, đủ và trích dẫn hoàn toàn chính xác chunk gold `p32_0`. Citation support: True. |
| **Q17** | Có đáp án (Truthful-QA) | `search("Truthful-QA benchmark measure...")` → Chunks `p26_5`, `p31_6`, `p30_0` | Trả lời về đánh giá ngôn ngữ tổng quát kèm citation `[para_llm_survey_2024_p26_5]` | **0 / 2**: Bị trượt chunk định nghĩa Truthful-QA ở p32 do bảng benchmark tổng hợp ở p26 chiếm điểm cao hơn. Citation support: False. |
| **Q18** | Có đáp án (Explainability) | `search("making LLMs more explainable...")` → Chunks `p33_4`, `p1_1`, `p3_0` | Trả lời về phân tích văn bản pháp lý kèm citation `[para_llm_survey_2024_p33_4]` | **1 / 2**: Đúng về khía cạnh khả năng giải thích (explanation) trong ngữ cảnh pháp lý nhưng trích dẫn gián tiếp thay vì Section 8 (p35). Citation support: False. |
| **Q19** | Có đáp án (Tokenization) | `search("tokenization schemes commonly used...")` → Chunks `p4_0`, `p8_1`, `p15_0` | Trả lời đầy đủ wordpiece, BPE, unigramLM kèm citation `[para_llm_survey_2024_p4_0]` | **2 / 2**: Đúng, đủ và trích dẫn hoàn toàn chính xác. Citation support: True. |
| **Q20** | Có đáp án (LLM-QAT) | `search("LLM-QAT train quantized student...")` → Chunks `p21_6`, `p22_0`, `p21_4` | Trả lời về Quantization-Aware Training và chưng cất tri thức kèm citation `[para_llm_survey_2024_p22_0]` | **2 / 2**: Đúng, đủ và trích dẫn chuẩn xác chunk gold. Citation support: True. |
| **Q23_unanswerable** | Thiếu bằng chứng (Claude 3.5) | `search("hyperparameter settings GPU cluster...")` → Không tìm thấy căn cứ | Trả lời: *"Không đủ bằng chứng trong tài liệu để trả lời câu hỏi này."*, Citations: `[]` | **Abstention đúng (True)**: Nhận biết câu hỏi ngoài phạm vi khảo sát, từ chối chính xác không hallucinate. |
| **Q24_unanswerable** | Thiếu bằng chứng (Chất làm mát GPU) | `search("chemical synthesis cooling liquid B200...")` → Không có căn cứ | Trả lời: *"Không đủ bằng chứng trong tài liệu để trả lời câu hỏi này."*, Citations: `[]` | **Abstention đúng (True)**: Nhận biết đề tài hóa học tản nhiệt nằm ngoài tài liệu LLMs, từ chối chính xác. |

---

## 4. Tổng kết và Đánh đổi kỹ thuật

- **Đường dẫn artifacts & traces:**
  - Dữ liệu câu hỏi & ca kiểm thử: `data/retrieval_questions.json`, `data/agent_cases.json`.
  - Trace truy xuất các phương pháp: `results/retrieval_trace_*.json` và bảng tổng hợp `results/retrieval_benchmark_summary.json`.
  - Trace agent và nhãn audit: `results/agent_traces_*.json`, `results/reviews_*.json`, `results/agent_metrics_*.json`.
- **Đánh đổi kỹ thuật chính:**
  - *Context Budget vs. Prompt Latency:* Tăng budget từ 800 lên 2400 ký tự giúp giải quyết triệt để nút thắt cổ chai về bằng chứng context (tăng từ 0% lên 80% trên test), đưa điểm Grounded Answer trung bình đạt **1.40 / 2.0** trên test và **1.60 / 2.0** trên dev.
  - *Hybrid RRF:* Mang lại sự ổn định vượt trội so với việc dùng đơn lẻ một cơ chế truy xuất.
  - *Cơ chế Abstention:* Đạt độ chính xác tuyệt đối **100% (2/2 dev, 2/2 test)** mà không có trường hợp nào bị từ chối nhầm (False Abstention: 0/15 dev, 0/5 test).
- **Hạn chế:** Quy mô bộ test (5 câu có đáp án, 2 câu thiếu bằng chứng) là chuẩn theo yêu cầu bài lab 4 giờ; để mở rộng ứng dụng thực tế trên tài liệu lớn, cần bổ sung thêm mô hình Re-ranking đa tầng (Cross-Encoder).
