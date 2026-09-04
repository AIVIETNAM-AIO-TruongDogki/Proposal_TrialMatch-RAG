# TrialMatch-RAG — Kế hoạch pipeline end-to-end

**Cập nhật:** 2026-08-30 · **Trạng thái:** Phase 0–4 xong · Phase 5–7 code xong, chưa chạy đủ ·
Phase 8 code xong + smoke test · Phase 9–11 chưa bắt đầu.

Tài liệu này mô tả **toàn bộ đường đi từ dữ liệu thô đến kết quả cuối**, cái gì đã xong, cái gì còn lại,
và **ba chỗ tắc** sẽ quyết định dự án có về đích được hay không.

Đặc tả từng phase: [`specs/`](../specs/README.md) · Tổng kết Phase 4: [`phase4-tong-ket.md`](phase4-tong-ket.md)

---

## 1. Câu hỏi nghiên cứu và tại sao pipeline có hình dạng này

> *Truy hồi và suy luận có nhận thức về điều kiện tham gia (eligibility-aware) có cải thiện việc ghép
> bệnh nhân với thử nghiệm lâm sàng so với truy hồi từ khoá hoặc ngữ nghĩa thông thường không?*

Toàn bộ thiết kế xoay quanh **một ý duy nhất**:

> **Liên quan y khoa KHÔNG đồng nghĩa với đủ điều kiện tham gia.**

Một thử nghiệm ung thư phổi giai đoạn III rất *liên quan* tới bệnh nhân ung thư phổi giai đoạn III —
nhưng nếu thử nghiệm loại trừ người đã hoá trị platinum, thì bệnh nhân từng dùng cisplatin **không đủ
điều kiện**, dù khớp ngữ nghĩa gần như hoàn hảo. Truy hồi thông thường không diễn đạt nổi sự phân biệt
này. Hệ thống này tồn tại để làm nó **tường minh và đo được**.

Pipeline vì thế tách ba câu hỏi mà RAG thông thường gộp làm một:

```
thử nghiệm có liên quan không?  →  bệnh nhân có thể đủ điều kiện không?  →  bằng chứng nào?
        (Phase 3–7)                        (Phase 8)                        (Phase 9)
```

### Bằng chứng đã đo được cho luận điểm này

Đây không còn là giả thuyết. Đã đo ở **hai tầng độc lập**:

| Tầng | Khi chất lượng truy hồi tăng | Ô nhiễm (trial bị loại trừ lọt top-10) |
|---|---|---|
| **Index** (Phase 3) | elig nDCG@10 0,1112 → 0,2399 | 0,1400 → **0,2840** |
| **Truy vấn** (Phase 4) | mọi biến thể trích xuất | chuẩn hoá: 0,335 → **0,396–0,427** |

Càng giỏi ở *relevance*, vấn đề *eligibility* càng **tệ đi**. Phase 8 sinh ra để đảo ngược điều đó.

---

## 2. Bốn nguyên tắc bất biến

Đây là ràng buộc thiết kế, không phải khuyến nghị. Vi phạm bất kỳ điều nào là phá hỏng đóng góp.

1. **Ba trạng thái, không bao giờ nhị phân.** Mỗi tiêu chí phải rơi vào `satisfied` / `violated` /
   `unverifiable`. Gộp `unverifiable` vào hai cái kia là xoá bỏ đóng góp.
2. **Thông tin thiếu phải giữ nguyên là thiếu.** LLM không được suy diễn, áp đặt hay đoán thuộc tính
   bệnh nhân không có trong bệnh án. *Vắng bằng chứng được báo cáo, không được giải quyết.*
3. **Mọi kết luận đều có căn cứ.** Mỗi đánh giá phải trích dẫn nguyên văn văn bản tiêu chí nó dựa vào —
   kiểm chứng bằng máy, không phải bằng lời hứa.
4. **Hỗ trợ quyết định, không phải ra quyết định.** Đây là nguyên mẫu nghiên cứu. Kết luận cuối cùng
   thuộc về bác sĩ / điều phối viên thử nghiệm.

---

## 3. Sơ đồ pipeline

```
rawdata/  375.580 trial XML (ClinicalTrials.gov 2021-04-27)
    │
    │  ┌─ Phase 0 ─ môi trường, fetch dữ liệu ──────────────── ✅ XONG
    │  │  src/fetch_data.py  →  Hugging Face (1,3 GB tar.gz)
    ▼
[Phase 1] Xây kho dữ liệu chuẩn ─────────────────────────────── ✅ XONG
    src/corpus/{parse,store,build_db}.py
    → data/trials.db (2,8 GB) · 375.580 trial · 4.985.262 tiêu chí
      kèm span_start/span_end trỏ ngược vào criteria_raw gốc
    │
[Phase 2] Bộ khung đánh giá ──────────────────────────────────── ✅ XONG
    src/eval/{data,metrics,sig,score,run_io}.py
    → HAI HỌ ĐỘ ĐO, luôn báo cáo cạnh nhau:
        chính thức (excluded=1 được tính điểm) — để so với bài báo
        eligibility (chỉ eligible) + contamination@k — để đo luận điểm
    → tập test 2022 KHÓA, chỉ chấm MỘT LẦN ở Phase 11
    │
[Phase 3] Truy hồi từ khoá ── BẬC 1 ──────────────────────────── ✅ XONG
    src/retrieval/{export_corpus,build_index,bm25,tune}.py
    → 4 index: base / crit / fields / crit_fields
    → THẮNG: crit_fields, k1=1,8 b=1,0 → elig nDCG@10 = 0,2399
    → Recall@1000 = 0,4176  ← TRẦN CỨNG của mọi tầng phía sau
    │
[Phase 4] Trích xuất hồ sơ bệnh nhân ─────────────────────────── ✅ XONG
    src/extraction/{schema,gemini,extract,verify,query}.py
    → 75/75 bệnh án, grounding 99,3%, tuổi/giới 100%
    → 3 biến thể truy vấn: prof / prof_narr / hyde
    │
[Phase 5] Truy hồi ngữ nghĩa ── BẬC 2 ───────────────────────── 🔨 CODE XONG
    src/dense/{encode,search,bench}.py
    → 3 encoder × 2 biến thể trên subsample 46.162 doc
    → mã hoá model thắng cuộc trên toàn 375.580 trial
    │
[Phase 6] Hợp nhất lai ── BẬC 3 ─────────────────────────────── 🔨 CODE XONG
    src/retrieval/fusion.py  (RRF + weighted, bảng phần bù)
    │
[Phase 7] Xếp hạng lại ── BẬC 4 ─────────────────────────────── 🔨 CODE XONG
    src/rerank/rerank.py  (3 họ reranker, đo cả độ trễ)
    │
[Phase 8] Suy luận điều kiện ── BẬC 5 ── ĐÓNG GÓP CHÍNH ─────── 🔨 CODE + SMOKE TEST
    src/reasoning/{schema,verify,aggregate,reason,score}.py
    → mỗi tiêu chí: satisfied / violated / unverifiable + 2 trích dẫn
    → ⛔ TẮC: cần 1.665 lần gọi API (xem §6)
    │
[Phase 9] Sinh văn bản có căn cứ ────────────────────────────── ⬜ CHƯA
[Phase 10] Hệ thống end-to-end ──────────────────────────────── ⬜ CHƯA
[Phase 11] Ablation + viết báo cáo ──────────────────────────── ⬜ CHƯA
```

---

## 4. Thang ablation — trạng thái thật

Mỗi bậc phải đo được so với bậc dưới. Đây là số **thật** đã chạy, không phải mục tiêu:

| Bậc | Cấu hình | elig nDCG@10 | contam@10 ↓ | Recall@1000 |
|---|---|---|---|---|
| — | BM25 mặc định Lucene | 0,1112 | 0,1400 | 0,3078 |
| — | + tinh chỉnh k1/b | 0,1600 | 0,1840 | 0,4592 |
| — | + gộp tiêu chí vào index | 0,2070 | 0,2507 | 0,4535 |
| **1** | **+ field boost + tinh chỉnh lại** (`bm25_best`) | **0,2399** | 0,2840 | 0,4176 |
| 1b | + truy vấn `prof` (Phase 4) | 0,2470 | 0,2653 | 0,5249 |
| 1b | + truy vấn `prof_narr` (Phase 4) | **0,2782** | 0,3240 | 0,5307 |
| 1b | + truy vấn `hyde` (Phase 4) | 0,2145 | 0,1813 | 0,5141 |
| **2** | dense | *chưa đo* | | |
| **3** | lai (hybrid) | *chưa đo* | | |
| **4** | + xếp hạng lại | *chưa đo* | | |
| **5** | + suy luận eligibility | *chưa đo* | **← phải GIẢM** | |

**Cách đọc bảng này:** cột `contam@10` tăng đều ở mọi bậc cải thiện relevance. Bậc 5 là bậc **duy nhất**
được kỳ vọng làm nó giảm. Nếu bậc 5 không giảm contamination, đóng góp của đề tài chưa được chứng minh —
và điều đó phải được báo cáo, không phải tinh chỉnh cho tới khi đẹp.

---

## 5. Việc còn lại, theo thứ tự

### Phase 5 — truy hồi ngữ nghĩa (đang dở)

Đã xong: code + **self-test 3 encoder ĐẠT** (bge-m3 0,94 vs 0,43 · qwen3 0,88 vs 0,28 ·
medcpt 0,72 vs 0,39, xác nhận dùng đúng 2 model riêng cho query/document).

Còn lại:
1. Benchmark 6 tổ hợp trên subsample 46.162 doc (~2 giờ GPU). Đã đo tốc độ thật: **52 chunk/s** cho
   bge-m3 → mỗi tổ hợp `base` ~16 phút, `crit` ~40 phút.
2. Chọn model theo **union-recall với BM25**, không theo nDCG. Lý do: Recall@1000 = 0,4176 là trần cứng;
   một model thua nDCG nhưng tìm ra 15% trial mà BM25 **không bao giờ** trả về thì đáng giá hơn một
   model hoà điểm nhưng trả về cùng tập tài liệu. Phase 6 tiêu thụ **phần bù**, không tiêu thụ điểm.
3. Mã hoá model thắng trên toàn 375.580 trial (~2 giờ), tìm kiếm bằng matmul chính xác (769 MB fp16,
   vừa GPU 8 GB). **Không dùng FAISS** — sai số xấp xỉ sẽ trộn vào chênh lệch giữa các bậc.

⚠️ **Điểm subsample bị thổi phồng** (trial đã chấm chiếm 57% subsample nhưng chỉ 7% corpus thật) —
chỉ dùng để xếp hạng ứng viên với nhau, **không bao giờ đặt cạnh 0,2399 của Phase 3**.

### Phase 6 — hợp nhất lai

Code đã chạy thật (dùng 2 run BM25 làm đầu vào thử): RRF nâng elig nDCG@10 0,2399 → **0,2919**.

Còn lại: chạy với dense thật, và xuất **bảng phần bù** — deliverable bắt buộc, không phải tuỳ chọn.
Nếu hai chân chồng lấn gần hoàn toàn thì việc hợp nhất mua được rất ít, và luận điểm "cả hai chân đều
chịu lực" phải được nói lại cho đúng. Đó là kết quả phải báo cáo.

### Phase 7 — xếp hạng lại

Code xong, chờ GPU rảnh để self-test 3 reranker.

Lưu ý thiết kế: **ba họ, ba cách gọi khác nhau**. MedCPT và bge là cross-encoder thật (điểm = logit).
Qwen3-Reranker **không phải** — nó là LLM được hỏi "có liên quan không?" và điểm là
`logP(yes) − logP(no)`. Áp khuôn cross-encoder lên nó sẽ ra điểm vô nghĩa mà không ném lỗi nào.

Phải báo cáo **độ trễ cạnh chất lượng**: một reranker thêm 0,01 nDCG với giá 40× độ trễ là kết quả *âm*
cho một hệ thống định dùng được thật.

### Phase 8 — suy luận điều kiện (⛔ điểm tắc chính)

Code xong và **đã smoke test thành công**: 98/99 quyết định qua kiểm chứng grounding (1,0% bị vứt).

Ví dụ thật từ smoke test — đúng loại ca mà cả đề tài sinh ra để bắt:

> **Tiêu chí:** *"histologically confirmed anaplastic astrocytoma **on the tentorium** at first relapse"*
> **Bệnh án:** *"The tumor is located in the **T-L spine**, unresectable anaplastic astrocytoma"*
> **Nhãn:** `violated` — đúng bệnh, sai vị trí. BM25 xếp trial này hạng cao vì trùng tên bệnh.

Ba nhánh đã kiểm chứng chạy được: ba trạng thái, ablation ép buộc (bỏ `unverifiable`), và ba luật gộp
(`strict` / `lenient` / `count`).

**Còn lại: chạy thật trên 1.665 lần gọi.** Xem §6.

### Phase 9–11 — chưa bắt đầu

- **Phase 9** sinh giải thích có căn cứ cho bác sĩ đọc. Phụ thuộc Phase 8.
- **Phase 10** đóng gói end-to-end (API, README, chạy được từ máy mới).
- **Phase 11** chấm tập test 2022 **một lần duy nhất** + viết báo cáo ablation.

---

## 6. Ba chỗ tắc

### 6.1. ✅ Hạn ngạch API — ĐÃ GỠ (sửa 31/08/2026)

**Mục này trước đây kết luận sai.** Nó lấy trần `20 request/ngày` — đo bằng lỗi 429 thật, nhưng
**trên `gemini-3.6-flash`** — rồi áp cho `gemini-3.5-flash-lite`, model thực tế được chọn. Hạn ngạch
được ghi rõ là **theo từng model**, nên phép suy rộng đó không có căn cứ.

**Artifact của chính dự án bác bỏ nó.** Ngày 30/08/2026, `gemini-3.5-flash-lite` phục vụ:

```
23:05  trích xuất 75 bệnh án ÷ batch 5 = 15 lần gọi
23:14  HyDE       75 bệnh án ÷ batch 5 = 15 lần gọi
                                  tổng = 30 lần gọi, đều thành công
```

Trần 20/ngày sẽ chặn lượt HyDE giữa chừng. Nó chạy trọn ⇒ trần của Lite **> 30**.

| Chế độ | Số lần gọi | Ở mức 500/ngày (Lite) |
|---|---|---|
| Gọi từng tiêu chí | 27.045 | ~54 ngày |
| **Gộp cả trial (chia nhỏ ở mức 30 tiêu chí)** | **1.665** | **~3,3 ngày** |
| Gộp + xoay vòng 2 model Lite | 1.665 | **~1,7 ngày** |

**Kết luận đảo ngược: bậc miễn phí CHẠY ĐƯỢC Phase 8 trên toàn bộ 75 bệnh án.** Không cần trả phí,
không cần giảm N xuống top-5, không cần cắt số bệnh án — sức mạnh thống kê của tập dev giữ nguyên.

**Mức độ bằng chứng, ghi rõ:** trần `>30` là *đã chứng minh* bằng artifact. Con số *500* là do người
dùng báo và nhất quán với bằng chứng đó, nhưng **chưa được xác nhận bằng một lỗi 429 nêu `quotaValue`
cho đúng model này**. Nếu thực tế thấp hơn 500, phép chia lại đơn giản: `1.665 ÷ trần thật`.

**Khuyến nghị: trả phí.** Ở 1.665 lần gọi với model Flash-Lite, chi phí thực tế rất nhỏ so với việc
đánh đổi phạm vi khoa học của đóng góp chính.

Lệnh `--estimate` in ra con số này rồi dừng, **không gọi API**:
```bash
PYTHONPATH=. .venv/bin/python -m src.reasoning.reason --estimate
```

### 6.2. Trần cứng Recall@1000 = 0,4176

**Hơn một nửa số trial đủ điều kiện không bao giờ được truy hồi trả về.** Không tầng nào phía sau cứu
được: rerank không thể xếp hạng thứ nó chưa từng thấy, suy luận eligibility cũng vậy.

Đây là lý do Phase 5 chọn model theo **union-recall** chứ không theo nDCG, và là con số phải in ngay
đầu mọi bảng Phase 7.

### 6.3. Không có nhãn vàng mức tiêu chí

qrels của TREC chỉ có nhãn mức **trial** (0/1/2). Không thể chấm trực tiếp từng tiêu chí. Phase 8 vì thế
dùng đường (a) của `specs/08`: gộp quyết định tiêu chí thành quyết định trial rồi đối chiếu qrels.

**Hệ quả phải nói rõ trong báo cáo:** phép đo này đo **cả** luật gộp lẫn chất lượng suy luận, không tách
được hai thứ. Muốn tách cần bộ nhãn tự gán tay (~1.000 tiêu chí) — deliverable vươn tới, chưa có.

---

## 7. Bài học vận hành đã trả giá để có

Ghi lại vì đắt để khám phá lại, rẻ để viết một lần.

1. **Quota Gemini tính theo project × model, không theo key.** Tạo key mới trong cùng project **không**
   thêm quota (đã thử, 429 ngay). Key từ project khác thì có. Model khác cũng có quota riêng — đó là
   đường ra khi một model cạn.
2. **Xoay vòng model hợp lệ cho sản xuất hàng loạt, KHÔNG hợp lệ trong một nhánh ablation.** Nếu 75 mô
   tả HyDE đến từ nhiều model, nhánh đó đo "HyDE với một mớ model hỗn hợp" — không tái lập, không quy
   được chênh lệch. Vì vậy `query.py` có `--hyde-model` (một model cho cả 75) chứ không có bộ xoay model.
3. **`response_schema` của Gemini gãy ở ~40 phần tử** (n=35 chạy, n=40 trả 400 INVALID_ARGUMENT). Không
   phải giới hạn token mà là độ phức tạp schema. 10,3% trial có hơn 30 tiêu chí (tối đa 76), nên chia
   nhỏ ở mức 30 là **bắt buộc**, không phải tối ưu.
4. **Grounding gần như không phân biệt được model** (99,9 / 99,3 / 98,7 cho ba model Gemini) — vì trích
   ít đi thì grounding dễ hơn. Cột phân biệt thật là **độ phủ** và **recall phủ định**. Một báo cáo chỉ
   có grounding sẽ kết luận nhầm rằng ba model tương đương.
5. **Code chưa chạy là code chưa đúng.** Phiên này tìm ra 3 lỗi tiềm ẩn chỉ lộ ra khi chạy thật, trong
   những file đã được đánh dấu "BUILT AND PASSING" từ trước.

---

## 8. Thứ tự thực thi đề xuất

```
1. Phase 5 benchmark subsample          ~2 giờ GPU, miễn phí       [chạy được ngay]
2. Phase 5 mã hoá toàn corpus           ~2 giờ GPU, miễn phí       [sau bước 1]
3. Phase 6 fusion + bảng phần bù        vài phút, miễn phí         [sau bước 2]
4. Phase 7 benchmark 3 reranker         ~30 phút GPU, miễn phí     [sau bước 3]
5. ── QUYẾT ĐỊNH HẠN NGẠCH ──                                      [cần bạn quyết]
6. Phase 8 chạy đầy đủ                  1.665 lần gọi API          [sau bước 5]
7. Phase 9 sinh văn bản có căn cứ                                  [sau bước 6]
8. Phase 10 đóng gói end-to-end                                    [sau bước 7]
9. Phase 11 chấm test 2022 MỘT LẦN + báo cáo                       [cuối cùng]
```

Bước 1–4 **không tốn một lần gọi API nào** và có thể chạy ngay. Chỉ bước 5 cần bạn quyết định.
