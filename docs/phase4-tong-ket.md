# Phase 4 — Trích xuất hồ sơ bệnh nhân: tổng kết

**Ngày:** 2026-08-30 · **Tập dữ liệu:** TREC Clinical Trials 2021 (dev, 75 bệnh án) ·
**Model:** `gemini-3.6-flash` (trích xuất) · `gemini-3.5-flash` (sinh HyDE — xem §9)

Đặc tả gốc: [`specs/04-patient-profile-extraction.md`](../specs/04-patient-profile-extraction.md).
Quyết định đổi backend: [`docs/decisions/phase4-gemini-backend.md`](decisions/phase4-gemini-backend.md).

---

## 1. Mục tiêu và vị trí trong hệ thống

Phase 4 biến **bệnh án tự do** thành **hồ sơ lâm sàng có cấu trúc**, phục vụ hai nơi khác nhau:

- **Truy hồi (Phase 4 bước 4)** — dùng các thực thể trích được để dựng truy vấn BM25 tốt hơn.
- **Suy luận điều kiện tham gia (Phase 8)** — cần biết bệnh nhân *có gì* và *không có gì* để kết luận
  từng tiêu chí là `satisfied` / `violated` / `unverifiable`.

Hai mục đích này kéo schema về hai hướng ngược nhau, và đó là lý do có quy ước dưới đây.

## 2. Schema: ba trạng thái, không phải hai

| Trạng thái | Ý nghĩa | Ví dụ |
|---|---|---|
| `present` | bệnh án nói CÓ | `"severe aortic stenosis"` |
| `negated` | bệnh án nói KHÔNG có | `"no history of diabetes"` |
| *(vắng mặt)* | bệnh án KHÔNG nhắc tới | mục không xuất hiện trong danh sách |

Ba quy ước then chốt trong [`src/extraction/schema.py`](../src/extraction/schema.py):

1. **`status` chỉ có `present` | `negated`** — không có giá trị `absent`, vì không thể trích dẫn bằng
   chứng cho thứ chưa từng được nhắc đến. Vắng mặt được biểu đạt bằng cách **không xuất hiện**.
2. **Cấm `null`.** Nếu cho phép, ta sẽ nhận cả `{"age": null}` lẫn `{"age": {"value": null}}` và mất khả
   năng phân biệt "đã đọc và không thấy" với "chưa đọc".
3. **Mọi giá trị đều bắt buộc có `evidence`** — trích nguyên văn từ bệnh án. Đây là thứ biến lời hứa
   "mọi kết luận đều có căn cứ" thành một **phép đo** thực sự (xem `verify.py`).

Gộp `negated` vào "vắng mặt" là lỗi tai hại nhất ở phase này: *"bệnh nhân không bị tiểu đường"* là một
sự thật lâm sàng có thể **thoả mãn** tiêu chí loại trừ; *"bệnh án không nói gì về tiểu đường"* thì không
thoả mãn gì cả, nó phải thành `unverifiable` ở Phase 8.

## 3. Đổi backend: Ollama (cục bộ) → Gemini API

Quyết định ban đầu là **chạy hoàn toàn cục bộ** (Ollama, trần 8 GB VRAM), với lập luận y khoa: bệnh án
không rời khỏi máy. Quyết định này **đã bị đảo ngược** ở Phase 4.

**Đánh đổi, nói thẳng:** bệnh án giờ được gửi lên API của Google. Chấp nhận được *ở đây* vì topic của
TREC 2021/2022 là dữ liệu benchmark công khai đã khử định danh — **không phải** dữ liệu bệnh nhân thật.
Quyết định này **phải được xem xét lại**, không được kế thừa âm thầm, trước khi mở rộng sang Phase 8/9
hoặc bất kỳ triển khai nào với dữ liệu thật.

Ba điều học được khi triển khai (chi tiết trong decision doc):

- **`gemini-2.5-flash` đã biến mất với tài khoản mới** — trả về 404, chính lỗi của Google gợi ý
  `gemini-3.6-flash`. *Kiểm chứng model bằng cách gọi thật, đừng tin cái tên đã ghi từ trước.*
- **`response_schema` của Gemini là tập con OpenAPI, không phải JSON Schema đầy đủ** —
  `additionalProperties` bị từ chối thẳng; `_to_gemini_schema()` lọc đúng khoá đó và không đụng gì khác.
- **Xoay vòng 3 key KHÔNG nhân ba thông lượng nếu các key chung một project.** Quota tính theo *project*.
  Xoay key trong cùng project để lấy thêm quota là vô ích; tạo key từ **project khác** mới có quota độc lập.

## 4. Gọi API theo lô

Free tier giới hạn **5 request/phút** và **20 request/ngày/project/model**. Gọi từng bệnh án một cần 75
request — bất khả thi. `extract.py` gom **5 bệnh án/lần gọi** (mặc định `--batch-size 5`), giảm ~5× số
request.

**Rủi ro và cách xử lý.** Nhồi nhiều bệnh nhân vào một prompt tạo nguy cơ model trộn lẫn thông tin giữa
các bệnh nhân — vi phạm trực tiếp nguyên tắc "mọi kết luận phải có căn cứ đúng nguồn". Hai lớp bảo vệ:

1. **Prompt** (`prompts/extraction_batch_addendum.txt`) yêu cầu xử lý từng bệnh nhân độc lập, và *thà bỏ
   sót còn hơn đoán* khi không rõ chi tiết thuộc về ai.
2. **Cơ học:** mỗi hồ sơ trả về mang `index` riêng. Index **thiếu / trùng / ngoài khoảng** chỉ làm hỏng
   **đúng bệnh nhân đó**, không bao giờ gán nhầm hồ sơ của người này cho người khác.

**Phân biệt hai loại thất bại** — điểm này khác với hành vi ban đầu:

| Loại lỗi | Có ghi cache? | Hệ quả |
|---|---|---|
| Hạ tầng (429, mạng) | **Không** | Lần chạy sau tự động thử lại, không cần `--force` |
| Model trả JSON sai / khớp index hỏng | Có | Ghi lại như một thất bại đã đo, vì chạy lại cùng đầu vào hiếm khi tự sửa |

## 5. Kết quả trích xuất: 75/75 bệnh án

Chạy hết trong **285 giây** (15 lô), trung bình **3,87 s/bệnh án**.

Chỉ số từ [`src/extraction/verify.py`](../src/extraction/verify.py)
(`results/_extract_bench.2021.json`):

| Chỉ số | Kết quả | Ngưỡng yêu cầu | Đạt? |
|---|---|---|---|
| Hợp lệ schema | **100,0 %** | ≥ 95 % | ✅ |
| Grounding (evidence là chuỗi con thật) | **99,92 %** | ≥ 90 % | ✅ |
| Localized (evidence ≤ 30 từ) | **100,0 %** | — | ✅ |
| Độ chính xác tuổi | **100,0 %** | ≥ 95 % | ✅ |
| Độ chính xác giới tính | **100,0 %** | ≥ 95 % | ✅ |
| Độ phủ (số giá trị/bệnh án) | 16,9 | — | — |
| Bắt được phủ định | 80,0 % | — | xem §6 |
| Bẫy bịa đặt | **1** | 0 | ⚠ xem §6 |

Chỉ **1 giá trị duy nhất bị loại** vì không trích dẫn được: topic `2021_21`, model gán
`prior_treatments: "medications"` với evidence `"no history of using any medications"` — đây là câu
**phủ định**, không phải trị liệu đã dùng. Bộ lọc grounding bắt đúng.

## 6. Kiểm tra thủ công 25 bệnh án

Grounding cơ học là điều kiện **cần**, không phải **đủ**: nó xác nhận *evidence* có thật, nhưng không xác
nhận *`name` suy ra được từ evidence đó*. Vì vậy phải đọc tay. 25 topic được chọn: 15 topic có dấu hiệu
phủ định + 10 topic rải đều.

### 6.1. Không có ca bịa đặt nào

Kiểm tra 440 giá trị trong 25 topic: **92 ca (21 %) có `name` không nằm trong chính `evidence`**. Đọc
từng ca cho thấy **tuyệt đại đa số là mở rộng viết tắt hợp lệ** — đúng thứ cần cho truy hồi:

| `name` | `evidence` | Đánh giá |
|---|---|---|
| coronary artery disease | `CAD` | ✅ mở rộng đúng |
| polycystic ovary syndrome | `PCOS` | ✅ |
| anti-dsDNA antibodies | `anti-dsDNA Ab+` | ✅ |
| upper gastrointestinal bleeding | `UGIB` | ✅ |

Không tìm thấy trường hợp nào kiểu *"chẩn đoán thay vì trích xuất"* (lỗi từng thấy ở model 3B:
`name="hypertrophic cardiomyopathy"` từ evidence chỉ nói `"left ventricular hypertrophy..."`).

### 6.2. Phát hiện mới: evidence đôi khi **quá ngắn** để tự chứng minh

Bốn ca đáng ngờ nhất, sau khi đối chiếu bệnh án gốc, đều **đúng sự thật** nhưng **trích dẫn quá hẹp**:

| Topic | `name` | `evidence` | Bệnh án thật sự nói |
|---|---|---|---|
| `2021_16` | sildenafil | `"inpatient trial"` | *"compassionate **sildenafil** trial… tolerated an inpatient trial"* |
| `2021_5` | cataract surgery | `"s/p surgery"` | *"**Cataracts** s/p surgery"* |
| `2021_5` | polyp resection | `"s/p resection"` | *"Colon **polyps** s/p resection"* |
| `2021_10` | mastocytosis flare | `"flare"` | *"systemic **mastocytosis**, with flares…"* |

Đây là **mặt đối xứng** của lỗi đã được lường trước. `verify.py` đã đề phòng evidence **quá dài** (trích
cả đoạn văn → grounding 100 % mà không định vị được gì, nên mới có cột `localized`). Nhưng chưa ai lường
trước evidence **quá ngắn**: `localized` đạt 100 % chính vì các trích dẫn đều ngắn — trong đó có những
trích dẫn ngắn tới mức không tự chứng minh được nhãn của nó.

Không phải lỗi sai sự thật, nhưng **có ảnh hưởng tới Phase 9** (sinh văn bản có căn cứ): nếu hiển thị
evidence `"inpatient trial"` cho bác sĩ để giải thích vì sao ghi nhận `sildenafil`, đó là bằng chứng
không thuyết phục. Nên cân nhắc thêm ràng buộc **độ dài tối thiểu theo ngữ cảnh** (ví dụ: evidence phải
chứa được `name` hoặc dạng viết tắt của nó) ở Phase 9.

Trường hợp tương tự với phủ định — `2021_5` gán `orthopnea`, `palpitations` là `negated` với evidence chỉ
là chính từ đó. Nhãn **đúng** (bệnh án viết *"notable for absence of chest pain, dyspnea on exertion,
paroxysmal nocturnal dyspnea, orthopnea, palpitations…"*) nhưng evidence tách rời khỏi chữ `absence of`.

### 6.3. "Bắt phủ định 80 %" là con số bị đánh giá thấp

6/30 topic có dấu hiệu phủ định nhưng model không gán `negated` nào. Đọc tay cả 6:

| Topic | Cụm khớp regex | Có phải phủ định lâm sàng thật? |
|---|---|---|
| `2021_15` | `"CT abdomen **without** contrast"` | ❌ mô tả quy trình chụp CT, không phải phủ định |
| `2021_3` | `"ambulating independently **without** difficulty"` | ❌ khẳng định tích cực về vận động |
| `2021_9` | `"went 20 years **without** another seizure"` | ❌ tiền sử, không phải tình trạng hiện tại bị phủ định |
| `2021_26` | `"tenderness… **without** rebound"` | ⚠ khám thực thể, ranh giới |
| `2021_70` | `"family history is **negative for** any psychologic problems"` | ⚠ phủ định **tiền sử gia đình**, không phải bệnh nhân |
| `2021_4` | `"lumbar puncture and bone marrow biopsy were **negative for** any involvement"` | ✅ **phủ định thật — model bỏ sót** |

Chỉ **1/6** là bỏ sót thật sự. Bốn ca là **dương tính giả của regex** `NEG_CUES` (từ `without` bắt cả
"without contrast", "without difficulty"). Nghĩa là **recall phủ định thực tế cao hơn 80 % đáng kể** —
con số 80 % đo *cả chất lượng của regex phát hiện cue*, không chỉ đo model.

### 6.4. Bẫy bịa đặt `2021_14`: cả nhãn vàng lẫn model đều có vấn đề

`verify.py` đánh dấu `2021_14` là bẫy, với lý do "bệnh án thật sự KHÔNG nói giới tính". Model trả về
`sex: "female"` → bị tính là **bịa đặt**. Đọc kỹ cho thấy tình huống phức tạp hơn:

- **Nhãn vàng sai.** Bệnh án **có** nói giới tính: *"**She** has had decreased appetite, PO intake,
  energy level at home"* (vị trí 566). Regex `gold_age_sex()` chỉ đọc **220 ký tự đầu** nên không thấy.
  → `2021_14` **không phải** bẫy bịa đặt như tài liệu mô tả; ghi chú này cần sửa.
- **Model vẫn sai về lập luận.** Evidence model đưa ra là `"Daughter"` — **có con gái không chứng minh
  bệnh nhân là nữ**. Kết luận đúng, căn cứ sai.

Đây chính xác là lỗ hổng mà kiểm tra cơ học không bắt được: evidence là chuỗi con thật (`grounded` ✅),
đủ ngắn (`localized` ✅), kết luận đúng — nhưng **suy luận không hợp lệ**.

## 6.5. So sánh model: Flash vs Flash-Lite

Hạn ngạch tính theo **từng model**, và model Lite bị giới hạn nhẹ hơn — nên câu hỏi *"model rẻ hơn có
trích xuất đủ tốt không?"* là câu hỏi về **ngân sách Phase 8**, không chỉ về chất lượng. Đây cũng chính
là phép so sánh mà `specs/04` thiết kế từ đầu (bảng 6 model) nhưng chưa từng chạy vì bảng đó dựng cho
model cục bộ, đã bị thay thế.

Cả 3 chạy trên **cùng prompt, cùng schema, cùng `--batch-size 5`, cùng 75 topic** — chỉ đổi model.

| model | schema | ground | local | tuổi | giới | **cover** | **neg** | s/gọi | Phase 8 |
|---|---|---|---|---|---|---|---|---|---|
| `gemini-3.6-flash` | 100 % | **99,9 %** | 100 % | **100 %** | **100 %** | **16,9** | **80,0 %** | 3,87 s | 29,1 h |
| `gemini-3.5-flash-lite` | 100 % | 99,3 % | 100 % | **100 %** | **100 %** | 13,0 | 50,0 % | **1,36 s** | **10,2 h** |
| `gemini-3.1-flash-lite` | 100 % | 98,7 % | 100 % | *93,3 %* ❌ | *93,2 %* ❌ | 8,9 | 43,3 % | **1,33 s** | **10,0 h** |

### Grounding gần như nhau — và đó là cái bẫy

Nhìn cột `ground` (99,9 / 99,3 / 98,7) thì ba model như nhau. Nhưng **grounding cao mà độ phủ thấp thì
dễ đạt một cách tầm thường**: trích ít đi thì ít cơ hội sai hơn. Hai cột thật sự phân biệt là `cover` và
`neg`.

Phân rã độ phủ theo từng trường (giá trị/bệnh án):

| model | conditions | biomarkers | prior_treat | labs | comorbid | trùng lặp chéo trường |
|---|---|---|---|---|---|---|
| `3.6-flash` | **7,5** | 0,4 | **2,4** | **3,4** | **1,2** | 44 |
| `3.5-flash-lite` | 5,4 | 0,6 | 1,7 | 2,6 | 0,7 | 20 |
| `3.1-flash-lite` | 3,2 | **0,7** | 1,4 | 1,7 | *0,1* | 0 |

`3.6-flash` có 44 ca trùng lặp chéo trường (cùng một tên xuất hiện ở cả `conditions` lẫn `comorbidities`
— ví dụ `2021_13`), thổi phồng độ phủ khoảng 0,6/bệnh án. Trừ đi rồi chênh lệch vẫn thật: `3.1-flash-lite`
chỉ trích **3,2** conditions/bệnh án so với **7,5**, và gần như bỏ hẳn `comorbidities` (0,1 so với 1,2).

### Phủ định: khoảng cách quyết định

Chỉ số `neg` thô bị nhiễu bởi dương tính giả của regex (§6.3). Loại 4 topic đã xác định là dương tính
giả, còn **26 topic có phủ định thật**:

| model | tổng nhãn `negated` | bắt được / 26 phủ định thật |
|---|---|---|
| `gemini-3.6-flash` | **143** | **24 (92 %)** |
| `gemini-3.5-flash-lite` | 91 | 15 (58 %) |
| `gemini-3.1-flash-lite` | 60 | 13 (50 %) |

**Đây là lý do loại cả hai model Lite.** Phase 8 cần đúng các term `negated` để kết luận `satisfied` cho
tiêu chí loại trừ — bỏ sót gần một nửa số phủ định làm hỏng chính đóng góp trung tâm của đề tài, không
phải một chỉ số phụ.

Ví dụ rõ nhất, `2021_5` (bệnh án liệt kê thẳng *"notable for absence of chest pain, dyspnea on exertion,
paroxysmal nocturnal dyspnea, orthopnea, palpitations, syncope or presyncope"*):

| model | số `negated` bắt được |
|---|---|
| `3.6-flash` | 10 — đủ cả danh sách |
| `3.5-flash-lite` | 10 (xuất 12 nhưng **trùng lặp** `chest pain`, `syncope`) |
| `3.1-flash-lite` | **1** — sót gần trọn danh sách |

### Nhưng model nhỏ KHÔNG tệ hơn ở mọi mặt

Kiểm tay `2021_14` — bẫy bịa đặt đã phân tích ở §6.4 — cho kết quả **ngược với dự đoán**:

| model | `sex` | `evidence` | Đánh giá |
|---|---|---|---|
| `gemini-3.6-flash` | female | `"Daughter"` | ❌ **căn cứ sai** — có con gái không chứng minh bệnh nhân là nữ |
| `gemini-3.5-flash-lite` | female | `"She"` | ✅ **căn cứ đúng** |
| `gemini-3.1-flash-lite` | female | `"She"` | ✅ **căn cứ đúng** |

**Cả hai model Lite trích dẫn đúng ở chỗ model lớn trích dẫn sai.** Chất lượng không đơn điệu theo kích
thước model, và điều này đồng thời xác nhận phân tích ở §6.4 là đúng: bệnh án **có** nói giới tính qua đại
từ `"She"`, nhãn vàng regex chỉ không thấy vì đọc có 220 ký tự đầu.

### Quyết định: chọn `gemini-3.5-flash-lite`

`3.1-flash-lite` bị **loại** — trượt ngưỡng tuổi/giới (93,3 % / 93,2 % < 95 %) theo đúng tiêu chí thoát,
và sót gần trọn danh sách phủ định ở `2021_5`.

Giữa `3.6-flash` và `3.5-flash-lite`, **đã chọn `3.5-flash-lite`** (`gemini.MODEL`). Đây là một **đánh
đổi có ý thức**, không phải kết luận rằng hai model tương đương:

| | `3.6-flash` | `3.5-flash-lite` |
|---|---|---|
| Recall phủ định (26 ca thật) | **92 %** | 58 % |
| Độ phủ (giá trị/bệnh án) | **16,9** | 13,0 |
| Tốc độ | 3,87 s | **1,36 s** (2,8×) |
| Hạn ngạch | đã cạn trong ngày | **riêng, còn dư** |
| Dự phóng Phase 8 | 29,1 h | **10,2 h** |

Lý do chấp nhận: hạn ngạch và tốc độ là **ràng buộc khả thi** của Phase 8 (27.045 lần gọi — xem §11),
còn khoảng cách chất lượng hoá ra **không ảnh hưởng tới truy hồi** (§7.4).

**Rủi ro được ghi nhận, không được quên:** recall phủ định 58 % sẽ làm yếu bước kết luận `satisfied` cho
tiêu chí loại trừ ở Phase 8 — đúng chỗ đóng góp trung tâm của đề tài nằm. Phải **đo lại trực tiếp ở
Phase 8**, không được suy ra từ kết quả Phase 4. Nếu ảnh hưởng rõ rệt, `3.6-flash` vẫn còn nguyên hồ sơ
đã trích (`data/profiles/2021.gemini-3.6-flash.json`) để quay lại ngay, không cần chạy lại.

## 7. Ba biến thể truy vấn (bước 4)

Chạy trên **đúng cấu hình thắng của Phase 3**: `indexes/bm25-critfields`, `k1=1.8`, `b=1.0`. Đổi index
hay tham số là đổi hai thứ cùng lúc và chênh lệch sẽ không quy được cho ai.

**Phủ định bị loại khỏi truy vấn nhưng vẫn giữ trong hồ sơ.** Ném `"diabetes"` (từ *"no history of
diabetes"*) vào BM25 sẽ kéo về đúng những thử nghiệm về căn bệnh bệnh nhân **không** có — bản sao ngược
của bẫy phủ định đã gặp ở Phase 3. Nhưng Phase 8 vẫn cần các term đó để kết luận `satisfied` cho tiêu chí
loại trừ.

| Biến thể | Nội dung truy vấn | Độ dài TB |
|---|---|---|
| baseline (Phase 3) | bệnh án gốc, không xử lý | 135 từ |
| `prof` | chỉ các term đã trích | **19 từ** |
| `prof_narr` | bệnh án gốc + term nối thêm | 154 từ |
| `hyde` | mô tả thử nghiệm giả định do LLM sinh từ hồ sơ | 70 từ |

### Kết quả

| | official nDCG@10 | elig nDCG@10 | contam@10 ↓ | **judged@10** | **COND nDCG@10** | bpref | recall@1000 |
|---|---|---|---|---|---|---|---|
| `bm25_best` (baseline) | **0,3859** | 0,2399 | 0,2840 | 0,8467 | 0,2549 | 0,1611 | 0,4176 |
| `prof` | 0,3785 | 0,2324 | 0,2853 | 0,6680 | 0,3210 | 0,2293 | **0,5352** |
| `prof_narr` | **0,4528** | **0,2707** | 0,3467 | **0,8747** | 0,2917 | 0,2027 | **0,5379** |
| `hyde` | 0,3148 | 0,2188 | *0,1920* | **0,4813** | **0,3876** | **0,2609** | 0,5142 |

Kiểm định ý nghĩa thống kê (so với baseline):

| Độ đo | `prof` | `prof_narr` | `hyde` |
|---|---|---|---|
| official nDCG@10 | −0,0074 (p=0,82) *ns* | **+0,0669 (p=0,0001)** | −0,0711 (p=0,056) *ns* |
| elig nDCG@10 | −0,0075 (p=0,79) *ns* | **+0,0308 (p=0,020)** | −0,0210 (p=0,50) *ns* |
| elig P@10 | −0,0040 (p=0,87) *ns* | **+0,0320 (p=0,030)** | −0,0267 (p=0,38) *ns* |
| contamination@10 | +0,0013 (p=0,95) *ns* | **+0,0627 (p=0,0008)** ⚠ | *−0,0920 (p=0,0014)* † |
| recall@1000 | **+0,1177 (p=0,0002)** | **+0,1203 (p<0,0001)** | **+0,0966 (p=0,0016)** |

† Con số này **không đọc được như một chiến thắng** — xem §7.3.

### Đọc kết quả

**`prof` — nén 7 lần mà không mất chất lượng.** 19 từ so với 135 từ, mọi độ đo top-10 **không khác biệt
có ý nghĩa** so với bệnh án gốc, nhưng **recall@1000 tăng có ý nghĩa** (+0,118). Trích xuất giữ được
gần như toàn bộ tín hiệu truy hồi trong 1/7 độ dài. Đây là kết quả có giá trị cho Phase 5–7: truy vấn
ngắn rẻ hơn nhiều khi phải encode bằng mô hình dense hoặc đưa qua reranker.

Đánh đổi: `judged@10` giảm mạnh (0,847 → 0,668) — `prof` kéo về nhiều tài liệu **nằm ngoài pool đã
chấm**. Contamination@10 của nó vì thế bị *đánh giá thấp một cách giả tạo* (tài liệu chưa chấm mặc định
được coi là không gây ô nhiễm). Phải đọc hai cột này cùng nhau.

**`prof_narr` — truy vấn lexical tốt nhất từ trước tới nay, và cũng là bằng chứng rõ nhất cho luận điểm
của đề tài.** Thắng baseline ở **mọi** độ đo chất lượng (official +0,067, eligibility +0,031, recall
+0,120), tất cả đều có ý nghĩa thống kê. **Nhưng contamination@10 cũng tăng có ý nghĩa: 0,2840 → 0,3467**
(+22 % tương đối, p=0,0008), trong khi `judged@10` *tăng* (0,847 → 0,875) nên đây **không** phải hiệu ứng
pool — số ô nhiễm là thật.

Đây là lần **thứ hai** hiện tượng này được đo trong dự án. Lần đầu ở Phase 3 (gộp tiêu chí vào index:
contamination 0,140 → 0,284 khi relevance cải thiện). Giờ lặp lại ở phía **truy vấn**:

> Cải thiện độ khớp y khoa thì đồng thời kéo thêm các thử nghiệm **liên quan nhưng loại trừ** lên top.
> Giỏi hơn ở *relevance* làm cho vấn đề *eligibility* **tệ đi**.

Hiện tượng lặp lại nhất quán ở hai tầng độc lập (index và query) — không phải hiện tượng đơn lẻ. **Đây
chính là lý do Phase 8 tồn tại.**

### 7.3. `hyde` — kết quả đảo ngược hoàn toàn tùy cách chấm

`hyde` là biến thể **khó đọc nhất**, và đọc ẩu sẽ dẫn tới hai kết luận trái ngược, cả hai đều sai:

- Nhìn độ đo thô: **tệ nhất** (official nDCG@10 0,3148 — thấp hơn cả baseline).
- Nhìn contamination: **tốt nhất** (0,1920, giảm có ý nghĩa p=0,0014).
- Nhìn độ đo chống pool bias: **tốt nhất áp đảo** (COND nDCG@10 **0,3876** so với 0,2549 của baseline;
  bpref **0,2609** so với 0,1611 — cao nhất trong cả 4 lượt, cách biệt lớn).

Chìa khoá là **`judged@10` = 0,4813**: hơn **một nửa top-10 của `hyde` chưa từng được chấm**. Hệ quả:

**(a) Contamination thấp của `hyde` là giả tạo.** `contamination_at_k()` lấy mẫu số là `k`, tài liệu
chưa chấm mặc định coi như không gây ô nhiễm. Chuẩn hoá lại theo phần đã chấm:

| run | contam thô | judged@10 | **contam / judged** |
|---|---|---|---|
| `bm25_best` | 0,2840 | 0,8467 | **0,3354** |
| `prof` | 0,2853 | 0,6680 | 0,4271 |
| `prof_narr` | 0,3467 | 0,8747 | 0,3963 |
| `hyde` | *0,1920* | 0,4813 | **0,3989** |

Sau chuẩn hoá, `hyde` (0,3989) **không hề tốt hơn** `prof_narr` (0,3963) và **tệ hơn baseline** (0,3354).
Lợi thế contamination biến mất hoàn toàn.

**Và đây là phát hiện mạnh nhất của cả §7:** cả **ba** biến thể dựa trên trích xuất đều có tỷ lệ ô nhiễm
chuẩn hoá **cao hơn** bệnh án gốc (0,427 / 0,396 / 0,399 so với 0,335). Không phải một biến thể cá biệt
— **mọi cách dùng hồ sơ trích xuất để truy vấn đều làm vấn đề eligibility tệ đi**, dù cải thiện hay không
cải thiện relevance.

**(b) Không thể kết luận `hyde` tốt hay xấu từ dữ liệu này.** Trong số tài liệu **đã được chấm**, `hyde`
xếp hạng **tốt nhất trong cả 4 lượt** (COND@10, condP@10, bpref đều cao nhất). Nhưng 52 % top-10 nằm
ngoài pool, và dữ liệu hiện có **không phân biệt được** hai khả năng:

- `hyde` thật sự tìm ra thử nghiệm phù hợp mà pool 2021 bỏ sót (pool bias phạt oan một hệ thống tốt);
- `hyde` trôi khỏi chủ đề và kéo về tài liệu không liên quan, tình cờ chưa ai chấm.

Đây đúng là tình huống mà docstring của `metrics.py` đã dặn trước: *"nếu judged@10 thấp thì điểm chính
thức đang bị đánh giá thấp một cách hệ thống, và báo cáo cuối phải nói rõ điều đó."* Ghi nhận là **chưa
kết luận được**, không làm tròn thành "HyDE thắng" hay "HyDE thua".

**Hệ quả thực tế:** `hyde` **không** được chọn cho pipeline chính, nhưng lý do là *chưa chứng minh được*,
không phải *đã bác bỏ*. Nếu Phase 5–7 (dense + rerank) kéo `judged@10` lên, câu hỏi này đáng mở lại.

### 7.4. Model nhỏ hơn KHÔNG làm truy hồi kém đi

Sau khi chọn `gemini-3.5-flash-lite`, cả ba biến thể được chạy lại trên hồ sơ của model đó (cùng index,
cùng `k1`/`b`). Nếu khoảng cách trích xuất ở §6.5 (độ phủ 13,0 vs 16,9; phủ định 58 % vs 92 %) có ảnh
hưởng tới truy hồi thì phải thấy ở đây:

| | off@10 | elig@10 | contam | judged | COND@10 | rec@1k | **contam/judged** |
|---|---|---|---|---|---|---|---|
| baseline (bệnh án gốc) | 0,3859 | 0,2399 | 0,2840 | 0,8467 | 0,2549 | 0,4176 | **0,3354** |
| `prof` (3.6-flash) | 0,3785 | 0,2324 | 0,2853 | 0,6680 | 0,3210 | 0,5352 | 0,4271 |
| **`prof` (lite)** | 0,3873 | **0,2470** | 0,2653 | 0,6413 | **0,3511** | 0,5249 | 0,4137 |
| `prof_narr` (3.6-flash) | **0,4528** | 0,2707 | 0,3467 | 0,8747 | 0,2917 | 0,5379 | 0,3963 |
| **`prof_narr` (lite)** | 0,4438 | **0,2782** | 0,3240 | 0,8840 | 0,2946 | 0,5307 | **0,3665** |
| `hyde` (3.6-flash) | 0,3148 | 0,2188 | 0,1920 | 0,4813 | 0,3876 | 0,5142 | 0,3989 |
| **`hyde` (lite)** | 0,3046 | 0,2145 | 0,1813 | 0,4467 | **0,3950** | 0,5141 | 0,4060 |

**Không có khác biệt đáng kể ở bất kỳ cặp nào** — chênh lệch nằm trong khoảng dao động, và ở độ đo
eligibility model lite còn **nhỉnh hơn** cả ba lần (`prof` 0,2470 vs 0,2324; `prof_narr` 0,2782 vs
0,2707). `prof_narr` bản lite thậm chí có tỷ lệ ô nhiễm chuẩn hoá **thấp nhất** trong các biến thể trích
xuất (0,3665 so với 0,3963).

**Diễn giải:** BM25 hưởng lợi từ **những term hiển nhiên nhất** — tên bệnh, tên thuốc, biomarker — và cả
hai model đều bắt được nhóm đó. Phần `3.6-flash` trích thêm (độ phủ 16,9 vs 13,0, phần lớn là mục trùng
lặp chéo trường và chi tiết vụn) **không đóng góp gì cho truy hồi lexical**.

Điều này **thu hẹp** rủi ro của việc chọn model lite chứ không xoá bỏ nó: khoảng cách chất lượng là thật
nhưng nằm ở **phủ định**, mà phủ định **cố ý bị loại khỏi truy vấn** (§7) — nên Phase 4 bước 4 không thể
nhìn thấy nó. Chỗ nó sẽ hiện ra là **Phase 8**, nơi `negated` là đầu vào bắt buộc.

Một tín hiệu nhỏ đáng ghi: `prof` bản lite có **2 topic phải lùi về bệnh án gốc** vì không trích được term
nào dùng được (bản `3.6-flash` không có ca nào).

## 8. Bảng kiểm tiêu chí thoát

| # | Tiêu chí | Trạng thái |
|---|---|---|
| 1 | Bảng benchmark 6 model | **Bị thay thế, không phải bỏ qua.** Thiết kế cho quyết định "chạy cục bộ" đã bị đảo ngược; giờ chỉ còn một backend, `ollama.py` đã xoá. Phương pháp luận được giữ lại trong specs vì sẽ sống lại nếu Phase 8/9 quay về inference cục bộ. |
| 2 | schema ≥ 95 %, grounding ≥ 90 %, tuổi/giới ≥ 95 % | ✅ **100 % / 99,92 % / 100 % / 100 %** |
| 3 | Kiểm tay 25 bệnh án: 0 giá trị bịa lọt qua, phủ định gán đúng mọi ca | ⚠ **Gần đạt.** 0 ca bịa đặt sự thật. Nhưng `2021_14` có **căn cứ sai** (evidence `"Daughter"` cho kết luận giới tính nữ) và `2021_4` **bỏ sót một phủ định thật**. Xem §6.2–6.4. |
| 4 | Ghi cả 3 lượt truy vấn vào `results/` | ✅ **3/3.** `bm25_prof`, `bm25_prof_narr`, `bm25_hyde` đều có trong `results/` và `runs/`, kể cả lượt thua baseline — đúng như đặc tả yêu cầu. |

## 9. Hạn ngạch API và một quyết định về phương pháp

`gen_hyde_batch()` gom 5 topic/lần gọi, cache ra `data/profiles/hyde.{year}.{model}.json` khoá theo
`prompt_hash`, dùng lại đúng cơ chế khớp `index` an toàn của `extract.py`. Đã kiểm chứng bằng mock cho
cả 3 chế độ hỏng (index thiếu / trùng / ngoài khoảng) lẫn tái dùng cache.

**Quota chặn giữa chừng.** Hạn ngạch là **20 request/ngày/project/*mỗi model***
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). Sau khi extraction dùng 16 request,
`gemini-3.6-flash` chỉ sinh được **60/75** mô tả rồi cạn hẳn.

**Hai điều học được về quota, đều kiểm chứng bằng thực nghiệm:**

1. **Xoay key trong cùng project không reset quota** — tạo 3 key mới trong project cũ, request kế tiếp
   vẫn 429 ngay. Quota tính theo *project*, không theo *key*. Xoay key vì **bảo mật** và xoay key vì
   **quota** là hai việc khác nhau, chỉ việc thứ nhất có tác dụng.
2. **Quota tính theo từng MODEL.** Đây là đường ra: `gemini-3.5-flash` còn nguyên 20 request nên sinh
   được cả 75 mô tả trong một lượt (15 lô, không lô nào lỗi).

**Nhưng không được xoay vòng model giữa các topic.** Cùng logic round-robin đang dùng cho API key,
áp cho model sẽ nhân quota lên nhiều lần — và **sai** ở đây: nhánh `hyde` khi đó đo *"HyDE với một mớ
model hỗn hợp"*, không tái lập được và không quy chênh lệch cho ai. Cả 75 mô tả phải đến từ **một**
model. Vì vậy `query.py` tách `--hyde-model` khỏi `--model`: đổi model sinh HyDE mà vẫn nạp đúng hồ sơ
đã trích, và cache khoá theo model nên bản 60 topic của `3.6-flash` vẫn nguyên vẹn.

*(Xoay vòng model vẫn là ý tốt cho **Phase 8** — 27.045 lần gọi là sản xuất hàng loạt, không phải phép
đo so sánh giữa các nhánh.)*

**Ghi rõ để không bị hiểu nhầm:** hồ sơ bệnh nhân trích bằng `gemini-3.6-flash`, mô tả HyDE sinh bằng
`gemini-3.5-flash`. Việc so sánh `prof` / `prof_narr` / `hyde` vẫn hợp lệ (cả ba chạy trên cùng index,
cùng tham số, cùng bộ hồ sơ), nhưng nhánh `hyde` mang thêm một biến: model sinh mô tả. Nếu muốn loại bỏ
biến này, chạy lại bằng:

```bash
PYTHONPATH=. .venv/bin/python -m src.extraction.query --mode hyde --year 2021   # dung 3.6-flash
```

Cache đã giữ 60/75, chỉ cần 3 lô nữa khi quota hồi phục.

**Một lỗi đã được chặn trong lúc làm việc này.** Khi thiếu mô tả HyDE, `build_query()` lùi về **bệnh án
gốc**. Nếu để nguyên, lệnh sẽ ghi ra file run trộn 60 truy vấn HyDE với 15 bệnh án gốc — trông như kết
quả hợp lệ nhưng **không đo được HyDE mà cũng không đo được baseline**. `query.py` giờ **từ chối ghi
run** và thoát mã 1 khi còn topic thiếu mô tả.

## 10. Ba lỗi tiềm ẩn được phát hiện khi chạy thật

Cả ba đều nằm trong code đã viết từ trước nhưng **chưa từng được chạy**, và chỉ lộ ra khi có dữ liệu thật:

1. **`verify.py` đọc nhầm cấp trong JSON** — `json.load(...)` trả về `{prompt_hash, model, records}`
   nhưng code lặp trên cấp ngoài cùng thay vì `["records"]`, nên `n = 3` thay vì `75`.
2. **`verify.py` cắt sai tên model** — `f.split(".", 2)[1]` trên `2021.gemini-3.6-flash.json` cho
   `"gemini-3"` (mất `.6-flash`) vì tên model chứa dấu chấm. Đã sửa thành cắt theo tiền tố/hậu tố cố định.
3. **`query.py` âm thầm trộn run HyDE** — mô tả ở §9.

Ghi lại vì đây là bằng chứng cụ thể cho một điều dễ quên: **code chưa chạy là code chưa đúng**, kể cả khi
đã đọc kỹ. Đặc tả ghi "BUILT AND PASSING" cho những file này từ trước, nhưng "built" và "passing" là hai
chuyện khác nhau.

---

## Tóm tắt

**Trích xuất đạt và vượt mọi ngưỡng cơ học** — schema 100 %, grounding 99,9 %, tuổi/giới 100 % trên cả
75 bệnh án dev.

**Về truy hồi, ba biến thể cho ba câu trả lời khác nhau:**

- `prof_narr` là **truy vấn lexical tốt nhất từ trước tới nay** (official nDCG@10 0,4528, +0,067 so với
  baseline, p=0,0001).
- `prof` nén truy vấn **xuống 1/7 độ dài mà không mất chất lượng top-10**, lại tăng recall — hữu ích cho
  Phase 5–7 nơi độ dài truy vấn tốn kém thật.
- `hyde` **chưa kết luận được**: xếp hạng tốt nhất trong số tài liệu đã chấm, nhưng hơn nửa top-10 nằm
  ngoài pool nên dữ liệu không phân biệt được "tìm ra thứ pool bỏ sót" với "trôi khỏi chủ đề".

**Và phát hiện quan trọng nhất:** sau khi chuẩn hoá theo độ phủ pool, **cả ba** biến thể dựa trên trích
xuất đều có tỷ lệ ô nhiễm **cao hơn** bệnh án gốc (0,427 / 0,396 / 0,399 so với 0,335). Cộng với kết quả
Phase 3 (contamination 0,140 → 0,284 khi index tốt lên), đây là **bằng chứng thực nghiệm ở hai tầng độc
lập** cho luận điểm trung tâm của đề tài:

> Giỏi hơn ở *relevance* làm cho vấn đề *eligibility* **tệ đi**.

Đó chính xác là thứ Phase 8 sinh ra để đảo ngược.
