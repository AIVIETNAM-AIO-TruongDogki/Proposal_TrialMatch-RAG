[← specs index](README.md) · Phase 8 of 11 · prev: [7 — Reranking](07-reranking.md) · next: [9 — Evidence-grounded generation](09-evidence-grounded-generation.md)

# Phase 8 — Eligibility reasoning (ladder rung 5 — the contribution)

**Goal.** For each of the top-N reranked trials, label **every criterion** `satisfied` / `violated` /
`unverifiable`, with a grounded evidence span.

This is the phase the project exists for. Budget accordingly — it is the reason weeks 6 and 8 are
protected.

**Steps.**

1. **Get criterion-level labels.** TREC qrels are trial-level only; they cannot supervise or evaluate a
   criterion-level classifier. Three options, in increasing order of cost:
   - **(a)** Evaluate only at trial level: aggregate predicted criterion labels into a trial-level
     decision and compare against qrels `2/1/0`. Free, but it measures the aggregation rule as much as
     the reasoning.
   - **(b)** Check what the TrialGPT release [E1] provides — it published criterion-level annotations
     over patient cohorts including TREC-derived ones. If the licence and coverage fit, this removes the
     single largest cost in the project. **Verify this in [Phase 0](00-ground-truth-environment.md), not
     Phase 8.**
   - **(c)** Annotate your own gold set: ~20 topics × 5 trials × ~10 criteria ≈ 1,000 criteria. Expensive,
     but it is the asset that makes the three-state claim defensible.

   Recommended: (a) as the guaranteed path, (b) checked early, (c) as the stretch deliverable.

2. **Per-criterion prompt, structured output.** One criterion at a time, or small batches — not the whole
   criteria blob at once. Enforce a schema:
   ```json
   {
     "criterion_idx": 3,
     "label": "violated",
     "patient_evidence": "received cisplatin in 2021",
     "criterion_quote": "No prior platinum-based chemotherapy",
     "reasoning": "..."
   }
   ```
3. **Enforce grounding mechanically.** Reject any output where `criterion_quote` is not a literal
   substring of the stored criterion text, or `patient_evidence` is not a literal substring of the
   narrative. This converts invariant 3 from a hope into a check. Log the rejection rate — it is a
   faithfulness measurement in its own right and a reportable number.
4. **Make `unverifiable` the default.** Prompt for it explicitly: if the narrative does not state the
   information, the label is `unverifiable` — inference from typical patients is forbidden. Then test the
   invariant adversarially: take narratives with a field removed and confirm the label flips to
   `unverifiable` rather than staying `satisfied`.
5. **Aggregate to a trial score.** Define the rule explicitly and ablate it — for example:
   any `violated` exclusion ⇒ trial excluded; all inclusions `satisfied` ⇒ eligible; otherwise ranked by
   the count of `unverifiable`. The aggregation rule is a free parameter that materially affects the
   headline result, so it must be stated and varied, not buried in code.
6. **Evaluate with paired metrics.** Macro-F1 over the three labels is the proposal's metric, but on its
   own it is gameable: a model that answers `unverifiable` everywhere can score respectably while being
   useless. Always report jointly:
   - macro-F1 over `{satisfied, violated, unverifiable}`;
   - **abstention rate** (share labelled `unverifiable`) and **accuracy on the non-abstained subset**;
   - a **risk–coverage curve** if you have confidence scores.

   And run a **forced-choice ablation** — the same model with `unverifiable` removed from the label set.
   If two-state does as well, the three-state contribution is not yet demonstrated. That ablation is the
   direct empirical test of invariant 1, and reviewers will ask for it.
7. **Cost control.** N trials × M criteria × 125 topics is a large number of LLM calls. Cache aggressively
   keyed on `(nct_id, criterion_idx, topic_id)`, cap N at 20–50 for the full runs, and measure cost per
   topic — it is a reportable systems result.

**Decide.** Reasoning LLM (on reasoning quality, structured-output support, context length, cost,
reproducibility — the proposal's own criteria); N; aggregation rule.

**Deliverable.** `src/reasoning/`, criterion-level predictions, the three-state evaluation table, the
forced-choice ablation, the grounding-violation rate.

**Exit criterion.** Rung 5 is measured against rung 4 on **both** metric families from
[Phase 2](02-evaluation-harness.md) — and the central claim is stated in terms of **exclusion
contamination@10**, where the eligibility stage should show a clear reduction even if official nDCG@10
moves little or drops.

## Status: COMPLETE — rung 5 measured, forced-choice ablation measured, both significance-tested.

Chạy đầy đủ 03/09/2026 trên `runs/hybrid.dev.txt --top-n 20`, dev 2021, `gemini-3.5-flash-lite`:
**1.499 cặp · 22.828 tiêu chí · 1.633 lời gọi · 2,23 M token vào / 2,03 M ra**, đủ 75/75 topic.
Kết quả: [`data/reasoning/2021.gemini-3.5-flash-lite.trial.json`](../data/reasoning/), run xếp lại:
`runs/elig_strict.dev.txt`.

### Tiêu chí thoát — đạt

Điều kiện thoát yêu cầu đo bậc 5 trên **cả hai** họ metric, và phát biểu luận điểm trung tâm bằng
**contamination@10**:

| | bậc 3 (hybrid) | **bậc 5 (+eligibility)** | |
|---|---|---|---|
| **contamination@10** | 0.3573 | **0.2813** | **−21,3% tương đối** ✅ |
| elig nDCG@10 | 0.3501 | **0.4182** | +0.0681 |
| official nDCG@10 | 0.5309 | **0.5472** | +0.0163 |

Điều kiện thoát đã dự phòng rằng official nDCG@10 **có thể giảm mà vẫn đúng**, vì qrels chính thức
cho điểm dương cho trial `excluded`. Nó **không giảm** — nên bộ lọc eligibility ở đây không đánh đổi
độ liên quan lấy độ sạch.

**Paired bootstrap đã chạy** (`src.eval.score runs/elig_strict.dev.txt --vs results/hybrid.dev.json`,
10.000 lần lấy mẫu lại, 75 topic chung):

| đo | bậc 3 | bậc 5 | chênh | p | |
|---|---|---|---|---|---|
| **contamination@10** | 0.3573 | **0.2813** | **−0.0760** | **0.0000** | **có ý nghĩa** |
| eligible nDCG@10 | 0.3501 | 0.4182 | +0.0681 | 0.0007 | có ý nghĩa |
| eligible P@10 | 0.3333 | 0.3920 | +0.0587 | 0.0000 | có ý nghĩa |
| official nDCG@10 | 0.5309 | 0.5472 | +0.0163 | 0.3120 | **không phân biệt được** |
| recall@1000 | 0.6490 | 0.6490 | +0.0000 | 1.0000 | không phân biệt được (đúng như thiết kế) |

Con số tiêu đề — độ giảm contamination@10 — **có ý nghĩa thống kê**, đây là kết quả mạnh nhất của
đề tài. Nhưng phải nói đúng trong bài: **official nDCG@10 tăng KHÔNG có ý nghĩa** (p=0.31) — không
thể khẳng định eligibility reasoning cải thiện độ liên quan chính thức, chỉ khẳng định được nó giảm
ô nhiễm loại trừ. recall@1000 giữ nguyên tuyệt đối (p=1.0) vì bậc 5 chỉ **xếp lại thứ tự** trong
top-N đã truy hồi (`aggregate.rerank_by_eligibility`), không đổi tập tài liệu.

### Ba luật gộp, đọc cạnh nhau

```
luat            n       P       R      F1     acc
strict      1,183  0.5621  0.6967  0.6222  0.6982   <- tot nhat
lenient     1,183  0.4574  0.8270  0.5890  0.5883
count       1,183  0.3570  1.0000  0.5262  0.3576   <- doi chung
```

`count` cho **R = 1.0000, P = 0.3570** — nó không loại ai nên đoán mọi trial đều eligible, và
precision của nó *chính là tỷ lệ nền* (35,7% trial đã chấm là eligible). Đó đúng là hành vi phải có
của một đường cơ sở, và `strict` vượt nó **F1 0.6222 so với 0.5262**: bước loại bỏ đóng góp thật,
không phải luật gộp tình cờ hợp qrels.

Bảng này chỉ đọc được **nhờ bản vá `count` cùng ngày** — trước đó cả ba luật cùng chạm sàn `1e-6`,
mọi trial hoà nhau, và ba dòng trên sẽ vô nghĩa. Xem mục 2 bên dưới.

**Tỷ lệ kiêng = 59,7%** (`unverifiable` 12.851 · `satisfied` 7.052 · `violated` 1.632). Phải đọc
cùng lúc với F1: một model trả `unverifiable` cho mọi tiêu chí vẫn có F1 coi được mà vô dụng.

### Vi phạm grounding: 5,7% (1.293/22.828)

Cao hơn 0,9% đo trên smoke test n=3. Lý do áp đảo là `patient_evidence_empty` — model gán
`satisfied`/`violated` nhưng bỏ trống trích dẫn. **Giả thuyết chưa kiểm chứng:** đây là tác dụng phụ
của chính câu thêm vào prompt ("muốn trích đoạn dài để chứng minh sự vắng mặt thì đó là tín hiệu của
`unverifiable` với evidence rỗng"). Không kiểm được vì ta không có tỷ lệ của prompt cũ ở quy mô này —
smoke test n=3 quá nhỏ để làm mốc. Ghi lại như giả thuyết, không như kết luận.

### Ablation lựa chọn ép buộc — invariant 1 được chứng minh, không chỉ sử dụng

`--forced` chạy đủ 1.499/1.499 cặp (16.566 quyết định giữ lại, thấp hơn 21.535 của bản ba trạng
thái vì tỷ lệ vứt bỏ grounding cao hơn hẳn — 21,0% ở lượt cuối so với 5,2%, đa phần là
`patient_evidence_empty`: bị ép chọn nhãn, model hay để trống trích dẫn hơn). Chấm bằng
`src.reasoning.score` (luật `strict`, so cùng 1.183 trial đã chấm qrels):

| | P | R | **F1** | acc | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|
| **Ba trạng thái** | 0.5621 | **0.6967** | **0.6222** | 0.6982 | 294 | 229 | 128 | 532 |
| Hai trạng thái (ép buộc) | 0.6051 | 0.5664 | 0.5851 | **0.7134** | 239 | 156 | 183 | 605 |

**Ba trạng thái thắng trên F1 (+0.0371), thua trên accuracy (−0.0152) — hai chỉ số đi ngược
chiều nhau, và hiểu đúng đòi hỏi nhìn vào ma trận nhầm lẫn, không chỉ hai con số tổng.**

Cơ chế: bỏ `unverifiable` buộc model chọn `satisfied`/`violated` cho cả tiêu chí thật sự mơ hồ,
và nó nghiêng về `violated` nhiều hơn — dưới luật `strict` (một `violated` bất kỳ ⇒ loại), điều
này làm **nhiều trial bị loại hơn hẳn** (788 vs 660/1.183). Accuracy tăng giả tạo vì lớp đa số
vốn là "không đủ điều kiện" (chỉ 35,7% trial đã chấm là eligible — xem baseline `count` ở mục
trên), nên loại nhiều hơn "trông có vẻ đúng" nhiều hơn. Nhưng cái giá thật là **fn tăng từ
128→183**: hai trạng thái loại nhầm nhiều trial *thật sự đủ điều kiện* hơn hẳn — đúng thất bại mà
thiết kế ba trạng thái sinh ra để ngăn (invariant 1). Đây chính xác là lý do bước 6 ở trên dặn
không được tin accuracy một mình: nhìn accuracy sẽ kết luận sai là hai trạng thái tốt hơn.

**Kết luận: đóng góp ba trạng thái được chứng minh trên F1 — chỉ số bài báo đã chọn làm tiêu đề —
chứ không phải trên mọi chỉ số.** Ghi cả hai chiều vào báo cáo, không chỉ chiều có lợi.

### Ghi chú thi hành — bốn lỗi lộ ra trong lần chạy thật

Cả bốn cùng một họ: **mất công trong im lặng, không ném ngoại lệ nào.**

- **`save_cache` ghi không nguyên tử.** `json.dump` thẳng vào file thật, mất vài giây cho 7 MB; bị
  kill đúng khoảng đó (chính là thao tác đổi khoá API giữa chừng) thì file cụt. Nay `.tmp` + `fsync`
  + `os.replace`, giữ một bản `.bak`.
- **`load_cache` nuốt lỗi hỏng.** Bắt `JSONDecodeError` rồi trả `{}` — cache hỏng bị đối xử y như
  cache trống, nên lần chạy sau âm thầm gọi lại từ đầu, đốt một ngày hạn ngạch không một dòng cảnh
  báo. Nay dừng hẳn và chỉ ra bản `.bak`.
- **`load_decisions` không kiểm `prompt_hash`.** Nó nạp file ablation sót lại từ 30/08 (hash
  `ceb71ba6c6cb`, **đúng 1 bản ghi**, prompt cũ) rồi in ra `hai trang thai F1=0.0000 acc=1.0000` —
  cặp số vô nghĩa nhưng trông như kết quả. So hai file sinh bởi hai prompt khác nhau thì không đo
  được gì. Nay từ chối thẳng khi hash lệch, và cảnh báo khi file ablation thiếu cặp.
- **Vòng quay vô ích khi cạn hạn ngạch.** Xem `MAX_CONSECUTIVE_FAILS` ở
  [risk-register](risk-register.md).

**Đính chính ước tính token của chính mục này:** thực tế **2,23 M vào / 2,03 M ra**, gấp ~3 lần con
số 0,64 M / 0,96 M ghi ở bản trước. Sai vì ngoại suy *theo mỗi tiêu chí* từ 2 trial có 37–38 tiêu
chí, nên chi phí **cố định mỗi lời gọi** (system prompt ~700 token + bệnh án ~450) bị amortize quá
tốt; trung bình thật chỉ 15,2 tiêu chí/trial nên phần cố định chiếm ưu thế.

### Smoke test (hồ sơ — cơ sở để mở lần chạy thật)

3 cặp (topic, trial) trên `runs/hybrid.dev.txt`, topic `2021_1`, `gemini-3.5-flash-lite`.
Cả hai chiều đều được kiểm, chứ không chỉ chiều dễ.

| trial | qrel | kỳ vọng | `strict` | `lenient` | `count` |
|---|---|---|---|---|---|
| NCT00004259 | 1 = excluded | loại | **0.0000** ✅ | **0.0000** ✅ | 0.3514 |
| NCT00783393 | 1 = excluded | loại | **0.0000** ✅ | **0.0000** ✅ | 0.2432 |
| NCT00003470 | 2 = eligible | giữ | 0.1290 ✅ | 0.2742 ✅ | 0.1290 |

Ba lỗi bị bắt bởi lần smoke đầu và đã sửa **trước khi** tiêu lời gọi nào cho lần chạy thật —
đúng thời điểm, vì đổi prompt làm hỏng cache (`prompt_hash` 205f6932a4f4 → 91bc0fee7f55).

1. **Suy diễn định ngữ khuyết → `violated` giả.** Tiêu chí `"No uncontrolled hypertension"`, bệnh án
   chỉ nói `"hypertension"`, model kết luận *violated* vì "control status is not specified as
   controlled". Đó là vi phạm invariant 2 theo **chiều ngược** với `satisfied`-từ-vắng-mặt, và nguy
   hiểm hơn nhiều: `strict` loại trial chỉ với một vi phạm, nên nó **loại nhầm một trial qrel=2**.
   Sửa: `prompts/eligibility_system.txt` thêm mục về tiêu chí mang định ngữ
   (*uncontrolled / significant / adequate / severe / active*) với ba ví dụ đối chiếu.
   Sau khi sửa: `[14] UNVERIFIABLE — "mentions hypertension but does not state its control status"`.

2. **`count` — nhánh đối chứng — suy biến.** `base − penalty = sat/n − 0.5·unv/n` âm bất cứ khi nào
   `unv > 2·sat`, và với `unverifiable` chiếm ~68% nhãn (đúng như thiết kế ba trạng thái mong đợi)
   đó là trường hợp *thường*, không phải ngoại lệ: cả ba trial đều chạm sàn `1e-6` và hoà nhau, làm
   xếp hạng rơi hết về tie-break truy hồi. Sửa bằng `_spread()` — ánh xạ đơn điệu vào `[1e-6, 1]` —
   và `count` giờ đúng như docstring hứa: thuần `sat/n`, không phạt `unverifiable`.
   `DISQUALIFIED = 0.0` thành giá trị riêng, không trial nào khác chạm tới.

3. **`section=unknown` làm `lenient` tuỳ tiện.** Định dạng NCI cũ viết cả hai loại tiêu chí thành một
   danh sách phẳng, phân biệt bằng thể phủ định. NCT00004259 có ba tiêu chí loại trừ thật
   (`"No prior chemotherapy"`…) nhưng không có header nên `vio_exc=0` → `lenient` **không loại**, sai
   so với qrel. Cùng lỗi đó lại **cứu nhầm** NCT00003470 — hai hướng ngẫu nhiên.
   Sửa: `aggregate.effective_section()` suy ra cực phủ định, đặt ở tầng gộp **chứ không ghi đè DB** —
   `section` trong DB là sự thật về *cấu trúc*, hàm này là suy diễn *ngữ nghĩa*; trộn hai thứ vào nhau
   sẽ mất khả năng đo. Ablate được bằng `trial_score(..., infer_section=False)`; trên NCT00004259 cờ
   đó là hiệu số 0.4143 (giữ, sai) so với 0.0000 (loại, đúng).

**Vi phạm grounding 4.0% → 0.9%** (1/106). Toàn bộ 4% ban đầu là một kiểu duy nhất: model dựng
`patient_evidence` bằng cách cắt ruột giữa đoạn văn bằng `...` để "chứng minh" một điều *không* có
trong bệnh án. Prompt nay cấm dấu ba chấm và nói rõ: muốn trích cả đoạn dài để chứng minh sự vắng mặt
chính là tín hiệu của `unverifiable` với evidence rỗng.

**Hai điều đã đo và phải nói trong báo cáo:**

- **`temperature=0` KHÔNG tất định.** Phát lại đúng prompt cũ trên NCT00004259 cho tập bị vứt khác
  (`[26,36,37]` so với `[22,23,26]`) và số token khác (3.146 so với 3.182). Tính tái lập của Phase 8
  nằm ở **cache** `data/reasoning/*.json`, không nằm ở nhiệt độ.
- **Tỷ lệ vi phạm grounding là chặn dưới của vi phạm invariant 2, không phải bộ lọc đầy đủ.** Nó chỉ
  bắt được khi model tình cờ trích sai; ca `dacarbazine` lọt qua vì trích dẫn của nó là chuỗi con
  nguyên vẹn. Đừng báo cáo nó như thể đã bắt hết.

**Chưa đo được:** n = 3 trial. Cơ chế hỏng thì rõ và lặp lại được; **tỷ lệ** thì không.
Bảng ba trạng thái và ablation lựa chọn ép buộc vẫn còn nguyên ở phần trên.

**Hạn ngạch — số đã chạy thật, không còn là ước tính.** Trần **500/ngày/project/model**, xác nhận
bằng payload 429 sống: `quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier`,
`quotaValue: 500`. 3 khoá = 3 project riêng → 1.500/ngày. Lần chạy thật: **1.461 lời gọi trong một
lượt 2,4 giờ** (6 s/cặp) trước khi chạm trần, 172 lời gọi còn lại chạy tiếp sau khi thay khoá — tổng
1.633. Thời gian API thuần ~2,7 giờ; phần còn lại là chờ hạn ngạch.

**Reading.** TrialGPT [E1] — closest prior work, criterion-level three-way prediction with published
numbers (87.3% criterion-level accuracy) and released code; Wornow et al. [E2] for zero-shot prompting
on n2c2; Jullien et al. [E3] for controlled/set-guided LLM reasoning evaluated on *this exact collection*
(nDCG@10 0.693, P@10 0.73 on TREC 2022 — a concrete target to compare against); SatIR [E4] for the
constraint-satisfaction framing. Full citations in [reading-list.md](reading-list.md).

---
[← specs index](README.md) · prev: [7 — Reranking](07-reranking.md) · next: [9 — Evidence-grounded generation](09-evidence-grounded-generation.md)
