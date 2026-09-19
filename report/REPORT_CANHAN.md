# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đức Thắng
**Nhóm:** G25
**Ngày:** 

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

> **Cấu hình đo của báo cáo này.** Corpus `data/quy-che-dao-tao-neu/` (10 tài liệu quy chế đào tạo NEU) · nhúng bằng **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều, chạy cục bộ)** · sinh câu trả lời bằng `gemini-3.5-flash`. Mọi con số đều từ lần chạy thật.
>
> ⚠ **Cấu hình này khác cấu hình chính thức của nhóm**, vốn dùng `gemini-embedding-001` (3.072 chiều) và đạt **10/10**. Báo cáo cá nhân giữ nguyên số liệu đo trên mô hình 384 chiều vì phần phân tích câu hỏng ở mục 5 chỉ quan sát được ở cấu hình đó. Xem `REPORT_NHOM.md` mục 2, tiểu mục *"Đổi mô hình nhúng đáng giá hơn đổi chiến lược chunking"* để biết vì sao hai con số chênh nhau.

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector embedding chỉ về **cùng một hướng** trong không gian nhiều chiều, nghĩa là mô hình cho rằng hai đoạn văn bản nói về cùng một ý — bất kể chúng dùng từ ngữ khác nhau hay dài ngắn khác nhau. Cosine chỉ đo góc, không đo độ dài vector.

**Ví dụ có độ tương tự CAO:**
- Câu A: `Sinh viên đăng ký học phần`
- Câu B: `Thủ tục đăng ký môn học của sinh viên`
- Tại sao tương đồng: đo được **+0.875**. Hai câu không dùng chung cụm "học phần"/"môn học" nhưng mô hình vẫn nhận ra cùng một nghiệp vụ. Đây chính là điểm hơn của embedding so với so khớp từ khoá — tìm kiếm bằng từ khoá sẽ trượt cặp này.

**Ví dụ có độ tương tự THẤP:**
- Câu A: `Sinh viên nộp đơn phúc khảo`
- Câu B: `Sinh vien nop don phuc khao`
- Tại sao khác: đo được **+0.082**, gần như trực giao — dù đây là **cùng một câu**, chỉ khác ở dấu tiếng Việt. Mô hình đa ngữ được huấn luyện trên tiếng Việt có dấu, nên bản không dấu rơi ra ngoài phân phối và bị coi là một chuỗi ký tự xa lạ.

**Tại sao độ tương tự cosine được ưu tiên hơn khoảng cách Euclid cho text embeddings?**
> Vì độ dài vector phụ thuộc vào độ dài văn bản, còn hướng vector mới mang ngữ nghĩa. Một điều khoản 1.300 ký tự và một câu hỏi 40 ký tự nói cùng nội dung sẽ có khoảng cách Euclid rất lớn nhưng góc rất nhỏ. Ngoài ra các backend trong `src/embeddings.py` đều trả vector đã chuẩn hoá (‖v‖ = 1), nên tích vô hướng bằng đúng cosine và `search()` tính nhanh hơn.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10.000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính: `ceil((10000 − 50) / (500 − 50))` = `ceil(9950 / 450)` = `ceil(22,11)` = **23**
> Đáp án: **23 chunks** — đã kiểm lại bằng chính code trong repo, không tin công thức suông:
> ```
> FixedSizeChunker(chunk_size=500, overlap=50).chunk('a' * 10000)  ->  23 chunks
> ```

**Nếu overlap tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Bước nhảy giảm từ 450 xuống 400 nên số chunk tăng lên **25** (`ceil(9900/400) = 25`, cũng đã kiểm bằng code). Overlap lớn hơn để **một câu bị cắt ngang không bị mất nghĩa ở cả hai chunk**: phần chồng lấn bảo đảm ý nằm ở ranh giới vẫn xuất hiện trọn vẹn trong ít nhất một chunk. Cái giá là nhiều chunk hơn, tốn thêm chi phí nhúng và lưu trữ.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng regex lookbehind `(?<=[.!?])\s+` để cắt **tại khoảng trắng sau dấu câu** chứ không cắt tại chính dấu câu — nhờ vậy dấu `.`/`!`/`?` được giữ lại trong câu. Một biểu thức này đã phủ cả `". "`, `"! "`, `"? "` và `".\n"`. Sau khi tách thì `strip()` từng câu, bỏ câu rỗng, rồi gom theo `max_sentences_per_chunk`. Text rỗng hoặc toàn khoảng trắng trả `[]`.
>
> Edge case tôi biết là **chưa xử lý được**: chữ viết tắt bị cắt sai — `"Giảng viên là TS. Nguyễn Văn A."` ra hai chunk `'Giảng viên là TS.'` và `'Nguyễn Văn A.'`; `"v.v. Nộp tại phòng"` cũng vậy. Nguy hiểm nhất với corpus quy chế là **mục đánh số**: văn bản viết `1. Đối với điểm đánh giá...` nên với `max_sentences_per_chunk=1` thì `'2.'` thành một chunk riêng, tách rời khỏi khoản mà nó đánh số. Số thập phân (`2,25` / `2.00`) thì **không** vỡ vì không có khoảng trắng sau dấu chấm.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán có **hai chiều**, và chỗ dễ thiếu là chiều thứ hai:
> 1. *Đệ quy xuống*: thử separator theo thứ tự `["\n\n", "\n", ". ", " ", ""]`; mảnh nào vẫn dài hơn `chunk_size` thì gọi lại `_split` với danh sách separator còn lại.
> 2. *Gom lên* (`_merge`): nối các mảnh nhỏ liền kề lại cho tới sát `chunk_size`. Thiếu bước này, `"word " * 200` sinh ra 200 chunk 4 ký tự — vẫn qua test nhưng retrieval vô dụng. Có gom thì ra **10 chunk ~99 ký tự**.
>
> Ba base case: (a) text rỗng → `[]`; (b) mảnh đã ngắn hơn `chunk_size` → giữ nguyên; (c) **hết separator hoặc separator là `""`** → cắt cứng theo `chunk_size` bằng `_hard_slice`. Nhánh (c) là nhánh mà `test_empty_separators_falls_back_gracefully` nhắm vào.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `add_documents` không tự chunk: một `Document` thành đúng một record, nên việc chia nhỏ nằm ở tầng ngoài và đổi chiến lược không phải đụng vào store. Mỗi record được `_make_record` chuẩn hoá gồm `id` duy nhất, bản **sao** metadata, nội dung và vector.
>
> `search` uỷ quyền cho `_search_records` — dùng tích vô hướng thay vì gọi `compute_similarity`, vì vector từ mọi backend đều đã chuẩn hoá nên dot product **bằng đúng** cosine mà đỡ hai lần tính căn. Kết quả trả về **bỏ trường `embedding`** vì vector 384 chiều làm rác output khi in ra.
>
> Tôi đã **bỏ hẳn nhánh ChromaDB** mà skeleton phác: `requirements.txt` không cài nó, không test nào chạy qua nó, nên giữ lại là để một đường code chưa từng được kiểm chứng trong bài nộp.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> **Lọc trước, xếp hạng sau.** Nếu xếp hạng trước rồi mới loại, cả `top_k` slot có thể bị tài liệu sai chiếm hết và kết quả về rỗng dù store còn thừa tài liệu hợp lệ. Tôi đo thử với store 20 tài liệu `faculty` + 3 tài liệu `student`, `top_k=3`:
>
> | Cách làm | Số kết quả |
> |---|---|
> | Lọc trước rồi search | **3** |
> | Search rồi lọc sau | **0** |
>
> Cả `search` và `search_with_filter` đều đi qua cùng một `_search_records`, chỉ khác tập ứng viên đầu vào — nên hai hàm không thể lệch kết quả, và `test_no_filter_returns_all_candidates` đúng hiển nhiên.
>
> `delete_document` lọc theo `metadata['doc_id']`, so sánh độ dài trước/sau để trả `True`/`False`. Điểm quan trọng: `_make_record` dùng `metadata.setdefault("doc_id", doc.id)` chứ không gán đè — nhờ vậy khi nhiều chunk của một file được nạp dưới dạng `Document("file#0")`, `Document("file#1")`… thì `doc_id` vẫn trỏ về **file gốc** và một lệnh xoá gỡ sạch cả file.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Ba nhịp: truy xuất top-k → `_build_prompt` → gọi `llm_fn`. Phần đầu tư nằm ở cách dựng ngữ cảnh: mỗi chunk được **đánh số `[1] [2] [3]`** kèm `source_url` và score, rồi prompt yêu cầu model **trích dẫn số đó** trong câu trả lời — nhờ vậy mọi câu đều truy ngược được về đúng chunk và đúng file (tiêu chí *Source Traceability*).
>
> Thêm ràng buộc chống bịa: *"Answer using ONLY the context below. If the context does not contain the answer, say so instead of guessing."* Ràng buộc này **có tác dụng thật** — xem câu 1 ở mục 5: retrieval trượt, và agent trả lời "không có thông tin" thay vì bịa ra một con số.
>
> Store rỗng thì trả câu thông báo sẵn, không gọi LLM với ngữ cảnh rỗng.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\CODE\AITHUCCHIEN\LABS\K4-L3A-Data-Foundations
collected 42 items

TestProjectStructure::test_root_main_entrypoint_exists PASSED
TestProjectStructure::test_src_package_exists PASSED
TestClassBasedInterfaces::test_chunker_classes_exist PASSED
TestClassBasedInterfaces::test_mock_embedder_exists PASSED
TestFixedSizeChunker::test_chunks_respect_size PASSED
TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
TestFixedSizeChunker::test_returns_list PASSED
TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
TestSentenceChunker::test_chunks_are_strings PASSED
TestSentenceChunker::test_respects_max_sentences PASSED
TestSentenceChunker::test_returns_list PASSED
TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
TestRecursiveChunker::test_handles_double_newline_separator PASSED
TestRecursiveChunker::test_returns_list PASSED
TestEmbeddingStore::test_add_documents_increases_size PASSED
TestEmbeddingStore::test_add_more_increases_further PASSED
TestEmbeddingStore::test_initial_size_is_zero PASSED
TestEmbeddingStore::test_search_results_have_content_key PASSED
TestEmbeddingStore::test_search_results_have_score_key PASSED
TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
TestEmbeddingStore::test_search_returns_list PASSED
TestKnowledgeBaseAgent::test_answer_non_empty PASSED
TestKnowledgeBaseAgent::test_answer_returns_string PASSED
TestComputeSimilarity::test_identical_vectors_return_1 PASSED
TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
TestComputeSimilarity::test_zero_vector_returns_0 PASSED
TestCompareChunkingStrategies::test_counts_are_positive PASSED
TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
TestCompareChunkingStrategies::test_returns_three_strategies PASSED
TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.08s ==============================
```

**Số lượng bài test vượt qua (pass):** **42** / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Dự đoán được ghi **trước khi chạy**. Nhúng bằng `paraphrase-multilingual-MiniLM-L12-v2`.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên đăng ký học phần | Thủ tục đăng ký môn học của sinh viên | cao | **+0.875** | ✅ |
| 2 | Điều kiện xét công nhận tốt nghiệp | Yêu cầu để được cấp bằng tốt nghiệp | cao | **+0.863** | ✅ |
| 3 | Mức thu học phí mỗi học kỳ | Trách nhiệm coi thi của giảng viên | thấp | **+0.458** | ⚠️ cao hơn dự đoán |
| 4 | Sinh viên nộp đơn phúc khảo | Sinh vien nop don phuc khao | cao | **+0.082** | ❌ sai hẳn |
| 5 | Quy định chương trình Tiên tiến | Quy định chương trình Chất lượng cao | thấp | **+0.759** | ❌ sai hẳn |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> **Cặp 4 là cú sốc lớn nhất: cùng một câu, chỉ khác dấu tiếng Việt, mà chỉ đạt +0.082 — gần như trực giao.** Embedding không hiểu "ý nghĩa" một cách trừu tượng; nó hiểu những chuỗi token giống với dữ liệu nó từng thấy. Tiếng Việt không dấu nằm ngoài phân phối huấn luyện nên bị đẩy sang một vùng hoàn toàn khác. Hệ quả thực tế: **benchmark query bắt buộc phải gõ đủ dấu**, nếu không ta đang đo nhiễu mà tưởng đang đo chất lượng truy xuất.
>
> Cặp 5 bất ngờ theo hướng ngược lại: hai văn bản của **hai chương trình khác nhau** lại đạt +0.759, vì chúng được soạn song song gần như từng Điều. Đây chính là lý do corpus cần trường `program` — khoảng cách ngữ nghĩa không đủ để phân biệt, phải dùng metadata filter mới tách được.
>
> Cặp 3 cho thấy điểm số **không phải thang tuyệt đối**: hai chủ đề khác hẳn nhau vẫn đạt +0.458 chỉ vì cùng là từ vựng hành chính đại học. Phải so sánh điểm *tương đối trong cùng một truy vấn*, đừng đặt ngưỡng cứng kiểu "trên 0.5 là liên quan".

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chiến lược của tôi: **`HeadingChunker`** — cắt theo tiêu đề điều khoản (`## Điều N.`), section dài quá 1.200 ký tự thì hạ xuống recursive và **gắn lại tiêu đề vào từng mảnh con**. Corpus 10 tài liệu → **62 chunk**, dài trung bình 914 ký tự.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Trường tổ chức cho sinh viên đăng ký học muộn nhất bao lâu trước khi bắt đầu học kỳ? | `trach-nhiem-giang-vien` — Điều 22 Đề thi kết thúc học phần | +0.621 | ❌ Không | "Không có thông tin về thời gian muộn nhất trường tổ chức cho sinh viên đăng ký học" — **từ chối bịa** |
| 2 | Học cải thiện điểm được tối đa bao nhiêu tín chỉ trong học kỳ 1? | `dang-ky-hoc-phan` — Điều 11 Học lại | +0.770 | ✅ Có | "Trong học kỳ 1, sinh viên được học cải thiện điểm tối đa không quá **8 tín chỉ** [1]" |
| 3 | Khi không đồng ý với điểm thi thì làm gì? *(lọc `audience=student`)* | `phuc-khao-khieu-nai-diem` — Điều 26 | +0.498 | ✅ Có | Phân biệt 2 trường hợp: điểm giảng viên → khiếu nại trực tiếp giảng viên [1]; điểm thi học phần → nộp đơn Phòng Thanh tra, ĐBCLGD & Khảo thí [1] |
| 4 | Sinh viên được tuyển chọn vào chương trình Chất lượng cao như thế nào? *(lọc `program`)* | `dao-tao-chat-luong-cao` — Điều 11 | +0.764 | ✅ Có | Liệt kê diện xét tuyển thẳng: đội tuyển Olympic quốc tế, giải nhất/nhì/ba HSG quốc gia lớp 12 [3] |
| 5 | Điều kiện để được xét công nhận tốt nghiệp gồm những gì? | `tot-nghiep` — Điều 30 | +0.813 | ✅ Có | Liệt kê đủ 7 điều kiện a–g theo Điều 30 khoản 1 |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **4** / 5

> **Vì sao con số này khác báo cáo nhóm (10/10)?** Bảng trên đo bằng mô hình nhúng **cục bộ 384 chiều**. Cấu hình chính thức của nhóm dùng **`gemini-embedding-001` 3.072 chiều** và sửa được đúng câu 1, thành 5/5. Tôi giữ lại số 4/5 ở đây vì ca hỏng bên dưới **chỉ xuất hiện ở mô hình 384 chiều** — mà đó lại là phần phân tích có giá trị nhất của mục này: nó cho thấy chẩn đoán đầu tiên của tôi đã sai. Đối chiếu đầy đủ ở `REPORT_NHOM.md` mục 2.

### Phân tích câu hỏng (câu 1)

Top-1 là Điều 22 *"Đề thi kết thúc học phần"* — sai chủ đề hoàn toàn. Nguyên nhân **không phải** retrieval kém chung chung mà là **nhầm theo khuôn câu**: câu hỏi có dạng *"đăng ký X muộn nhất bao lâu trước khi bắt đầu X"*, và corpus có một điều khoản dạng *"đăng ký **thực tập** chậm nhất 1 tháng trước khi bắt đầu thực tập"*. Mô hình bắt đúng cấu trúc nhưng sai thực thể: **đăng ký thực tập ≠ đăng ký học phần**.

Chunk đúng (`dang-ky-hoc-phan`) đứng hạng **4/62**, điểm +0.576 so với +0.621 của top-1 — trượt rất sát.

**Đề xuất cải thiện:** chunk nhỏ hơn cho loại câu tra số liệu. Tôi đã đo lại cùng 5 câu qua 4 chiến lược:

| Chiến lược | Chunk | Câu 1 | Câu 2 | Câu 3 | Câu 4 | Câu 5 | Đúng |
|---|---|---|---|---|---|---|---|
| Theo câu (`SentenceChunker`) | 81 | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| Theo tiêu đề (của tôi) | 62 | ❌ | ✅ | ✅ | ✅ | ✅ | 4/5 |
| Đệ quy | 52 | ❌ | ❌ | ✅ | ✅ | ✅ | 3/5 |
| Kích thước cố định | 52 | ❌ | ❌ | ✅ | ✅ | ✅ | 3/5 |

`SentenceChunker` trả đúng câu 1 ở **hạng 1, điểm +0.849**, vì chunk 665 ký tự chứa đúng câu "muộn nhất 3 tuần trước thời điểm bắt đầu học kỳ" có mật độ tín hiệu cao hơn hẳn chunk 914 ký tự chứa cả Điều, nơi con số bị pha loãng.

Nhưng **đừng kết luận chunk nhỏ luôn tốt hơn**: chỉ tiêu "gold_doc trong top-3" không đo *Chunk Coherence* và *Grounding Quality*. Chunk theo Điều giữ trọn một điều khoản nên câu trả lời của agent ở câu 3 và câu 5 liệt kê được đầy đủ các khoản a–g; chunk theo câu dễ cắt mất phần sau của điều khoản. Đây là đánh đổi **precision cho tra cứu số liệu** đối lại **context đầy đủ cho sinh câu trả lời**, chứ không có chiến lược thắng tuyệt đối.

### Đính chính: đề xuất trên sai ở nguyên nhân gốc

Đề xuất "chunk nhỏ hơn" của tôi **chữa triệu chứng chứ không chữa nguyên nhân**. Khi nhóm đo lại cùng chiến lược `HeadingChunker` trên mô hình nhúng **3.072 chiều**, câu 1 tự đúng — cả top-3 đều là `dang-ky-hoc-phan` (+0.784 / +0.763 / +0.731) — và điểm lên **10/10** mà không đổi một tham số chunking nào.

| | Nhúng cục bộ 384 chiều | Gemini 3.072 chiều |
|---|---|---|
| Theo tiêu đề (chiến lược của tôi) | 8/10 | **10/10** |
| Theo câu | 9/10 | **10/10** |

Nghĩa là ưu thế của `SentenceChunker` ở bảng trên **không phải ưu thế bản chất**, nó chỉ là cách bù cho một mô hình nhúng yếu; nâng mô hình lên thì khác biệt giữa hai chiến lược biến mất hoàn toàn.

Bài học cá nhân tôi rút ra: **tôi đã chẩn đoán đúng hiện tượng nhưng sai tầng.** Nhìn thấy "chunk 914 ký tự pha loãng con số" là một quan sát hợp lý, nhưng tôi kết luận ngay đó là lỗi chunking mà chưa thử đổi biến khác. Đúng quy trình thì phải cô lập từng biến một — giữ nguyên chunking, chỉ đổi embedder — trước khi đổ lỗi cho tầng mình đang làm. Chi tiết đầy đủ ở `REPORT_NHOM.md` mục 2.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *[Điền sau buổi demo — ghi cụ thể chiến lược của nhóm nào, khác mình chỗ nào, và số liệu của họ ra sao.]*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 8 / 10 |
| **Tổng phần cá nhân** | **58 / 60** |

> Tự trừ 2 điểm ở mục 5 vì chỉ đạt 4/5 câu có chunk liên quan trong top-3.
