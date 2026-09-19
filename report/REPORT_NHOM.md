# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** G25
**Thành viên:** Nguyễn Đức Thắng · Ngô Tiến Dũng · Nguyễn Hải Long · Trần Anh Quân
**Ngày:** 2026-09-20

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng. Chi tiết thang điểm: `docs/SCORING.md`.
>
> Bốn báo cáo cá nhân kèm theo: `REPORT_CANHAN.md` (Nguyễn Đức Thắng) · `REPORT_TRANANHQUAN.md` · `REPORT_NGUYENHAILONG.md` · `REPORT_NGOTIENDUNG.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

> **Cấu hình đo.** Corpus `data/quy-che-dao-tao-neu/` (10 văn bản). Nhóm đo trên **hai backend nhúng** để tách bạch ảnh hưởng của chiến lược chunking với ảnh hưởng của mô hình nhúng: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều, chạy máy cục bộ) và `gemini-embedding-001` (3.072 chiều, gọi API). Sinh câu trả lời: `gemini-3.5-flash`. Công cụ: `bench.py`, `scripts/check_corpus.py`.

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Quy chế đào tạo đại học chính quy — Đại học Kinh tế Quốc dân (NEU)

**Tại sao nhóm chọn chủ đề này?**
> Đây là loại văn bản mà **bản thân nhóm là người dùng thật** — sinh viên NEU tra quy định đăng ký học phần, học lại, phúc khảo, tốt nghiệp hằng kỳ, nên nhóm tự kiểm chứng được câu trả lời đúng hay sai. Văn bản pháp quy còn có ba đặc tính rất hợp với bài lab: cấu trúc Chương/Điều rõ ràng nên chunk theo tiêu đề có căn cứ thật; nội dung chứa nhiều con số và mốc thời gian cụ thể nên gold answer kiểm chứng được, không mơ hồ; và cùng một văn bản lại phân tách sẵn theo đối tượng (Chương VI là trách nhiệm của Khoa/Viện/giảng viên) nên `metadata_filter` có việc thật để làm.

### Danh sách tài liệu (Data Inventory)

Nguồn HTML: `mfe.neu.edu.vn` (Bộ Quy định đào tạo tín chỉ ĐHKTQD, toàn văn 41 Điều).
Nguồn PDF: `aep.neu.edu.vn` (hai quy định chương trình đặc thù).

| # | Tên tài liệu (`doc_id`) | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | `dang-ky-hoc-phan` — Đăng ký khối lượng học tập, rút học phần và học lại | mfe.neu.edu.vn/bo-quy-dinh-dao-tao-tin-chi-dhktqd-2/ | 2026-09-19 / 1212/QĐ-ĐHKTQD (12-12-2012) | 4.184 | `audience=student`, `program=dai-tra`, `category=registration` |
| 2 | `nhiem-vu-quyen-sinh-vien` — Nhiệm vụ, quyền và tiêu chí đánh giá sinh viên | (như trên) | 2026-09-19 / 1212/QĐ-ĐHKTQD | 4.417 | `audience=student`, `program=dai-tra`, `category=student-rights` |
| 3 | `xep-hang-canh-bao-thoi-hoc` — Xếp hạng học lực, cảnh báo và buộc thôi học | (như trên) | 2026-09-19 / 1212/QĐ-ĐHKTQD | 2.639 | `audience=student`, `program=dai-tra`, `category=academic-standing` |
| 4 | `thi-ket-thuc-hoc-phan` — Kiểm tra, thi kết thúc học phần và cách tính điểm | (như trên) | 2026-09-19 / 1212/QĐ-ĐHKTQD | 5.115 | `audience=student`, `program=dai-tra`, `category=assessment` |
| 5 | `phuc-khao-khieu-nai-diem` — Khiếu nại điểm và xem lại kết quả bài thi | (như trên) | 2026-09-19 / 1212/QĐ-ĐHKTQD | 1.159 | `audience=student`, `program=dai-tra`, `category=assessment` |
| 6 | `tot-nghiep` — Thực tập cuối khoá, xét và công nhận tốt nghiệp | (như trên) | 2026-09-19 / 1212/QĐ-ĐHKTQD | 12.827 | `audience=student`, `program=dai-tra`, `category=graduation` |
| 7 | `trach-nhiem-giang-vien` — Trách nhiệm giảng viên trong ra đề, coi thi, chấm thi | (như trên) | 2026-09-19 / 1212/QĐ-ĐHKTQD | 8.813 | **`audience=faculty`**, `program=dai-tra`, `category=assessment` |
| 8 | `trach-nhiem-khoa-vien-phong` — Trách nhiệm Khoa, Viện, Phòng QLĐT, cố vấn học tập | (như trên) | 2026-09-19 / 1212/QĐ-ĐHKTQD | 4.422 | **`audience=staff`**, `program=dai-tra`, `category=administration` |
| 9 | `dao-tao-chat-luong-cao` — Quy định đào tạo chương trình Chất lượng cao | aep.neu.edu.vn/.../Quy-dinh-dao-tao-Chat-luong-cao.pdf | 2026-09-19 / 1299/QĐ-ĐHKTQD | 5.412 | `audience=student`, **`program=chat-luong-cao`**, `category=regulation` |
| 10 | `dao-tao-tien-tien` — Quy định đào tạo chương trình Tiên tiến | aep.neu.edu.vn/.../Quy-dinh-ve-dao-tao-theo-chuong-trinh-tien-tien.pdf | 2026-09-19 / **not-stated** | 5.146 | `audience=student`, **`program=tien-tien`**, `category=regulation` |

> Tài liệu 10 ghi `not-stated` vì trang bìa bản gốc chỉ ghi "HÀ NỘI THÁNG ……. NĂM 2012" mà không nêu số quyết định — nhóm **không bịa số hiệu**.

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Corpus chỉ chứa nguồn công khai và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc `not-stated`) trong metadata — xác minh bằng `python scripts/check_corpus.py data/quy-che-dao-tao-neu` → **CHECKPOINT 2: ĐẠT**.

> **Ghi chú đạo đức thu thập:** 4/8 URL trong danh sách ban đầu nằm trên `daotao.neu.edu.vn` và `phongctctqlsv.neu.edu.vn`, cả hai trả về **177 byte JavaScript cookie challenge** cho mọi client không phải trình duyệt. `robots.txt` không cấm, nhưng đó là cơ chế giới hạn truy cập, nên nhóm **đổi nguồn thay vì lách** theo đúng `docs/DATA_COLLECTION.md` mục 2. Bản quy định tín chỉ đăng lại trên `mfe.neu.edu.vn` (site Khoa Toán Kinh tế) là nguồn thay thế hợp lệ.

### Xử lý dữ liệu trước khi nạp

Hai bước làm sạch có ảnh hưởng đo được, không phải thao tác hình thức:

1. **Sửa mã ký tự tiếng Việt trong PDF.** Cả ba PDF bóc ra bằng `pymupdf4llm` đều mã hoá `ư` thành **U+01A3 (LATIN SMALL LETTER OI)** thay vì U+01B0. File Chất lượng cao hỏng **984/986 lần** — toàn văn bản là "chƣơng trình chất lƣợng cao". Để nguyên thì mọi truy vấn gõ đúng chính tả **không bao giờ khớp**, mà nhìn bằng mắt rất khó phát hiện vì `ƣ` và `ư` gần giống nhau. Script `scripts/build_pdf_docs.py` thay thế rồi chuẩn hoá NFC; hiện còn **0** ký tự hỏng.
2. **Cắt phần trùng lặp.** Hai quy định PDF ban đầu dài ~69.000 ký tự mỗi bản, song song gần như từng Điều với quy định đại trà. Nhóm chỉ giữ phần **đặc thù** (Điều 2 mục tiêu, Điều 7 tuyển chọn, Điều 8 chuyển đổi hệ, Điều 10 giảng viên/ngôn ngữ, Điều 11 nhập học) — xem phân tích ở mục 3 về lý do.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | `phuc-khao-khieu-nai-diem` | Khoá định danh tài liệu gốc; `delete_document()` và mọi thống kê đều nhóm theo trường này |
| `audience` | enum | `student` / `faculty` / `staff` | **Trục lọc bắt buộc của L3A.** Cùng chủ đề "bài thi" nhưng Điều 26 là quy trình của sinh viên còn Điều 22–23 là trách nhiệm của giảng viên — hai đáp án khác nhau |
| `program` | enum | `dai-tra` / `tien-tien` / `chat-luong-cao` | Ba chương trình có quy định song song gần như từng Điều; đo được similarity giữa hai văn bản này là **+0.759**, tức embedding *không* đủ sức phân biệt, phải dựa vào metadata |
| `category` | enum | `registration`, `assessment`, `graduation`… | Thu hẹp theo nghiệp vụ khi câu hỏi đã rõ lĩnh vực |
| `source_url` | url | `https://mfe.neu.edu.vn/...` | Truy vết nguồn — agent trích dẫn kèm URL nên người đọc kiểm chứng được |
| `document_version` | string | `1212/QĐ-ĐHKTQD (2012-12-12)` | Cảnh báo độ mới: corpus là bản 2012, có thể đã bị văn bản mới thay thế |
| `retrieved_at` | date | `2026-09-19` | Mốc thời điểm thu thập |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

### Phân tích đường cơ sở (Baseline Analysis)

`ChunkingStrategyComparator().compare(body, chunk_size=800)` trên 3 tài liệu đại diện. **Đã bỏ khối YAML front matter trước khi so sánh** — nếu không thì đang đo cả metadata chứ không phải nội dung.

| Tài liệu | Chiến lược | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| `dang-ky-hoc-phan` (4.184 ký tự) | `fixed_size` | 6 | 764,0 | Không — cắt giữa khoản, mất phần đầu quy định |
| | `by_sentences` | 13 | 319,6 | Một phần — câu trọn vẹn nhưng tách rời khỏi Điều chứa nó |
| | `recursive` | 6 | 695,7 | Khá — bám ranh giới đoạn văn |
| `tot-nghiep` (12.827 ký tự) | `fixed_size` | 18 | 788,2 | Không |
| | `by_sentences` | 22 | 580,6 | **Không ổn định** — chunk dài nhất tới **2.185 ký tự** |
| | `recursive` | 19 | 673,2 | Khá, nhưng có chunk chỉ **69 ký tự** |
| `dao-tao-chat-luong-cao` (5.412 ký tự) | `fixed_size` | 8 | 746,5 | Không |
| | `by_sentences` | 14 | 383,4 | Một phần |
| | `recursive` | 8 | 674,8 | Khá |

**Quan sát đáng chú ý từ baseline:** `by_sentences` cho độ lệch lớn nhất — trên `tot-nghiep` có chunk **2.185 ký tự** vì các khoản a)–g) được liệt kê không có dấu chấm câu nên regex tách câu không tìm thấy ranh giới nào. Ngược lại `recursive` sinh ra chunk **69 ký tự** vô nghĩa. Con số trung bình đẹp không đồng nghĩa phân bố đều.

### Chiến lược của từng thành viên

Bốn thành viên chọn bốn chiến lược khác nhau, đúng yêu cầu của đề bài. Tổng hợp từ bốn báo cáo cá nhân:

| Thành viên | Chiến lược | Corpus đã chạy | Mô hình nhúng | Kết quả tự báo |
|---|---|---|---|---|
| Nguyễn Đức Thắng | `HeadingChunker(1200)` — custom, theo tiêu đề Điều | 10 tài liệu · 62 chunk *(bản chính thức trong repo)* | local 384 → gemini 3.072 | 8/10 → **10/10** |
| Trần Anh Quân | `RecursiveChunker(500, overlap=50)` | **7 tài liệu · 1.414 chunk** | local 384 | **9/10** (5/5 top-3) |
| Nguyễn Hải Long | `MarkdownSectionChunker` — theo tiêu đề/mục | **corpus khác** (`rut-hoc-phan-va-nghi-tam-thoi`, `thang-diem-va-danh-gia-hoc-phan`…) | **mock** | 3/5 |
| Ngô Tiến Dũng | `RecursiveChunker` — *trùng họ chiến lược với Trần Anh Quân* | khớp `doc_id` bản chính thức | **mock** | 4/5 |

> ### ⚠ Số liệu giữa các thành viên **không so sánh trực tiếp được**
>
> Đề bài yêu cầu mọi thành viên chạy **cùng một corpus** và **cùng 5 câu hỏi** để chênh lệch điểm phản ánh đúng chiến lược. Khi gom bốn báo cáo lại, nhóm phát hiện điều đó **đã không xảy ra**:
>
> - **Ba corpus khác nhau.** Bản chính thức trong repo có 10 tài liệu / 62 chunk. Trần Anh Quân chạy trên **7 tài liệu / 1.414 chunk** với các `doc_id` như `nhung-dieu-sv-dh-ktqd-can-biet--2014` và `1155-quyche-daotao-daihoc-k63`. Nguyễn Hải Long chạy trên bộ `doc_id` thứ ba (`canh-bao-hoc-tap-va-buoc-thoi-hoc`, `trach-nhiem-co-van-va-giang-vien`…), **không tồn tại** trong `data/quy-che-dao-tao-neu/`.
> - **Ba mô hình nhúng khác nhau.** Hai thành viên dùng `MockEmbedder` (băm MD5, điểm số là nhiễu), một dùng local 384 chiều, một dùng cả local lẫn Gemini 3.072 chiều.
> - **Bộ câu hỏi lệch.** Long dùng 5 câu **tự đặt** ("rút học phần từ tuần thứ 3 đến tuần thứ 6", "CPA loại Giỏi bị hạ bậc khi nào") thay vì 5 câu chung của nhóm. Hai người còn lại dùng đúng bộ chung.
>
> Hệ quả: **9/10 của Quân không chứng minh `RecursiveChunker` mạnh hơn `HeadingChunker`** — corpus của bạn ấy lớn gấp 23 lần về số chunk và chứa văn bản mới hơn (QĐ 1155/K63) mà corpus chính thức không lấy được. Tương tự, 3/5 và 4/5 của Long và Dũng phần lớn phản ánh việc chạy trên `MockEmbedder` chứ không phải chất lượng chiến lược.
>
> Vì vậy phần so sánh bên dưới dùng một **phép đo có kiểm soát**: cùng corpus chính thức, cùng 5 câu hỏi, cùng mô hình nhúng, chỉ đổi chiến lược. Đó mới là cơ sở để kết luận.

**Thành viên 1 — Nguyễn Đức Thắng** — `HeadingChunker(1200)`, custom *(vai bắt buộc của L3A: chunk theo tiêu đề/mục)*
- **Lý do chọn:** Văn bản pháp quy đã được **người soạn chia sẵn** thành Điều, mỗi Điều là một đơn vị ngữ nghĩa trọn vẹn — không ranh giới nào ta tự đoán tốt hơn thế. Hai chi tiết phải xử lý đúng trên corpus này: (a) file từ HTML đánh dấu `## Điều N` còn file từ PDF đánh dấu `**Điều N**`, chỉ tách theo markdown heading thì hai quy định PDF thành một khối khổng lồ và át hết top-k; (b) tiêu đề không có phần thân riêng (`CHƯƠNG III. TỔ CHỨC ĐÀO TẠO`) không phải đơn vị truy xuất nên được gộp vào section kế tiếp thay vì thành chunk rỗng nghĩa. Section dài quá ngưỡng thì hạ xuống recursive và **gắn lại tiêu đề vào từng mảnh con**, nếu không mảnh thứ hai trở đi mất ngữ cảnh "đây là Điều nào".
- **Code snippet:**
```python
class HeadingChunker:
    SECTION_BREAK = re.compile(r"\n(?=#{1,6} |\*\*\s*(?:Điều|CHƯƠNG|Chương)\s)")

    def __init__(self, max_chars: int = 1200) -> None:
        self.max_chars = max_chars
        self._fallback = RecursiveChunker(chunk_size=max_chars)

    def chunk(self, text: str) -> list[str]:
        chunks, carried = [], ""
        for section in self.SECTION_BREAK.split(text):
            section = section.strip()
            if not section:
                continue
            heading, body = self._split_heading(section)
            if not body:                       # tiêu đề trần -> mang sang section sau
                carried = f"{carried}\n{heading}".strip() if carried else heading
                continue
            if carried:
                heading, carried = f"{carried}\n{heading}".strip(), ""
            whole = f"{heading}\n\n{body}" if heading else body
            if len(whole) <= self.max_chars:
                chunks.append(whole)
                continue
            for piece in self._fallback.chunk(body):   # gắn lại tiêu đề vào mảnh con
                chunks.append(f"{heading}\n\n{piece}" if heading else piece)
        if carried:
            chunks.append(carried)
        return chunks
```
- **Hiệu quả của hai sửa lỗi trên:** chunk ngắn dưới 120 ký tự giảm từ **23 → 0**; chunk ngắn nhất từ 17 → 124 ký tự.

**Thành viên 2 — Trần Anh Quân** — `RecursiveChunker(chunk_size=500, overlap=50)`
- **Lý do chọn:** `FixedSizeChunker` cắt cơ học tại ký tự thứ 500 nên câu mở đầu bị tách khỏi danh sách điều kiện, chunk phía sau mất chủ ngữ và ngữ cảnh pháp lý. `SentenceChunker` thì bị dấu chấm phẩy (`;`) giữa các điểm và câu ghép rất dài làm kích thước biến thiên thất thường. `RecursiveChunker` ưu tiên ranh giới lớn (`\n\n` giữa các Điều, `\n` giữa các khoản) rồi greedy-merge các dòng ngắn liền kề tới ~500 ký tự, nên mỗi chunk giữ trọn một quy định.
- **Baseline của bạn ấy** (trên corpus 7 tài liệu): `FixedSizeChunker(200,20)` → 428 chunk (nhiều chunk cắt đôi giữa từ); `SentenceChunker(3)` → 312 chunk (80–650 ký tự, không đều); `RecursiveChunker(500)` → 196 chunk, trung bình 412 ký tự.
- **Kết quả:** 5/5 câu có chunk liên quan trong top-3, **9/10 điểm** (4 câu top-1, 1 câu top-3). Điểm số cao (0.75–0.86) vì dùng mô hình nhúng đa ngữ thật.

**Thành viên 3 — Nguyễn Hải Long** — `MarkdownSectionChunker` (theo tiêu đề/mục)
- **Lý do chọn:** mỗi điều khoản tự nó đã là một khối logic hoàn chỉnh; kết hợp cơ chế **gắn lại tiêu đề mục cha vào từng mảnh con** để xử lý vấn đề "mất gốc ngữ cảnh" mà `FixedSizeChunker` hay gặp.
- **Kết quả:** 3/5 câu có chunk liên quan trong top-3 — nhưng chạy trên `MockEmbedder` và trên bộ 5 câu tự đặt, nên con số này **không phản ánh chất lượng chiến lược**. Chính bạn ấy cũng ghi nhận điểm số âm bất thường ở mục 4 (cặp câu cùng nghĩa ra **−0.3352**) là do mock.

**Thành viên 4 — Ngô Tiến Dũng** — `RecursiveChunker`
- **Lý do chọn:** *"linh hoạt ứng dụng `RecursiveChunker` để bảo toàn cấu trúc văn bản pháp lý"* — cùng lập luận với Trần Anh Quân: ưu tiên ranh giới lớn (`\n\n` giữa các Điều, `\n` giữa các khoản) rồi gộp thông minh các mẩu nhỏ liền kề để tận dụng tối đa `chunk_size` trước khi mở chunk mới.
- **Kết quả:** 4/5 câu có chunk liên quan trong top-3, chạy trên `MockEmbedder` (điểm 0.34–0.44).
- **Đóng góp riêng đáng ghi nhận:** bạn ấy là người đầu tiên trong nhóm chỉ ra đúng nguyên nhân của ca hỏng ở câu 1 — *"Mock Embedder dựa trên hash từ vựng không nắm bắt được ngữ nghĩa 'thời hạn đăng ký'… cần Semantic Embedding thật (Gemini/OpenAI) để cải thiện recall"*. Nhận định này về sau được xác nhận bằng số: đổi sang `gemini-embedding-001` thì câu 1 tự đúng (xem tiểu mục cuối mục này).

> **Hai thành viên trùng chiến lược.** Ngô Tiến Dũng và Trần Anh Quân cùng chọn `RecursiveChunker`, nên nhóm thực chất chỉ phủ được **ba** họ chiến lược (theo tiêu đề · đệ quy · theo tiêu đề dạng markdown) trong khi đề bài khuyến khích mỗi người một hướng khác nhau. `FixedSizeChunker` — đường cơ sở đơn giản nhất — không ai nhận. Nhóm bù bằng cách đưa nó vào bảng đo có kiểm soát bên dưới, nhưng đó là đo thêm chứ không phải một thành viên thực sự thử nghiệm và giải thích.

### So Sánh Bốn Chiến Lược

Chấm theo `docs/SCORING.md`: **2 điểm** nếu chunk đúng ở top-1, **1 điểm** nếu có trong top-3 nhưng không phải top-1, **0 điểm** nếu ngoài top-3. Cùng corpus, cùng 5 câu hỏi, cùng mô hình nhúng cục bộ (384 chiều).

| Chiến lược | Số chunk | C1 | C2 | C3 | C4 | C5 | Điểm (/10) |
|----------|---|---|---|---|---|---|----------------------|
| Theo tiêu đề (`HeadingChunker`, 1200) | 62 | 0 | 2 | 2 | 2 | 2 | **8/10** |
| Theo câu (`SentenceChunker`, 5 câu) | 81 | 2 | 2 | 1 | 2 | 2 | **9/10** |
| Kích thước cố định (800 / overlap 80) | 80 | 2 | 2 | 1 | 2 | 2 | **9/10** |
| Đệ quy (`RecursiveChunker`, 800) | 81 | 1 | 0 | 1 | 2 | 2 | **6/10** |

| Chiến lược | Điểm mạnh | Điểm yếu |
|---|---|---|
| Theo tiêu đề | Chunk luôn là một Điều trọn vẹn → agent liệt kê được đủ khoản a)–g) ở câu 5; mọi chunk đều có tiêu đề nên truy vết nguồn rõ ràng | Con số cụ thể bị pha loãng trong chunk ~914 ký tự → **trượt câu 1** (với mô hình 384 chiều) |
| Theo câu | Mật độ tín hiệu cao, thắng câu tra số liệu (C1 đạt top-1, +0.849) | Phân bố rất lệch (chunk tới 2.185 ký tự trên `tot-nghiep`); cắt ngang khoản nên C3 chỉ còn 1 điểm |
| Kích thước cố định | Đơn giản, overlap 10% cứu được câu bị cắt ngang ranh giới | Cắt giữa câu, chunk không đọc độc lập được — kém cho phần sinh câu trả lời |
| Đệ quy | Bám ranh giới đoạn văn tự nhiên | Sinh chunk vụn 69 ký tự; **kém nhất trên bộ này (6/10)** |

> **Cảnh báo về tham số:** cùng `RecursiveChunker` nhưng `chunk_size=800` cho 81 chunk và 6/10, còn `chunk_size=1200` cho 52 chunk và kết quả khác hẳn. **Tham số ảnh hưởng ngang với việc chọn chiến lược** — so sánh giữa các thành viên bắt buộc phải ghi rõ tham số, nếu không là đang so hai thứ khác nhau.

### Đổi mô hình nhúng đáng giá hơn đổi chiến lược chunking

Đây là phát hiện quan trọng nhất của nhóm, và nó chỉ lộ ra khi đo cùng một chiến lược trên hai backend:

| Chiến lược | Nhúng cục bộ (384 chiều) | Gemini (3.072 chiều) |
|---|---|---|
| Theo tiêu đề | 8/10 | **10/10** |
| Theo câu | 9/10 | **10/10** |

Với mô hình 3.072 chiều, **cả hai chiến lược đều đạt điểm tuyệt đối** — khoảng cách giữa chúng biến mất hoàn toàn. Đổi chiến lược chunking chỉ xê dịch điểm trong khoảng 6–9/10; đổi mô hình nhúng đưa thẳng lên 10/10.

Cụ thể ở **câu 1** — câu duy nhất mà chiến lược theo tiêu đề trượt. Mô hình 384 chiều trả về Điều 22 *"Đề thi kết thúc học phần"*, tức nó bắt đúng khuôn câu *"đăng ký X trước khi bắt đầu X"* nhưng sai thực thể (**đăng ký thực tập** ≠ **đăng ký học phần**). Mô hình 3.072 chiều phân biệt được, trả về cả top-3 đều là `dang-ky-hoc-phan` (+0.784 / +0.763 / +0.731).

Nhóm từng kết luận sai rằng đó là **giới hạn của chiến lược chunking**, và định sửa bằng cách chia nhỏ hơn. Hoá ra đó là **giới hạn của mô hình nhúng**.

**Cái giá phải trả:**

| | Nhúng cục bộ | Gemini API |
|---|---|---|
| Nhúng 62 chunk | **2,7 s** | 33,9 s |
| Nhúng 81 chunk | 3,6 s | **106 s** (phải tiết lưu) |
| Hạn mức | không giới hạn, chạy offline | **100 request/phút** (free tier) |
| Số chiều | 384 | 3.072 |

Khi nhóm thử đo cả 4 chiến lược bằng Gemini trong một lượt (~300 lượt nhúng), API trả **429 RESOURCE_EXHAUSTED** ngay ở chiến lược thứ hai. Phải chèn nghỉ 65 giây sau mỗi 80 lượt gọi mới chạy xong. Đây là ràng buộc vận hành thật: mô hình mạnh hơn nhưng **không chạy nổi vòng lặp so sánh nhiều chiến lược** trên hạn mức miễn phí.

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> Câu trả lời **phụ thuộc vào mô hình nhúng**, và đó chính là bài học.
>
> Với mô hình cục bộ 384 chiều: chia theo câu và kích thước cố định cùng đạt 9/10, cao hơn chia theo tiêu đề (8/10) — **ngược với dự đoán ban đầu** của nhóm. Lý do: 4/5 câu hỏi là tra một dữ kiện cụ thể, mà chunk nhỏ cho mật độ tín hiệu cao hơn.
>
> Với mô hình 3.072 chiều: cả hai cùng 10/10, **thứ hạng biến mất**. Nghĩa là ưu thế của chunk nhỏ ở trên không phải ưu thế bản chất — nó chỉ là cách bù cho một mô hình nhúng yếu.
>
> Nên nhóm chọn **chia theo tiêu đề** làm cấu hình chính thức, dù nó thua ở bảng 384 chiều. Khi hai chiến lược ngang điểm truy xuất, cái giữ trọn một Điều thắng ở phần mà thang điểm không đo: **chất lượng câu trả lời**. Ở câu 5, chunk theo tiêu đề giữ trọn Điều 30 nên agent liệt kê đủ **7 điều kiện a)–g)**; chia theo câu cắt ngang giữa danh sách. Ở câu 3, chia theo câu cắt rời hai khoản của Điều 26 nên tụt xuống 1 điểm ngay cả trên bảng 384 chiều.
>
> Kết luận: **tối ưu chunking là việc làm sau cùng, không phải đầu tiên.** Đầu tư vào mô hình nhúng và vào việc làm sạch corpus cho kết quả lớn hơn nhiều. Chunking nên chọn theo tiêu chí "chunk có đọc độc lập được không", chứ không phải đấu điểm top-k.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Trường tổ chức cho sinh viên đăng ký học muộn nhất bao lâu trước khi bắt đầu học kỳ? *(tra số liệu)* | **Muộn nhất 3 tuần** trước thời điểm bắt đầu học kỳ | `dang-ky-hoc-phan` — Điều 10 khoản 3 |
| 2 | Học cải thiện điểm được tối đa bao nhiêu tín chỉ trong học kỳ 1? *(hỏi điều kiện)* | **Không quá 8 tín chỉ** với học kỳ 1 và học kỳ 3; không quá 5 tín chỉ với học kỳ còn lại | `dang-ky-hoc-phan` — Điều 11 |
| 3 | Khi không đồng ý với điểm thi thì làm gì? *(quy trình — **cần `audience=student`**)* | Điểm đánh giá của giảng viên và điểm kiểm tra → khiếu nại **trực tiếp đến giảng viên**. Điểm thi học phần → **nộp đơn cho Phòng Thanh tra, Đảm bảo chất lượng giáo dục và Khảo thí**, công bố kết quả **sau 3 tuần** kể từ ngày nhận đơn. Sai lệch điểm nhập mạng → đề nghị Phòng QLĐT đối chiếu **trong không quá 6 tháng** | `phuc-khao-khieu-nai-diem` — Điều 26 |
| 4 | Sinh viên được tuyển chọn vào chương trình Chất lượng cao như thế nào? *(hỏi điều kiện — cần `program`)* | **Xét tuyển thẳng** (nếu đạt yêu cầu tiếng Anh) cho: thành viên đội tuyển quốc gia dự Olympic quốc tế + tốt nghiệp THPT khá trở lên; học sinh đoạt giải nhất/nhì/ba HSG quốc gia lớp 12 + tốt nghiệp THPT khá trở lên; sinh viên được tuyển thẳng vào ĐHKTQD. Ngoài ra sinh viên trúng tuyển đạt mức điểm do Hiệu trưởng quy định hằng năm được **đăng ký dự tuyển** | `dao-tao-chat-luong-cao` — Điều 7 |
| 5 | Điều kiện để được xét công nhận tốt nghiệp gồm những gì? *(liệt kê)* | 7 điều kiện a)–g): không bị truy cứu hình sự / không đang bị đình chỉ; tích lũy đủ học phần; **GPA tích lũy ≥ 2,00**; có chứng chỉ GDQP và GDTC; hoàn thành học chính trị đầu khoá; hoàn thành học phí, lệ phí; có đơn nếu tốt nghiệp sớm/muộn | `tot-nghiep` — Điều 30 khoản 1 |

> **Thiết kế câu 3 (câu bắt buộc dùng metadata filter):** câu hỏi **cố ý không nêu người hỏi là ai**, và "điểm thi" là từ vựng dùng chung giữa Điều 26 (sinh viên khiếu nại) với Điều 22–23 (giảng viên ra đề, coi thi). Cả hai tài liệu còn cùng nhắc tên "Phòng Thanh tra, Đảm bảo chất lượng giáo dục và Khảo thí" nên cạnh tranh trực tiếp.

### Tổng hợp chất lượng truy xuất của nhóm

Cấu hình chính thức: `HeadingChunker(1200)` + `gemini-embedding-001` + `gemini-3.5-flash`.

| # | Câu hỏi | Có chunk liên quan trong top-3? | Điểm | Ghi chú |
|---|---------|-------------------------------|------|---------|
| 1 | Hạn đăng ký học | ✅ Có, cả top-3 (+0.784) | 2 | Với mô hình 384 chiều câu này **trượt**; mô hình 3.072 chiều sửa được |
| 2 | Cải thiện điểm | ✅ Có, top-1 (+0.809) | 2 | Agent trả đúng "không quá **8 tín chỉ**" [1] |
| 3 | Phúc khảo | ✅ Có, top-1 | 2 | Agent tách đúng 2 trường hợp, trích dẫn [1] |
| 4 | Tuyển chọn CLC | ✅ Có, top-1 | 2 | Lọc `program=chat-luong-cao` thu về đúng một tài liệu |
| 5 | Điều kiện tốt nghiệp | ✅ Có, top-1 | 2 | Agent liệt kê đủ 7 khoản a)–g) nhờ chunk giữ trọn Điều 30 |

**Tổng: 10/10 điểm** — 5/5 câu có chunk liên quan ở top-1.

> Với mô hình nhúng cục bộ, cùng chiến lược này đạt **8/10** (câu 1 trượt khỏi top-3). Nhóm giữ lại cả hai con số vì so sánh giữa chúng chính là nội dung mục 2.

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> **Có, nhưng mức độ phụ thuộc vào việc corpus có cân bằng hay không** — và nhóm học được điều này theo cách khó khăn.
>
> Ở **câu 3**, không lọc thì `trach-nhiem-giang-vien` (`audience=faculty`) chiếm hạng 1, tức sinh viên nhận về quy trình dành cho giảng viên; lọc `audience=student` thì Điều 26 lên hạng 1 — chênh lệch đúng **1 điểm → 2 điểm** theo rubric. Ở **câu 4**, lọc `program=chat-luong-cao` loại sạch quy định Tiên tiến vốn song song gần như từng Điều (similarity giữa hai văn bản là **+0.759**, embedding không tự phân biệt nổi).
>
> Bài học tốn công nhất: ở phiên bản corpus trước, hai quy định PDF được giữ nguyên **69.000 ký tự mỗi bản**, chiếm **76% tổng số chunk** và đẩy pool `audience=student` lên **93%** — lúc đó bật hay tắt filter cho **kết quả giống hệt nhau**, tức bộ lọc hoàn toàn vô dụng. Sau khi cắt hai tài liệu xuống chỉ phần đặc thù, tổng chunk giảm **203 → 62**, pool student còn 77%, và filter mới có tác dụng đo được. **Metadata schema đẹp trên giấy không có nghĩa là nó lọc được gì — phải đo.**

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

### Kịch bản demo (10 phút)

| Phút | Nội dung | Người trình bày | Chạy gì trên màn hình |
|---|---|---|---|
| 0–2 | Chủ đề, corpus 10 văn bản, cách thu thập và hai lỗi dữ liệu đã sửa | Ngô Tiến Dũng | `scripts/check_corpus.py` → CHECKPOINT 2: ĐẠT |
| 2–4 | Giao diện sinh viên: hỏi *"Khi không đồng ý với điểm thi thì làm gì?"*, mở khối Căn cứ, bấm trích dẫn `[1]` | Nguyễn Hải Long | `python server.py` |
| 4–7 | A/B bộ lọc `audience` và `program`; so sánh 4 chiến lược chunking trực tiếp | Trần Anh Quân | Streamlit, tab **So sánh chiến lược** |
| 7–9 | Phát hiện chính: đổi embedder 8/10 → 10/10; lỗi quy trình ba corpus | Nguyễn Đức Thắng | bảng ở mục 2 |
| 9–10 | Hỏi đáp | cả nhóm | |

**Hai điều nhóm muốn nghe ý kiến nhóm khác:** (1) nhóm nào giữ được corpus chung xuyên suốt và làm cách nào; (2) có nhóm nào thấy chunk theo tiêu đề thắng rõ khi dùng embedder mạnh không, hay kết quả "hoà 10/10" chỉ là do bộ 5 câu hỏi của nhóm quá dễ.

### Những phân tích (insights) hay nhất nhóm sẽ trình bày
> 1. **Đổi mô hình nhúng đáng giá hơn đổi chiến lược chunking.** Cùng chiến lược, đổi từ 384 lên 3.072 chiều: 8/10 → **10/10**. Trong khi đổi qua lại bốn chiến lược chunking chỉ xê dịch 6–9/10. Nhóm đã suýt kết luận sai rằng câu hỏng là do chunking, rồi định sửa bằng cách chia nhỏ hơn.
> 2. **Tiếng Việt không dấu phá hỏng embedding.** Cùng một câu, chỉ khác dấu, similarity chỉ đạt **+0.082** — gần như trực giao. Trong khi hai câu cùng nghĩa nhưng khác hoàn toàn từ vựng đạt **+0.875**. Hệ quả: benchmark query bắt buộc phải gõ đủ dấu.
> 3. **Lỗi mã ký tự trong PDF là sát thủ thầm lặng.** 984/986 chữ `ư` trong một văn bản bị mã hoá sai thành U+01A3. Nhìn bằng mắt gần như không phát hiện được, nhưng nó khiến mọi truy vấn gõ đúng chính tả không bao giờ khớp.
> 4. **Corpus mất cân bằng giết chết metadata filter.** Hai tài liệu chiếm 76% index thì bật/tắt filter cho kết quả y hệt.
> 5. **Agent từ chối bịa là kết quả tốt, không phải thất bại.** Khi retrieval trượt, agent trả lời *"không có thông tin…"* thay vì bịa số liệu — nhờ ràng buộc "chỉ dùng ngữ cảnh được cung cấp" trong prompt.
> 6. **Robots.txt không phải giấy phép.** 4/8 URL ban đầu bị chặn bằng JS cookie challenge dù robots.txt cho phép; nhóm đổi nguồn thay vì lách.
> 7. **Không chốt corpus chung trước thì bốn bộ số liệu thành vô dụng để so sánh.** Bốn thành viên chạy trên ba corpus và ba mô hình nhúng khác nhau — xem phân tích ở mục 2.

**Bài học rút ra khi so sánh trong nhóm:**
> Bài học lớn nhất lại là một **lỗi quy trình**, không phải lỗi kỹ thuật. Khi gom bốn báo cáo cá nhân, nhóm phát hiện mình đã chạy trên **ba corpus khác nhau** (62 chunk / 1.414 chunk / một bộ `doc_id` thứ ba) và **ba mô hình nhúng khác nhau** (mock, local 384 chiều, Gemini 3.072 chiều), trong đó một thành viên còn dùng 5 câu hỏi tự đặt. Kết quả: 9/10 của người này và 3/5 của người kia **không nói lên điều gì về chiến lược của họ** — chênh lệch đến từ corpus và backend, không từ cách chunk.
>
> Nhóm phải làm lại một phép đo có kiểm soát (cùng corpus, cùng câu hỏi, cùng backend, chỉ đổi chiến lược) thì mới rút ra được kết luận. Và phép đo đó cho thấy: **chia theo câu và kích thước cố định (9/10) nhỉnh hơn chia theo tiêu đề (8/10)** trên mô hình 384 chiều — ngược dự đoán ban đầu — nhưng khi nâng lên 3.072 chiều thì **khác biệt biến mất hoàn toàn**, chứng tỏ ưu thế kia chỉ là cách bù cho mô hình yếu.
>
> Tóm lại: **chất lượng chunking không đo được nếu tách khỏi corpus, loại câu hỏi và mô hình nhúng.** Ba thứ đó định hình kết luận nhiều ngang với bản thân chiến lược — nên chúng phải được chốt và ghi rõ *trước* khi từng người bắt đầu đo.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu?**
> **Thứ nhất — chốt và khoá corpus chung trước khi ai bắt đầu đo.** Đây là thay đổi quan trọng nhất. Nhóm sẽ commit corpus vào repo, ghi rõ số tài liệu và số chunk, rồi mới chia việc; mỗi người bắt buộc ghi `Embedding backend:` và `doc_id` trong báo cáo để kiểm chéo được.
>
> **Thứ hai — thống nhất mô hình nhúng.** Hai trong bốn thành viên nộp số liệu chạy trên `MockEmbedder`, vốn là hàm băm MD5 nên mọi thứ hạng đều là nhiễu. Phụ lục B của lab có sẵn hướng dẫn bật embedder thật; đáng ra phải làm từ đầu buổi vì lần cài đầu tải khá lâu.
>
> **Thứ ba — thử fetch thật trước khi chốt danh sách URL**, nhóm mất thời gian vào 4 link chết vì chỉ xem `robots.txt`.
>
> **Thứ tư — đọc lại text sau mỗi bước chuyển đổi định dạng**, vì lỗi mã ký tự `ư` → U+01A3 không hiện ra ở bước nào khác.
>
> **Thứ năm — lấy nguồn mới hơn.** Corpus chính thức là bản 2012 (QĐ 1212), trong khi Trần Anh Quân lấy được cả **QĐ 1155/K63** và **Sổ tay 2014** mà bản chính thức không fetch được. Nếu làm lại, nhóm sẽ gộp nguồn của bạn ấy vào corpus chung — một hệ thống tra cứu quy chế trả về quy định đã hết hiệu lực là sai nguy hiểm, và `document_version` tồn tại chính để lộ ra rủi ro này.
>
> Chú ý một dấu hiệu của việc dùng nguồn khác nhau: cùng câu hỏi 2, corpus chính thức (QĐ 1212) ghi *"không quá 8 tín chỉ đối với học kỳ 1, học kỳ 3"*, còn nguồn Sổ tay 2014 của Quân ghi *"học kỳ 1, học kỳ 2"*. **Gold answer lệch nhau vì văn bản nguồn lệch nhau** — thêm một lý do phải khoá corpus trước.

---

## 5. Sản phẩm demo

| Thành phần | Chạy bằng |
|---|---|
| Giao diện sinh viên (HTML/CSS/JS, không framework) | `python server.py` → http://localhost:8000 |
| Công cụ phòng lab (Streamlit: đổi chiến lược, so sánh, xem corpus) | `streamlit run streamlit_app.py` |
| Đo benchmark | `python bench.py` |
| Kiểm corpus | `python scripts/check_corpus.py data/quy-che-dao-tao-neu` |

Giao diện sinh viên dịch bộ lọc metadata sang ngôn ngữ đời thường — *"Tôi là [Sinh viên] chương trình [Đại trà]"* thay vì `audience` / `program` — có 6 thẻ tình huống thường gặp, và trích dẫn `[1]` bấm được để nhảy tới đúng điều khoản. Chân trang cảnh báo rõ dữ liệu dựa trên văn bản 2012.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | 9 / 10 |
| Thiết kế chiến lược (Strategy Design) | 12 / 15 |
| Chất lượng truy xuất (Retrieval Quality) | 10 / 10 |
| Thuyết trình (Demo) | 4 / 5 *(dự kiến — chưa thuyết trình)* |
| **Tổng phần nhóm** | **35 / 40** |

> **Mục 1 — trừ 1 điểm:** corpus là bản quy chế 2012 đã có khả năng bị thay thế, và 8/10 tài liệu đến từ một nguồn duy nhất.
>
> **Mục 2 — trừ 3 điểm:** hai lý do. (a) Bốn thành viên **chạy trên ba corpus và ba mô hình nhúng khác nhau**, nên số liệu giữa các thành viên không so sánh trực tiếp được — phần "so sánh trong nhóm" phải dựa vào một phép đo bổ sung có kiểm soát thay vì dùng thẳng kết quả của từng người. (b) **Hai thành viên trùng chiến lược** (`RecursiveChunker`), nên nhóm chỉ phủ được ba họ chiến lược thay vì bốn.
>
> **Mục 3 — 10/10:** đo trên cấu hình chính thức (`HeadingChunker` + `gemini-embedding-001`), 5/5 câu có chunk đúng ở top-1.
>
> **Mục 4 — dự kiến 4/5:** nhóm có sản phẩm demo chạy được (3 giao diện), kịch bản phân vai rõ và phần bài học đủ mạnh. Tự trừ 1 điểm vì phần "so sánh trong nhóm" — thứ rubric muốn thấy nhất ở buổi demo — phải trình bày như một **bài học từ sai sót** chứ không phải một so sánh hợp lệ giữa bốn thành viên.
