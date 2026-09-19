"use strict";

const form = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const submitButton = document.getElementById("submit");
const resultBox = document.getElementById("result");
const audienceSelect = document.getElementById("audience");
const programSelect = document.getElementById("program");

const PROFILE_KEY = "neu-quy-che-profile";

/** Tạo phần tử gọn. */
function el(tag, { class: className, text, ...attrs } = {}) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

/** localStorage có thể ném lỗi ở chế độ riêng tư — đừng để nó làm hỏng trang. */
function loadProfile() {
  try {
    return JSON.parse(localStorage.getItem(PROFILE_KEY)) || {};
  } catch {
    return {};
  }
}

function saveProfile() {
  try {
    localStorage.setItem(
      PROFILE_KEY,
      JSON.stringify({ audience: audienceSelect.value, program: programSelect.value })
    );
  } catch {
    /* không lưu được thì thôi, trang vẫn chạy */
  }
}

function fillSelect(select, options, anyLabel) {
  select.replaceChildren();
  if (anyLabel) select.append(new Option(anyLabel, ""));
  for (const [value, label] of Object.entries(options)) select.append(new Option(label, value));
}

async function loadMeta() {
  let meta;
  try {
    const response = await fetch("/api/meta");
    if (!response.ok) throw new Error();
    meta = await response.json();
  } catch {
    showNotice("Không kết nối được máy chủ. Kiểm tra xem lệnh python server.py còn chạy không.");
    return;
  }

  fillSelect(audienceSelect, meta.roles);
  fillSelect(programSelect, meta.programs, "Tất cả chương trình");

  const saved = loadProfile();
  audienceSelect.value = saved.audience || "student";
  programSelect.value = saved.program ?? "";

  document.getElementById("pill-version").textContent = `${meta.docs} văn bản · ${meta.chunks} điều khoản`;
  document.getElementById("pill-engine").textContent = meta.llm.split(":").pop();

  // Corpus hiện là quy chế 2012 — sinh viên cần biết trước khi tin con số.
  document.getElementById("stale").textContent =
    "Dữ liệu dựa trên Quy định đào tạo tín chỉ ban hành kèm QĐ 1212/QĐ-ĐHKTQD (2012). Nhà trường có thể đã ban hành văn bản mới hơn.";
}

function showNotice(message) {
  resultBox.replaceChildren(el("div", { class: "notice", text: message }));
}

/**
 * Dựng đoạn trả lời của mô hình.
 * Chỉ diễn giải **đậm** và trích dẫn [n]; phần còn lại chèn bằng text thuần —
 * nội dung do LLM sinh ra nên không dựng HTML tuỳ ý từ nó.
 */
function renderAnswer(text) {
  const wrap = el("div");
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;

    const paragraph = el("p");
    const pattern = /\*\*(.+?)\*\*|\[(\d+)\]/g;
    let cursor = 0;
    let match;

    while ((match = pattern.exec(line)) !== null) {
      if (match.index > cursor) paragraph.append(line.slice(cursor, match.index));

      if (match[1] !== undefined) {
        paragraph.append(el("strong", { text: match[1] }));
      } else {
        const number = match[2];
        const chip = el("button", {
          class: "cite",
          text: `[${number}]`,
          type: "button",
          title: `Xem điều khoản ${number}`,
        });
        chip.addEventListener("click", () => revealSource(number));
        paragraph.append(chip);
      }
      cursor = match.index + match[0].length;
    }
    paragraph.append(line.slice(cursor));
    wrap.append(paragraph);
  }
  return wrap;
}

function revealSource(number) {
  const details = document.querySelector(".sources");
  if (details) details.open = true;

  const target = document.getElementById(`nguon-${number}`);
  if (!target) return;
  target.scrollIntoView({ block: "center" });
  target.classList.add("flash");
  setTimeout(() => target.classList.remove("flash"), 1600);
}

function renderSource(source) {
  const node = document.getElementById("tpl-source").content.cloneNode(true);
  const article = node.querySelector(".source");
  article.id = `nguon-${source.n}`;

  node.querySelector(".cite-num").textContent = source.n;
  node.querySelector(".src-title").textContent = source.title;
  node.querySelector(".src-body").textContent = source.content;
  node.querySelector(".src-score").textContent = `độ khớp ${source.score.toFixed(3)}`;

  const parts = [source.version && `phiên bản ${source.version}`, source.program].filter(Boolean);
  node.querySelector(".src-meta").textContent = parts.join(" · ");

  const link = node.querySelector(".src-link");
  if (source.url) link.href = source.url;
  else link.remove();

  return node;
}

function renderResult(data) {
  resultBox.replaceChildren();

  if (!data.sources.length) {
    showNotice(
      "Không tìm thấy điều khoản nào khớp với lựa chọn của bạn. " +
        "Thử chọn “Tất cả chương trình”, hoặc hỏi theo cách khác."
    );
    return;
  }

  const answerBox = el("div", { class: "answer" });
  answerBox.append(el("h2", { text: "Trả lời" }));

  if (data.error) {
    answerBox.append(
      el("p", { text: `Không gọi được mô hình trả lời (${data.error}).` }),
      el("p", { text: "Các điều khoản liên quan vẫn ở ngay bên dưới, bạn đọc trực tiếp được." })
    );
  } else if (data.answer) {
    answerBox.append(renderAnswer(data.answer));
  }

  const details = el("details", { class: "sources" });
  details.open = true;
  details.append(el("summary", { text: `Căn cứ — ${data.sources.length} điều khoản` }));
  for (const source of data.sources) details.append(renderSource(source));
  answerBox.append(details);

  resultBox.append(answerBox);
}

async function ask(question) {
  submitButton.disabled = true;
  resultBox.setAttribute("aria-busy", "true");

  const loading = el("div", { class: "loading" });
  loading.append(el("div", { class: "spinner" }), el("span", { text: "Đang tra quy chế…" }));
  resultBox.replaceChildren(loading);

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        audience: audienceSelect.value || null,
        program: programSelect.value || null,
        top_k: 3,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      showNotice(data.error || "Máy chủ báo lỗi.");
      return;
    }
    renderResult(data);
  } catch {
    showNotice("Mất kết nối tới máy chủ.");
  } finally {
    submitButton.disabled = false;
    resultBox.setAttribute("aria-busy", "false");
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question) ask(question);
});

document.getElementById("topics").addEventListener("click", (event) => {
  const card = event.target.closest(".topic");
  if (!card) return;
  questionInput.value = card.dataset.q;
  ask(card.dataset.q);
});

for (const select of [audienceSelect, programSelect]) {
  select.addEventListener("change", saveProfile);
}

loadMeta();
