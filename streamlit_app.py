#!/usr/bin/env python3
"""Local multi-version PDF comparison app with topic feedback, placement, memos and persistence."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import uuid
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import streamlit as st
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PDF草案比較", layout="wide", page_icon="📄")
APP_DIR = Path(__file__).resolve().parent
AUTOSAVE_PATH = APP_DIR / ".pdf_compare_autosave.zip"
DEFAULT_PDFS_DIR = APP_DIR / "documents"
TRAINING_SEED_PATH = APP_DIR / "training_data_seed.json"
FORMAT = "pdf-compare-project-v2"
CSS = """
<style>
:root{--ink:#182721;--muted:#63716a;--line:#d9e2dd;--green:#176b4e;--greenbg:#e2f4e9;--red:#a03c35;--redbg:#fde9e6;--gold:#96600d;--goldbg:#fff1ce;--blue:#245f8c;--bluebg:#e6f1fb}
.block-container{max-width:1800px;padding-top:1.1rem}.app-note{padding:.8rem 1rem;border-left:4px solid #e2b84d;background:#fff8e7;border-radius:4px;color:#46391d}
.panel{border:1px solid var(--line);border-radius:12px;padding:12px;background:#fff;overflow-wrap:anywhere;min-height:100%}.empty-panel{border:1px dashed var(--line);border-radius:12px;padding:20px 8px;color:var(--muted);text-align:center;background:#f8faf8;font-size:.85rem}
.meta{font-size:.75rem;color:var(--muted);margin-bottom:.45rem}.en{font: .96rem/1.6 Georgia,'Times New Roman',serif;white-space:pre-wrap}.para-number{font-weight:700;color:#43564c}.memo{white-space:pre-wrap;margin-top:.75rem;padding-top:.6rem;border-top:1px solid #edf1ee;color:#35463d;font-size:.9rem}.memo-label{font-weight:700;font-size:.75rem;color:var(--muted)}
.search-hit{background:#ffb347!important;color:#4b2a00}.manual-highlight{background:#fff176;color:#3f3500}
.diff-add{background:var(--greenbg);color:var(--green)}.diff-del{background:var(--redbg);color:var(--red);text-decoration:line-through}.badge{display:inline-block;border-radius:99px;padding:.12rem .48rem;font-size:.7rem;font-weight:700;margin:.08rem .15rem .08rem 0;background:#edf1ef;color:#43564c}.b-add{background:var(--greenbg);color:var(--green)}.b-del{background:var(--redbg);color:var(--red)}.b-change{background:var(--goldbg);color:var(--gold)}.b-move{background:var(--bluebg);color:var(--blue)}.watch-item{border:1px solid #eadcae;background:#fffcf0;border-radius:10px;padding:12px;margin:.5rem 0}.small{font-size:.82rem;color:var(--muted)}
</style>
"""
STOP_LINES = {"DRAFT TEXT", "UNEDITED VERSION", "THE PRESENT DOCUMENT IS BEING ISSUED WITHOUT FORMAL EDITING."}
NUM_LINE = re.compile(r"^\s*(\d{1,3})\s*[.)](?:\s+(.*?)\s*|\s*)$")
CHAPTER_LINE = re.compile(r"^([IVXLCDM]+)\.\s+(.+?)\s*$", re.I)
SUBSECTION_LINE = re.compile(r"^([A-Z])\.\s+(.+?)\s*$")
DEFAULT_TOPIC_TAGS = [
    {"id": "mitigation", "name": "mitigation", "description": "排出削減、緩和目標、エネルギー転換など。"},
    {"id": "adaptation", "name": "adaptation", "description": "適応、レジリエンス、適応計画・目標など。"},
    {"id": "transparency", "name": "transparency", "description": "報告、透明性枠組み、測定・報告・検証など。"},
    {"id": "loss-and-damage", "name": "loss and damage", "description": "損失と損害、対応措置、基金・支援など。"},
    {"id": "just-transition", "name": "just transition", "description": "公正な移行、雇用、労働者・地域社会への影響など。"},
    {"id": "finance", "name": "finance", "description": "気候資金、資金目標、資金アクセス・提供など。"},
    {"id": "technology-capacity", "name": "technology & capacity-building", "description": "技術開発・移転、能力構築、実施手段など。"},
    {"id": "science-observation", "name": "science & observation", "description": "IPCC、気候科学、地球・海洋観測、早期警戒など。"},
    {"id": "trade-unilateral", "name": "trade & unilateral measures", "description": "貿易、越境影響、一方的措置、WTO等との対話など。"},
    {"id": "ecosystems-ocean-mountains", "name": "ecosystems, ocean & mountains", "description": "生態系、森林、海洋、山岳、自然を活用した解決策など。"},
    {"id": "inclusion-non-party", "name": "inclusion & non-party stakeholders", "description": "ジェンダー、先住民族、若者、地域社会、非国家主体など。"},
    {"id": "global-stocktake-ambition", "name": "global stocktake & ambition cycle", "description": "GST、NDC、LTS、野心向上サイクルなど。"},
    {"id": "article-6-carbon-markets", "name": "article 6 & carbon markets", "description": "パリ協定6条、市場・非市場メカニズム、国際協力など。"},
    {"id": "governance-process", "name": "governance & process", "description": "会合運営、制度、手続き、マンデート、報告・検討プロセスなど。"},
]


def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def extract_pdf(name: str, raw: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(raw).hexdigest()
    version_id = digest[:12]
    doc = fitz.open(stream=raw, filetype="pdf")
    page_count = len(doc)
    lines: list[tuple[int, str]] = []
    for page_no, page in enumerate(doc, 1):
        for raw_line in page.get_text("text", sort=True).splitlines():
            text = clean_line(raw_line)
            if not text or text.upper() in STOP_LINES or re.fullmatch(r"\d{1,3}", text):
                continue
            lines.append((page_no, text))
    markers = [i for i, (_, line) in enumerate(lines) if NUM_LINE.match(line)]
    paragraphs: list[dict[str, Any]] = []
    if len(markers) >= 2:
        chapter = ""
        subsection = ""

        def accept_heading(line: str) -> bool:
            nonlocal chapter, subsection
            major = CHAPTER_LINE.match(line)
            if major:
                title = re.sub(r"(?<=[A-Za-z])\d+$", "", major.group(2)).strip()
                chapter, subsection = f"{major.group(1).upper()}. {title}", ""
                return True
            minor = SUBSECTION_LINE.match(line)
            if minor:
                subsection = f"{minor.group(1)}. {minor.group(2)}"
                return True
            return False

        for _, line in lines[:markers[0]]:
            accept_heading(line)
        for marker_no, start in enumerate(markers):
            end = markers[marker_no + 1] if marker_no + 1 < len(markers) else len(lines)
            page_no, first = lines[start]
            marker = NUM_LINE.match(first)
            if not marker:
                continue
            number = marker.group(1)
            body_lines = [marker.group(2) or ""] if marker.group(2) else []
            last_page = page_no
            found_heading = False
            for pno, line in lines[start + 1:end]:
                if re.match(r"^\d{1,3}\s+The present document is being issued", line, re.I):
                    continue
                if accept_heading(line):
                    found_heading = True
                    continue
                if found_heading:
                    continue
                body_lines.append(line)
                last_page = pno
            body = re.sub(r"\s+", " ", " ".join(body_lines)).strip()
            if len(body) >= 20:
                paragraphs.append({"id": f"{version_id}-p{len(paragraphs)+1}", "number": number,
                                   "page": page_no, "page_end": last_page, "section": " / ".join(x for x in (chapter, subsection) if x),
                                   "text": body, "memo": ""})
    else:
        for page_no, page in enumerate(doc, 1):
            for block in page.get_text("blocks", sort=True):
                text = re.sub(r"\s+", " ", str(block[4])).strip()
                if len(text) >= 25 and text.upper() not in STOP_LINES:
                    paragraphs.append({"id": f"{version_id}-p{len(paragraphs)+1}", "number": None,
                                       "page": page_no, "page_end": page_no, "section": "", "text": text, "memo": ""})
    doc.close()
    if not paragraphs:
        raise ValueError("PDFから本文を抽出できませんでした。スキャン画像PDFの場合はOCRが必要です。")
    return {"id": version_id, "name": Path(name).stem, "filename": Path(name).name,
            "pages": page_count, "sha256": digest, "paragraphs": paragraphs, "pdf_bytes": raw}


def all_paragraphs(versions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for version in versions for p in version["paragraphs"]]


def automatic_topics(versions: list[dict[str, Any]], n_topics: int, membership_threshold: float = .18,
                     renames: dict[str, str] | None = None) -> tuple[dict[str, str], dict[str, list[str]], dict[str, dict[str, float]], list[dict[str, Any]]]:
    paras = all_paragraphs(versions)
    if not paras:
        return {}, {}, {}, []
    texts = [p["text"] for p in paras]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=30000, sublinear_tf=True)
    matrix = vec.fit_transform(texts)
    count = min(max(2, n_topics), len(paras), max(1, matrix.shape[1]))
    renames = renames or {}
    if count < 2 or matrix.shape[1] < 2:
        labels = [renames.get("auto-0", "全般")]
        weights = np.ones((len(paras), 1))
        features = np.array([])
        components = np.empty((1, 0))
    else:
        model = NMF(n_components=count, init="nndsvda", random_state=17, max_iter=350)
        weights = model.fit_transform(matrix)
        features = np.array(vec.get_feature_names_out())
        components = model.components_
        labels = []
        for index, component in enumerate(components):
            top = [str(features[i]).strip() for i in component.argsort()[::-1][:5] if str(features[i]).strip()]
            labels.append(renames.get(f"auto-{index}", f"トピック{index+1}: " + "・".join(top[:3])))
    totals = weights.sum(axis=1, keepdims=True)
    probabilities = weights / np.maximum(totals, 1e-12)
    membership: dict[str, list[str]] = {}
    primary: dict[str, str] = {}
    scores: dict[str, dict[str, float]] = {}
    counts = defaultdict(int)
    for row, para in zip(probabilities, paras):
        idx = int(np.argmax(row))
        selected_indices = [j for j in range(len(labels)) if row[j] >= membership_threshold]
        if idx not in selected_indices: selected_indices.append(idx)
        selected_indices = sorted(selected_indices, key=lambda j: row[j], reverse=True)[:4]
        selected = [labels[j] for j in selected_indices]
        membership[para["id"]] = selected
        primary[para["id"]] = labels[idx]
        scores[para["id"]] = {label: float(row[j]) for j, label in enumerate(labels)}
        for label in selected: counts[label] += 1
    summary = []
    for idx, label in enumerate(labels):
        keywords = [str(features[i]) for i in components[idx].argsort()[::-1][:8]] if len(features) else []
        summary.append({"id": f"auto-{idx}", "label": label, "keywords": keywords, "count": counts[label]})
    return primary, membership, scores, summary


def supervised_topics(versions: list[dict[str, Any]], custom_topics: list[dict[str, str]],
                      training_data: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, list[str]], dict[str, dict[str, float]], list[dict[str, Any]], dict[str, str]]:
    paras = all_paragraphs(versions)
    if not paras or not custom_topics:
        return {}, {}, {}, [], {}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=30000, sublinear_tf=True)
    matrix = vec.fit_transform([p["text"] for p in paras])
    names = {topic["id"]: topic["name"] for topic in custom_topics}
    train_texts = [sample["text"] for sample in training_data]
    train_matrix = vec.transform(train_texts) if train_texts else None
    scores = {p["id"]: {} for p in paras}
    quality: dict[str, str] = {}
    features = np.array(vec.get_feature_names_out())
    for topic_id, name in names.items():
        positive_ids = [i for i, sample in enumerate(training_data) if topic_id in sample.get("topics", [])]
        if not training_data:
            for p in paras: scores[p["id"]][name] = 0.0
            quality[name] = "未学習"
            continue
        y = np.array([1 if topic_id in sample.get("topics", []) else 0 for sample in training_data])
        if len(set(y.tolist())) == 2 and len(training_data) >= 4:
            clf = LogisticRegression(max_iter=400, class_weight="balanced", random_state=17)
            clf.fit(train_matrix, y)
            probs = clf.predict_proba(matrix)[:, list(clf.classes_).index(1)]
            quality[name] = "学習例からの推定"
        elif positive_ids:
            similarities = cosine_similarity(matrix, train_matrix[positive_ids]).max(axis=1)
            probs = np.clip(similarities, 0, 1)
            quality[name] = "例が少ないため類似度の参考値"
        else:
            probs = np.zeros(len(paras))
            quality[name] = "正例なし"
        for para, prob in zip(paras, probs): scores[para["id"]][name] = float(prob)
    train_by_id = {sample["paragraph_id"]: sample for sample in training_data}
    memberships: dict[str, list[str]] = {}
    primary: dict[str, str] = {}
    counts = defaultdict(int)
    for para in paras:
        sample = train_by_id.get(para["id"])
        selected = [names[x] for x in sample.get("topics", []) if x in names] if sample else []
        if not sample:
            selected = [name for name, prob in scores[para["id"]].items() if prob >= .5]
        memberships[para["id"]] = selected
        primary[para["id"]] = selected[0] if selected else "未分類"
        for name in selected: counts[name] += 1
        if sample:
            for topic_id, name in names.items(): scores[para["id"]][name] = 1.0 if topic_id in sample.get("topics", []) else 0.0
    summary = []
    for topic in custom_topics:
        name = topic["name"]
        topic_samples = [sample["text"] for sample in training_data if topic["id"] in sample.get("topics", [])]
        keywords: list[str] = []
        if topic_samples:
            vec_pos = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=2000)
            try:
                mat_pos = vec_pos.fit_transform(topic_samples)
                avg = np.asarray(mat_pos.mean(axis=0)).ravel()
                keywords = [str(vec_pos.get_feature_names_out()[i]) for i in avg.argsort()[::-1][:8]]
            except ValueError:
                pass
        summary.append({"id": topic["id"], "label": name, "keywords": keywords,
                        "count": counts[name], "training_examples": len(training_data),
                        "quality": quality.get(name, "未学習")})
    return primary, memberships, scores, summary, quality


def analyze_topics(versions: list[dict[str, Any]], mode: str, topic_count: int,
                   custom_topics: list[dict[str, str]], training_data: list[dict[str, Any]],
                   renames: dict[str, str], membership_threshold: float = .18) -> tuple[dict[str, str], dict[str, list[str]], dict[str, dict[str, float]], list[dict[str, Any]], dict[str, str]]:
    if mode == "user":
        return supervised_topics(versions, custom_topics, training_data)
    primary, membership, scores, summary = automatic_topics(versions, topic_count, membership_threshold, renames)
    quality = {topic["label"]: "自動発見モデルの推定値" for topic in summary}
    return primary, membership, scores, summary, quality


def paragraph_topics(paragraph_id: str, membership: dict[str, list[str]], primary: dict[str, str]) -> list[str]:
    return membership.get(paragraph_id, [primary[paragraph_id]] if paragraph_id in primary else [])


def topic_score_text(paragraph_id: str, scores: dict[str, dict[str, float]], limit: int = 6) -> str:
    ranked = sorted(scores.get(paragraph_id, {}).items(), key=lambda item: item[1], reverse=True)
    return " · ".join(f"{name} {probability:.0%}" for name, probability in ranked[:limit])


def make_multi_alignment(base: dict[str, Any], targets: list[dict[str, Any]], threshold: float,
                         overrides: dict[str, str], placements: dict[str, dict[str, str]],
                         manual_order: dict[str, int], topic_membership: dict[str, list[str]],
                         topic_primary: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    left = base["paragraphs"]
    if not left:
        return [], {}
    rows_by_base: dict[str, dict[str, list[dict[str, Any]]]] = {p["id"]: {t["id"]: [] for t in targets} for p in left}
    add_before: dict[str, list[dict[str, Any]]] = defaultdict(list)
    add_after: dict[str, list[dict[str, Any]]] = defaultdict(list)
    add_start: list[dict[str, Any]] = []
    links: dict[str, list[str]] = defaultdict(list)
    left_index = {p["id"]: i for i, p in enumerate(left)}
    base_texts = [p["text"] for p in left]
    base_topic_sets = {p["id"]: set(paragraph_topics(p["id"], topic_membership, topic_primary)) for p in left}

    for target in targets:
        raw_right = target["paragraphs"]
        indexed = list(enumerate(raw_right))
        indexed.sort(key=lambda pair: (manual_order.get(pair[1]["id"], pair[0] + 1), pair[0]))
        right = [p for _, p in indexed]
        texts = base_texts + [p["text"] for p in right]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=40000, sublinear_tf=True)
        matrix = vec.fit_transform(texts)
        sim = (matrix[len(left):] @ matrix[:len(left)].T).toarray()
        previous_mapped = None
        mapped_by_target: dict[str, str] = {}
        for j, p in enumerate(right):
            forced = overrides.get(p["id"])
            mapped_id = forced if forced and forced != "__ADD__" and forced in left_index else None
            best_i = int(np.argmax(sim[j]))
            best_score = float(sim[j, best_i])
            if forced is None and best_score >= threshold:
                mapped_id = left[best_i]["id"]
            if mapped_id:
                base_para = left[left_index[mapped_id]]
                q = dict(p)
                q["moved"] = previous_mapped is not None and left_index[mapped_id] < previous_mapped
                q["merged_from"] = [left[k].get("number") or k + 1 for k, score in enumerate(sim[j])
                                     if left[k]["id"] != mapped_id and score >= threshold and score >= best_score * .62]
                q["alignment_score"] = best_score
                rows_by_base[mapped_id][target["id"]].append(q)
                links[mapped_id].append(p["id"])
                links[p["id"]].append(mapped_id)
                mapped_by_target[p["id"]] = mapped_id
                previous_mapped = left_index[mapped_id]
                continue

            # Low text similarity means a new row. Topic similarity only chooses its location; it never turns it into a match.
            explicit = placements.get(p["id"], {})
            anchor_id = explicit.get("anchor_id")
            position = explicit.get("position", "after")
            if anchor_id not in left_index:
                p_topics = set(paragraph_topics(p["id"], topic_membership, topic_primary))
                topical = [i for i, base_para in enumerate(left) if p_topics & base_topic_sets[base_para["id"]]]
                if topical:
                    anchor_index = max(topical, key=lambda i: sim[j, i])
                    anchor_id = left[anchor_index]["id"]
                    position = "after"
                else:
                    prior_id = next((mapped_by_target[right[k]["id"]] for k in range(j - 1, -1, -1)
                                     if right[k]["id"] in mapped_by_target), None)
                    later_id = next((mapped_by_target[right[k]["id"]] for k in range(j + 1, len(right))
                                     if right[k]["id"] in mapped_by_target), None)
                    if prior_id:
                        anchor_id, position = prior_id, "after"
                    elif later_id:
                        anchor_id, position = later_id, "before"
                    else:
                        anchor_id, position = left[0]["id"], "before"
            addition_row = {"left": None, "right": {t["id"]: [] for t in targets}, "kind": "added", "anchor_id": anchor_id,
                            "manual_position": bool(explicit)}
            q = dict(p)
            q["moved"] = False
            q["merged_from"] = []
            q["alignment_score"] = best_score
            addition_row["right"][target["id"]] = [q]
            if anchor_id in left_index:
                (add_before if position == "before" else add_after)[anchor_id].append(addition_row)
            else:
                add_start.append(addition_row)

    rows: list[dict[str, Any]] = []
    for row in add_start:
        rows.append(row)
    for base_para in left:
        rows.extend(add_before.get(base_para["id"], []))
        rights = rows_by_base[base_para["id"]]
        for target in targets:
            rights[target["id"]].sort(key=lambda p: (manual_order.get(p["id"], next((i+1 for i, x in enumerate(target["paragraphs"]) if x["id"] == p["id"]), 10**9)), p.get("number") or ""))
        rows.append({"left": base_para, "right": rights,
                     "kind": "matched" if any(rights[t["id"]] for t in targets) else "deleted", "anchor_id": base_para["id"]})
        rows.extend(add_after.get(base_para["id"], []))
    return rows, links


def format_enumerated_clauses(text: str) -> str:
    """Put lower-case lettered subparagraph labels on their own display lines."""
    return re.sub(r"[ \t]+(?=\([a-z]\)(?:[ \t]|$))", "\n", text, flags=re.IGNORECASE)


def highlight_fragment(text: str, query: str = "", manual_phrases: list[str] | None = None) -> str:
    spans: list[tuple[int, int, str]] = []
    phrases = [(format_enumerated_clauses(query.strip()), "search-hit")] if query.strip() else []
    phrases += [(format_enumerated_clauses(phrase), "manual-highlight") for phrase in (manual_phrases or []) if phrase]
    for phrase, style in phrases:
        for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
            spans.append((match.start(), match.end(), style))
    if not spans: return html.escape(text)
    boundaries = sorted({0, len(text), *(point for start, end, _ in spans for point in (start, end))})
    output = []
    for start, end in zip(boundaries, boundaries[1:]):
        selected = [style for left, right, style in spans if left <= start and right >= end]
        chunk = html.escape(text[start:end])
        if "search-hit" in selected: chunk = f'<mark class="search-hit">{chunk}</mark>'
        elif "manual-highlight" in selected: chunk = f'<mark class="manual-highlight">{chunk}</mark>'
        output.append(chunk)
    return "".join(output)


def token_diff(a: str, b: str, search: str = "", a_highlights: list[str] | None = None,
               b_highlights: list[str] | None = None) -> tuple[str, str]:
    a = format_enumerated_clauses(a)
    b = format_enumerated_clauses(b)
    ta = re.findall(r"\s+|[\w]+|[^\w\s]", a, flags=re.UNICODE)
    tb = re.findall(r"\s+|[\w]+|[^\w\s]", b, flags=re.UNICODE)
    sm = SequenceMatcher(a=ta, b=tb, autojunk=True)
    left_html: list[str] = []
    right_html: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            left_html.append(highlight_fragment("".join(ta[i1:i2]), search, a_highlights))
            right_html.append(highlight_fragment("".join(tb[j1:j2]), search, b_highlights))
        elif tag in ("delete", "replace"):
            left_html.append('<mark class="diff-del">' + highlight_fragment("".join(ta[i1:i2]), search, a_highlights) + "</mark>")
            if tag == "replace":
                right_html.append('<mark class="diff-add">' + highlight_fragment("".join(tb[j1:j2]), search, b_highlights) + "</mark>")
        elif tag == "insert":
            right_html.append('<mark class="diff-add">' + highlight_fragment("".join(tb[j1:j2]), search, b_highlights) + "</mark>")
    return "".join(left_html), "".join(right_html)


def page_ref(p: dict[str, Any]) -> str:
    end = p.get("page_end", p["page"])
    return f"p.{p['page']}" if end == p["page"] else f"pp.{p['page']}–{end}"


def bundled_pdf_name(version: dict[str, Any]) -> str:
    return f"{version['id']}_{Path(version['filename']).name}"


def render_cell(version: dict[str, Any], paragraphs: list[dict[str, Any]], other: str,
                topic_primary: dict[str, str], topic_membership: dict[str, list[str]], topic_scores: dict[str, dict[str, float]],
                kind: str, memo_visible: bool, baseline: bool = False) -> None:
    if not paragraphs:
        label = "基準版に対応する文言なし" if kind == "added" and baseline else ("この版に対応文言なし" if kind != "added" else "")
        if label:
            st.markdown(f'<div class="empty-panel">{label}</div>', unsafe_allow_html=True)
        return
    for p in paragraphs:
        text_a, text_b = token_diff(p["text"], other if not baseline else other)
        rendered = text_a if baseline else text_b
        badges = []
        if kind == "added" and not baseline: badges.append('<span class="badge b-add">新設</span>')
        if p.get("moved"): badges.append('<span class="badge b-move">移動</span>')
        if p.get("merged_from"): badges.append('<span class="badge b-change">統合候補</span>')
        labels = paragraph_topics(p["id"], topic_membership, topic_primary)
        confidence = topic_scores.get(p["id"], {})
        topics = " · ".join(f"{label} {confidence.get(label, 0):.0%}" if label in confidence else label for label in labels)
        with st.container(border=True):
            st.caption(f"{version['name']} · Para. {p.get('number') or '—'} · {page_ref(p)}" + (f" · {topics}" if topics else ""))
            if badges: st.markdown(" ".join(badges), unsafe_allow_html=True)
            st.markdown(f'<div class="en">{rendered}</div>', unsafe_allow_html=True)
            if memo_visible:
                st.markdown(f'<div class="memo"><span class="memo-label">メモ</span><br>{html.escape(p.get("memo", "") or "（未入力）")}</div>', unsafe_allow_html=True)


def make_html_site(base: dict[str, Any], targets: list[dict[str, Any]], rows: list[dict[str, Any]],
                   topic_primary: dict[str, str], topic_membership: dict[str, list[str]],
                   topic_scores: dict[str, dict[str, float]], watches: list[dict[str, Any]],
                   manual_highlights: dict[str, list[str]]) -> str:
    columns = [base] + targets
    col_widths = " ".join(["minmax(290px,1fr)"] * len(columns))
    header = "".join(f'<div class="head" data-version="{html.escape(v["id"])}">{html.escape(v["name"])}</div>' for v in columns)
    version_toggles = "".join(
        f'<label><input type="checkbox" checked data-version-toggle="{html.escape(v["id"])}">{html.escape(v["name"])}</label>'
        for v in columns)
    rendered_rows = []
    watch_map = {(w.get("version_id"), w.get("paragraph_id")): w for w in watches}
    for row in rows:
        lp = row["left"]
        cells = []
        for version in columns:
            is_base = version["id"] == base["id"]
            paras = [lp] if is_base and lp else ([] if is_base else row["right"].get(version["id"], []))
            if not paras:
                cell = '<div class="empty">' + ("基準版の文言なし" if row["kind"] == "added" and is_base else ("追加なし" if row["kind"] == "added" else "削除／対応なし")) + "</div>"
            else:
                pieces = []
                for p in paras:
                    first_target = targets[0] if targets else None
                    other = (" ".join(x["text"] for x in row["right"].get(first_target["id"], []))
                             if is_base and first_target else (lp["text"] if lp else ""))
                    left_diff, right_diff = token_diff(
                        p["text"], other,
                        a_highlights=manual_highlights.get(p["id"], []) if is_base else (manual_highlights.get(lp["id"], []) if lp else []),
                        b_highlights=manual_highlights.get(p["id"], []) if not is_base else [])
                    display = left_diff if is_base else right_diff
                    labels = paragraph_topics(p["id"], topic_membership, topic_primary)
                    badges = []
                    if row["kind"] == "added" and not is_base: badges.append('<span class="badge add">新設</span>')
                    if p.get("moved"): badges.append('<span class="badge move">移動</span>')
                    if p.get("merged_from"): badges.append('<span class="badge">統合候補</span>')
                    if (version["id"], p["id"]) in watch_map: badges.append('<span class="badge watch">Watch</span>')
                    badge_html = " ".join(badges)
                    topic_html = " ".join(f'<span class="badge topic">{html.escape(name)} {topic_scores.get(p["id"], {}).get(name, 0):.0%}</span>'
                                           for name in labels)
                    section_html = f'<span class="section-meta"> · 章立て: {html.escape(p.get("section") or "見出し未検出")}</span>'
                    memo_text = p.get("memo", "") or ""
                    memo_html = f'<div class="memo"><b>メモ</b><br>{html.escape(memo_text)}</div>' if memo_text.strip() else ""
                    pieces.append(f'<article class="panel"><div class="meta">{page_ref(p)}{section_html} {topic_html} {badge_html}</div><div class="en"><span class="para-number">{html.escape(str(p.get("number") or "—"))}.</span> {display}</div>{memo_html}</article>')
                cell = "<br>".join(pieces)
            cells.append(f'<div data-version="{html.escape(version["id"])}">{cell}</div>')
        rendered_rows.append(f'<section class="row {"addrow" if row["kind"] == "added" else ""}" data-row>{"".join(cells)}</section>')
    watch_cards = []
    version_by_id = {v["id"]: v for v in columns}
    for w in watches:
        version = version_by_id.get(w.get("version_id"))
        ref = f'<a href="pdfs/{html.escape(bundled_pdf_name(version))}#page={w["page"]}">元PDF p.{w["page"]}</a>' if version else ""
        ref_html = f" · {ref}" if ref else ""
        watch_cards.append(f'<article class="panel"><b>{html.escape(w.get("priority", "通常"))} · {html.escape(w.get("version_name", ""))}</b>{ref_html}<blockquote>{html.escape(w.get("quote", ""))}</blockquote>{html.escape(w.get("note", ""))}</article>')
    topic_options = sorted({label for labels in topic_membership.values() for label in labels})
    select_topics = "".join(f'<option>{html.escape(t)}</option>' for t in topic_options)
    grid_style = f"grid-template-columns:{col_widths}"
    return f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(base['name'])} 比較</title><style>
:root{{--ink:#182721;--muted:#63716a;--line:#d9e2dd;--green:#176b4e;--greenbg:#e2f4e9;--red:#a03c35;--redbg:#fde9e6;--blue:#245f8c;--bluebg:#e6f1fb}}*{{box-sizing:border-box}}body{{margin:0;background:#f5f7f5;color:var(--ink);font:15px/1.58 system-ui,-apple-system,'Yu Gothic',sans-serif}}header{{background:#103d30;color:#fff;padding:25px 22px}}main{{padding:18px}}.notice{{padding:10px;border-left:4px solid #edc95e;background:#fff8e7;margin:10px 0}}.tools{{position:sticky;top:0;background:#f5f7f5ee;padding:10px;display:flex;gap:8px;z-index:2;flex-wrap:wrap}}input,select{{padding:9px;border:1px solid #cbd7d1;border-radius:8px;background:white}}input{{flex:1}}input[type=checkbox]{{flex:0}}.tools label{{display:inline-flex;align-items:center;gap:5px}}.scroll{{overflow-x:auto}}.grid{{display:grid;{grid_style};gap:10px;min-width:max-content}}.head,.panel{{background:white;border:1px solid var(--line);border-radius:10px;padding:12px;width:100%;min-width:280px}}.row{{display:grid;{grid_style};gap:10px;margin:10px 0;align-items:stretch}}.meta{{font-size:12px;color:var(--muted)}}.para-number{{font-weight:700;color:#43564c}}.section-meta{{display:none}}.sections-on .section-meta{{display:inline}}.en{{font:15px/1.6 Georgia,serif;white-space:pre-wrap}}.memo{{border-top:1px solid #eee;margin-top:10px;padding-top:8px;white-space:pre-wrap}}.empty{{border:1px dashed var(--line);padding:22px;color:var(--muted);min-width:280px}}.badge{{font-size:11px;border-radius:30px;padding:2px 7px;background:#edf1ef}}.add{{background:var(--greenbg)}}.move{{background:var(--bluebg)}}mark.diff-add{{background:var(--greenbg)}}mark.diff-del{{background:var(--redbg);text-decoration:line-through}}footer{{padding:16px;background:#eaf0ec;margin-top:20px}}@media(max-width:700px){{header{{padding:18px}}main{{padding:10px}}}}
</style><style>.search-hit{{background:#ffb347!important;color:#4b2a00}}.manual-highlight{{background:#fff176;color:#3f3500}}</style><body><header><h1>{html.escape(base['name'])} を基準にした複数版比較</h1><div>比較文書 {len(targets)}版 · 表示列は切り替えできます</div></header><main><div class="notice">「メモ」は手入力またはCSVで登録した内容です。分類確率はローカルモデルによる推定値です。</div><div class="tools"><input id="q" placeholder="本文・メモ・段落番号を検索"><select id="topic"><option value="">すべてのトピック</option>{select_topics}</select><label><input id="sections" type="checkbox">章立てを表示</label>{version_toggles}</div><div class="scroll"><div class="grid">{header}</div>{''.join(rendered_rows)}</div><h2>Watch付箋</h2>{''.join(watch_cards) if watch_cards else '<p>付箋はありません。</p>'}<footer>PDFページリンクを使うには配布ZIPを展開し、同梱の pdfs フォルダを保ってください。自動対応付けは候補です。</footer></main><script>const q=document.querySelector('#q'),t=document.querySelector('#topic'),s=document.querySelector('#sections'),vt=[...document.querySelectorAll('[data-version-toggle]')];function f(){{for(const r of document.querySelectorAll('[data-row]')){{const x=[...r.querySelectorAll('[data-version]')].filter(c=>c.style.display!=='none').map(c=>c.textContent).join(' ').toLowerCase();r.style.display=(!q.value||x.includes(q.value.toLowerCase()))&&(!t.value||x.includes(t.value.toLowerCase()))?'grid':'none'}}}}function versions(){{let on=vt.filter(c=>c.checked);if(!on.length){{vt[0].checked=true;on=[vt[0]]}}const visible=new Set(on.map(c=>c.dataset.versionToggle));for(const cell of document.querySelectorAll('[data-version]'))cell.style.display=visible.has(cell.dataset.version)?'':'none';const template=`repeat(${{on.length}}, minmax(290px, 1fr))`;for(const grid of document.querySelectorAll('.grid,.row'))grid.style.gridTemplateColumns=template;f()}}q.oninput=f;t.onchange=f;s.onchange=()=>document.body.classList.toggle('sections-on',s.checked);vt.forEach(c=>c.onchange=versions);versions();</script><script>function applySearchHighlights(){{document.querySelectorAll('mark.search-hit').forEach(m=>m.replaceWith(document.createTextNode(m.textContent)));const needle=q.value.trim().toLowerCase();if(!needle)return;for(const el of document.querySelectorAll('[data-row] [data-version]:not([style*=\"display: none\"]) .en')){{const walker=document.createTreeWalker(el,NodeFilter.SHOW_TEXT),nodes=[];while(walker.nextNode())if(!walker.currentNode.parentElement.closest(".para-number"))nodes.push(walker.currentNode);for(const node of nodes){{const text=node.nodeValue,lower=text.toLowerCase(),frag=document.createDocumentFragment();let pos=0,index;while((index=lower.indexOf(needle,pos))!==-1){{frag.append(document.createTextNode(text.slice(pos,index)));const mark=document.createElement('mark');mark.className='search-hit';mark.textContent=text.slice(index,index+needle.length);frag.append(mark);pos=index+needle.length}}if(pos){{frag.append(document.createTextNode(text.slice(pos)));node.replaceWith(frag)}}}}}}q.addEventListener('input',applySearchHighlights);document.querySelectorAll('[data-version-toggle]').forEach(c=>c.addEventListener('change',()=>setTimeout(applySearchHighlights,0)));</script></body></html>'''


def project_object() -> dict[str, Any]:
    keys = ["base_id", "target_ids", "visible_version_ids", "_visible_scope_ids", "compare_search", "compare_topic_filter", "hide_memo", "edit_memo_inline", "manual_tag_mode", "manual_highlight_mode", "manual_highlights", "show_watch_inline", "show_watch_toggle", "show_changes", "show_sections",
            "active_topic_mode", "topic_count", "topic_membership_threshold", "threshold", "topic_primary", "topic_membership", "topic_scores",
            "topic_summary", "topic_renames", "custom_topics", "training_data", "manual_topic_overrides", "alignment_overrides",
            "placement_overrides", "manual_order", "watch", "memo_version", "memo_paragraph",
            "base_select", "target_select", "alignment_target", "placement_edit_pid", "placement_link",
            "placement_anchor", "placement_position", "placement_display_order", "topic_correct_pid",
            "topic_correct_labels", "watch_version", "watch_paragraph", "watch_quotes", "watch_quote_override",
            "watch_note", "watch_tags", "watch_priority", "memo_text", "jump_para_number", "topic_quality", "active_view",
            "autosave_enabled"]
    ui_state = {key: st.session_state.get(key) for key in keys}
    clean_versions = []
    for version in st.session_state.versions:
        clean_versions.append({k: version[k] for k in ("id", "name", "filename", "pages", "sha256", "paragraphs")})
    return {"format": FORMAT, "saved_at": datetime.now(timezone.utc).isoformat(),
            "versions": clean_versions, "ui_state": ui_state}


def project_bundle() -> bytes:
    obj = project_object()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))
        for version in st.session_state.versions:
            raw = version.get("pdf_bytes")
            if raw:
                archive.writestr(f"pdfs/{bundled_pdf_name(version)}", raw)
    return buf.getvalue()


def project_content_fingerprint() -> str:
    project = project_object()
    state = project["ui_state"]
    content = {
        "versions": project["versions"],
        "selection": {key: state.get(key) for key in ("base_id", "target_ids")},
        "classification": {key: state.get(key) for key in (
            "active_topic_mode", "topic_count", "topic_membership_threshold", "threshold",
            "topic_primary", "topic_membership", "topic_scores", "topic_summary", "topic_renames",
            "custom_topics", "training_data", "manual_topic_overrides")},
        "editing": {key: state.get(key) for key in (
            "alignment_overrides", "placement_overrides", "manual_order", "watch", "manual_highlights")},
    }
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_project_download_button(label: str, key: str) -> None:
    if st.download_button(label, project_bundle(), "pdf_compare_project.zip", "application/zip", key=key,
                          type="primary", width="stretch"):
        st.session_state.last_saved_content_fingerprint = project_content_fingerprint()
        st.toast("プロジェクトZIPをダウンロードしました。")


def restore_project(raw: bytes) -> None:
    widget_keys = ["base_select", "target_select", "visible_version_ids", "compare_search", "compare_topic_filter", "hide_memo", "edit_memo_inline", "manual_tag_mode", "manual_highlight_mode", "show_watch_inline", "show_watch_toggle", "show_changes", "show_sections", "jump_para_number", "topic_membership_threshold",
                   "active_view", "alignment_target", "placement_edit_pid", "placement_link", "placement_anchor",
                   "placement_position", "placement_display_order", "topic_correct_pid", "topic_correct_labels",
                   "watch_version", "watch_paragraph", "watch_quotes", "watch_quote_override", "watch_note",
                   "watch_tags", "watch_priority", "memo_version", "memo_paragraph", "memo_text"]
    for key in list(st.session_state.keys()):
        if key in widget_keys or key.startswith(("version_name_", "rename_", "custom_topic_", "custom_topic_desc_", "memo_inline_", "manual_tags_", "highlight_draft_")):
            st.session_state.pop(key, None)
    pdfs: dict[str, bytes] = {}
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            project = json.loads(archive.read("project.json"))
            for name in archive.namelist():
                if name.startswith("pdfs/") and not name.endswith("/"):
                    pdfs[Path(name).name] = archive.read(name)
    else:
        project = json.loads(raw)
    if project.get("format") not in {FORMAT, "pdf-compare-project-v1"}:
        raise ValueError("未対応のプロジェクト形式です")
    versions = project.get("versions", [])
    for version in versions:
        for para in version.get("paragraphs", []):
            if "memo" not in para:
                para["memo"] = para.pop("translation", "")
        version["pdf_bytes"] = pdfs.get(bundled_pdf_name(version))
        if version["pdf_bytes"] and any("section" not in para for para in version.get("paragraphs", [])):
            try:
                extracted = extract_pdf(version["filename"], version["pdf_bytes"])
                section_by_id = {para["id"]: para.get("section", "") for para in extracted["paragraphs"]}
                for para in version.get("paragraphs", []):
                    para.setdefault("section", section_by_id.get(para["id"], ""))
            except Exception:
                for para in version.get("paragraphs", []): para.setdefault("section", "")
        else:
            for para in version.get("paragraphs", []): para.setdefault("section", "")
    state = project.get("ui_state", {})
    if not state:
        state = {"topic_count": project.get("topic_count", 6), "threshold": project.get("match_threshold", .20),
                 "watch": project.get("watches", []), "alignment_overrides": project.get("alignment_overrides", {}),
                 "topic_primary": project.get("topic_map", {}), "topic_summary": project.get("topic_summary", [])}
    defaults = {"versions": versions, "base_id": versions[0]["id"] if versions else "",
                "base_select": versions[0]["id"] if versions else "", "target_ids": [], "target_select": [],
                "visible_version_ids": [], "_visible_scope_ids": [],
                "topic_primary": {}, "topic_membership": {}, "topic_scores": {}, "topic_summary": [],
                "topic_renames": {}, "custom_topics": [], "training_data": [], "alignment_overrides": {},
                "placement_overrides": {}, "manual_order": {}, "watch": [], "topic_count": 6,
                "manual_topic_overrides": {}, "manual_highlights": {},
                "topic_membership_threshold": .18, "threshold": .20, "active_topic_mode": "automatic", "autosave_enabled": True,
                "topic_quality": {}, "compare_search": "", "compare_topic_filter": "", "hide_memo": False,
                "show_changes": False, "show_sections": False, "show_watch_inline": False, "show_watch_toggle": True, "watch_quotes": [], "jump_para_number": "", "active_view": "比較"}
    for key, value in defaults.items(): st.session_state[key] = state.get(key, value)
    for key in ("visible_version_ids", "_visible_scope_ids", "compare_search", "compare_topic_filter", "hide_memo", "edit_memo_inline", "manual_tag_mode", "manual_highlight_mode", "manual_highlights", "show_watch_inline", "show_watch_toggle", "show_changes", "show_sections", "jump_para_number", "topic_membership_threshold", "memo_version",
                "memo_paragraph", "base_select", "target_select", "alignment_target", "placement_edit_pid",
                "placement_link", "placement_anchor", "placement_position", "placement_display_order",
                "topic_correct_pid", "topic_correct_labels", "watch_version", "watch_paragraph", "watch_quotes",
                "watch_quote_override", "watch_note", "watch_tags", "watch_priority", "memo_text", "active_view"):
        if key in state: st.session_state[key] = state[key]
    if not isinstance(st.session_state.get("watch_quotes"), list): st.session_state.watch_quotes = []
    st.session_state.versions = versions
    st.session_state.project_restored = True


def csv_memos(versions: list[dict[str, Any]], blank: bool = False) -> bytes:
    output = io.StringIO(newline="")
    fields = ["version_id", "version_name", "paragraph_id", "paragraph_number", "page", "english", "memo"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for version in versions:
        for para in version["paragraphs"]:
            writer.writerow({"version_id": version["id"], "version_name": version["name"],
                             "paragraph_id": para["id"], "paragraph_number": para.get("number") or "",
                             "page": para["page"], "english": para["text"],
                             "memo": "" if blank else para.get("memo", "")})
    return output.getvalue().encode("utf-8-sig")


def load_memo_csv(upload: Any, versions: list[dict[str, Any]]) -> tuple[int, list[str]]:
    table = csv.DictReader(io.StringIO(upload.getvalue().decode("utf-8-sig")))
    by_id = {p["id"]: p for p in all_paragraphs(versions)}
    updated, errors = 0, []
    for row in table:
        pid = (row.get("paragraph_id") or "").strip()
        if pid not in by_id:
            errors.append(f"段落IDが見つかりません: {pid or '空欄'}")
            continue
        memo = (row.get("memo") or row.get("japanese") or row.get("translation") or "").strip()
        by_id[pid]["memo"] = memo
        inline_key = f"memo_inline_{pid}"
        if inline_key in st.session_state: st.session_state[inline_key] = memo
        updated += 1
    return updated, errors[:12]


def recalculate_topics() -> None:
    primary, membership, scores, summary, quality = analyze_topics(
        st.session_state.versions, st.session_state.active_topic_mode, int(st.session_state.topic_count),
        st.session_state.custom_topics, st.session_state.training_data, st.session_state.topic_renames,
        float(st.session_state.topic_membership_threshold))
    id_to_label = {topic["id"]: topic["label"] for topic in summary}
    for paragraph_id, tag_ids in st.session_state.manual_topic_overrides.items():
        if paragraph_id not in membership: continue
        selected_labels = [id_to_label[tag_id] for tag_id in tag_ids if tag_id in id_to_label]
        if tag_ids and not selected_labels: continue
        membership[paragraph_id] = selected_labels
        primary[paragraph_id] = selected_labels[0] if selected_labels else "未分類"
    st.session_state.topic_primary = primary
    st.session_state.topic_membership = membership
    st.session_state.topic_scores = scores
    st.session_state.topic_summary = summary
    st.session_state.topic_quality = quality


def save_inline_memo(paragraph_id: str) -> None:
    memo = st.session_state.get(f"memo_inline_{paragraph_id}", "").strip()
    for version in st.session_state.versions:
        for paragraph in version["paragraphs"]:
            if paragraph["id"] == paragraph_id:
                paragraph["memo"] = memo
                return


def save_manual_tags(paragraph_id: str) -> None:
    selected_ids = list(st.session_state.get(f"manual_tags_{paragraph_id}", []))
    st.session_state.manual_topic_overrides[paragraph_id] = selected_ids
    if st.session_state.active_topic_mode == "user":
        for version in st.session_state.versions:
            paragraph = next((p for p in version["paragraphs"] if p["id"] == paragraph_id), None)
            if paragraph:
                sample = {"paragraph_id": paragraph_id, "version_id": version["id"],
                          "text": paragraph["text"], "topics": selected_ids,
                          "reviewed_at": datetime.now(timezone.utc).isoformat()}
                st.session_state.training_data = [s for s in st.session_state.training_data
                                                  if s.get("paragraph_id") != paragraph_id] + [sample]
                break
    recalculate_topics()


def watch_annotation_html(items: list[dict[str, Any]]) -> str:
    parts = []
    for item in items:
        raw_tags = item.get("tags") or []
        tags = ", ".join(map(str, raw_tags)) if isinstance(raw_tags, list) else str(raw_tags)
        note = str(item.get("note") or "")
        quote = str(item.get("quote") or "")
        detail = f'<br>{html.escape(note)}' if note else ""
        if tags: detail += f'<br><small>タグ: {html.escape(tags)}</small>'
        parts.append(f'<div class="watch-item"><b>Watch · {html.escape(str(item.get("priority") or "通常"))}</b><br>{html.escape(quote)}{detail}</div>')
    return "".join(parts)


def render_compare_paragraph(paragraph: dict[str, Any], display_text: str, version_id: str, version_name: str,
                             row_kind: str, is_base: bool, show_sections: bool, show_watch: bool, show_watch_toggle: bool,
                             watch_by_paragraph: dict[str, list[dict[str, Any]]], manual_tag_mode: bool,
                             tag_options: dict[str, str], manual_highlight_mode: bool,
                             hide_memo: bool, edit_memo: bool, jump_target: bool = False) -> None:
    pid = paragraph["id"]
    labels = paragraph_topics(pid, st.session_state.topic_membership, st.session_state.topic_primary)
    is_manual = pid in st.session_state.manual_topic_overrides
    tag_text = "、".join(tag if is_manual else f"{tag} {st.session_state.topic_scores.get(pid, {}).get(tag, 0):.0%}" for tag in labels) or "未分類"
    section_text = f" · 章立て: {paragraph.get('section') or '見出し未検出'}" if show_sections else ""
    if jump_target:
        st.markdown('<div id="para-jump-target"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.caption(f"{version_name} · {page_ref(paragraph)}{section_text} · タグ: {tag_text}{' · 手動' if is_manual else ''}")
        badges = []
        if row_kind == "added" and not is_base: badges.append('<span class="badge b-add">新設</span>')
        if paragraph.get("moved"): badges.append('<span class="badge b-move">移動</span>')
        if paragraph.get("merged_from"):
            badges.append('<span class="badge b-change">統合候補 · 基準段落 ' + ', '.join(map(str, paragraph["merged_from"])) + '</span>')
        if badges: st.markdown(" ".join(badges), unsafe_allow_html=True)
        number = html.escape(str(paragraph.get("number") or "—"))
        st.markdown(f'<div class="en"><span class="para-number">{number}.</span> {display_text}</div>', unsafe_allow_html=True)
        if show_watch_toggle:
            full_paragraph_watches = [
                (index, item) for index, item in enumerate(st.session_state.watch)
                if item.get("version_id") == version_id and item.get("paragraph_id") == pid
                and item.get("quote") == paragraph["text"]
            ]
            if full_paragraph_watches:
                if st.button("段落全体のWatchを解除", key=f"watch_toggle_{version_id}_{pid}"):
                    remove_ids = {index for index, _ in full_paragraph_watches}
                    st.session_state.watch = [item for index, item in enumerate(st.session_state.watch) if index not in remove_ids]
                    st.rerun()
            elif st.button("段落全体をWatchに追加", key=f"watch_toggle_{version_id}_{pid}"):
                st.session_state.watch.append({
                    "version_id": version_id, "version_name": version_name, "paragraph_id": pid,
                    "paragraph_number": paragraph.get("number"), "page": paragraph["page"],
                    "quote": paragraph["text"], "note": "", "tags": [], "priority": "通常",
                })
                st.rerun()
        if show_watch and watch_by_paragraph.get(pid):
            st.markdown(watch_annotation_html(watch_by_paragraph[pid]), unsafe_allow_html=True)
        if manual_highlight_mode:
            draft_key = f"highlight_draft_{pid}"
            st.text_input("蛍光ペンで強調する語句・文", key=draft_key, placeholder="英文からコピーした語句・文を入力")
            if st.button("ハイライトを保存", key=f"save_highlight_{pid}"):
                phrase = str(st.session_state.get(draft_key) or "").strip()
                if not phrase:
                    st.warning("強調する語句を入力してください。")
                elif phrase.casefold() not in paragraph["text"].casefold():
                    st.warning("入力した語句がこの段落の英文に見つかりません。")
                else:
                    saved = st.session_state.manual_highlights.setdefault(pid, [])
                    if phrase not in saved: saved.append(phrase)
                    st.rerun()
            saved_highlights = st.session_state.manual_highlights.get(pid, [])
            if saved_highlights:
                st.caption("保存済みの蛍光ペン")
                for index, phrase in enumerate(saved_highlights):
                    remove_cols = st.columns([5, 1])
                    remove_cols[0].write(phrase)
                    if remove_cols[1].button("解除", key=f"remove_highlight_{pid}_{index}"):
                        st.session_state.manual_highlights[pid] = [saved for saved in saved_highlights if saved != phrase]
                        st.rerun()
        if manual_tag_mode:
            if tag_options:
                tag_key = f"manual_tags_{pid}"
                if tag_key not in st.session_state:
                    if pid in st.session_state.manual_topic_overrides:
                        st.session_state[tag_key] = st.session_state.manual_topic_overrides[pid]
                    else:
                        label_to_id = {label: topic_id for topic_id, label in tag_options.items()}
                        st.session_state[tag_key] = [label_to_id[label] for label in labels if label in label_to_id]
                st.multiselect("タグを指定", list(tag_options), key=tag_key,
                               format_func=lambda tag_id: tag_options.get(tag_id, tag_id))
                if st.button("この段落のタグを保存", key=f"save_manual_tags_{pid}"):
                    save_manual_tags(pid)
                    st.rerun()
                if is_manual: st.caption("手動指定済み")
            else:
                st.caption("タグ候補がありません。先にトピック分類を実行してください。")
        if not hide_memo:
            if edit_memo:
                memo_key = f"memo_inline_{pid}"
                if memo_key not in st.session_state: st.session_state[memo_key] = paragraph.get("memo", "")
                st.text_area("メモ", height=100, key=memo_key, on_change=save_inline_memo,
                             args=(pid,), placeholder="この段落のメモを入力")
            else:
                st.markdown(f'<div class="memo"><b>メモ</b><br>{html.escape(paragraph.get("memo") or "（未入力）")}</div>', unsafe_allow_html=True)


def init_state() -> None:
    if st.session_state.pop("clear_on_next_run", False):
        bundled_already_loaded = st.session_state.get("_bundled_documents_loaded", False)
        training_already_loaded = st.session_state.get("_seed_training_loaded", False)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        if bundled_already_loaded:
            st.session_state._bundled_documents_loaded = True
        if training_already_loaded:
            st.session_state._seed_training_loaded = True
    defaults = {"versions": [], "topic_count": 6, "topic_membership_threshold": .18, "threshold": .20, "active_topic_mode": "automatic",
                "topic_primary": {}, "topic_membership": {}, "topic_scores": {}, "topic_summary": [],
                "topic_quality": {}, "topic_renames": {}, "custom_topics": [], "training_data": [],
                "manual_topic_overrides": {}, "manual_highlights": {}, "manual_tag_mode": False, "manual_highlight_mode": False, "show_watch_inline": False, "show_watch_toggle": True, "watch_quotes": [],
                "alignment_overrides": {}, "placement_overrides": {}, "manual_order": {}, "watch": [],
                "target_ids": [], "visible_version_ids": [], "_visible_scope_ids": [], "jump_para_number": "", "jump_para_pending": "",
                "autosave_enabled": False, "project_restored": False,
                "last_saved_content_fingerprint": None, "project_load_error": ""}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
    if not isinstance(st.session_state.get("watch_quotes"), list): st.session_state.watch_quotes = []
    if not st.session_state.get("_seed_training_loaded", False):
        st.session_state._seed_training_loaded = True
        if TRAINING_SEED_PATH.exists():
            try:
                seed = json.loads(TRAINING_SEED_PATH.read_text(encoding="utf-8"))
                st.session_state.custom_topics = seed.get("custom_topics", [])
                st.session_state.training_data = seed.get("training_data", [])
                st.session_state.manual_topic_overrides = seed.get("manual_topic_overrides", {})
                st.session_state.active_topic_mode = "user"
            except (OSError, ValueError) as exc:
                st.sidebar.warning(f"初期学習データを読み込めませんでした: {exc}")
    if not st.session_state.get("_bundled_documents_loaded", False):
        st.session_state._bundled_documents_loaded = True
        if not st.session_state.versions:
            bundled = []
            for pdf_path in sorted(DEFAULT_PDFS_DIR.glob("*.pdf")):
                try:
                    version = extract_pdf(pdf_path.name, pdf_path.read_bytes())
                    version["name"] = pdf_path.stem.split("_", 1)[1].replace("_", " ")
                    bundled.append(version)
                except Exception as exc:
                    st.sidebar.warning(f"同梱PDFを読み込めませんでした: {pdf_path.name}: {exc}")
            st.session_state.versions = bundled


init_state()
pending_project_bytes = st.session_state.pop("_pending_project_bytes", None)
if pending_project_bytes:
    try:
        restore_project(pending_project_bytes)
        st.session_state.last_saved_content_fingerprint = project_content_fingerprint()
        st.session_state.project_load_error = ""
    except Exception as exc:
        st.session_state.project_load_error = f"プロジェクトを読み込めませんでした: {exc}"
st.session_state.autosave_enabled = False
st.markdown(CSS, unsafe_allow_html=True)
st.title("PDF草案 比較サイト作成ツール")
st.caption("複数版のPDFを比較し、段落対応・トピック・Watch・メモを管理します。アップロードPDFはこの配備アプリの実行環境で処理します。")
st.markdown('<div class="app-note"><b>メモ:</b>手入力またはCSVで登録した内容を英文の下に表示します。</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 作業画面")
    active_view = st.radio(
        "作業画面", ["比較", "トピックと学習", "Watch付箋", "メモ", "出力・保存"],
        key="active_view", label_visibility="collapsed")

    if active_view == "比較":
        st.markdown("#### 段落を探す")
        jump_para_number = st.text_input("Para番号", key="jump_para_number", placeholder="例: 23")
        if st.button("一番左の版のParaへ移動", width="stretch"):
            if jump_para_number.strip():
                st.session_state.jump_para_pending = jump_para_number.strip()
            else:
                st.warning("段落番号を入力してください。")
        search = st.text_input("本文・メモ・段落番号を検索", key="compare_search")
        topic_options = [""] + sorted({label for labels in st.session_state.topic_membership.values() for label in labels})
        selected_topic = st.selectbox("トピック", topic_options, format_func=lambda x: x or "すべて", key="compare_topic_filter")
        with st.expander("表示・編集の切替", expanded=False):
            hide_memo = st.checkbox("メモを隠す", key="hide_memo")
            edit_memo = st.checkbox("メモを編集", key="edit_memo_inline")
            show_sections = st.checkbox("章立てを表示", key="show_sections")
            show_changes = st.checkbox("変更行のみ", key="show_changes")
            manual_tag_mode = st.checkbox("手動タグ指定", key="manual_tag_mode")
            show_watch = st.checkbox("Watchを表示", key="show_watch_inline")
            manual_highlight_mode = st.checkbox("蛍光ペン", key="manual_highlight_mode")
            show_watch_toggle = st.checkbox("Watch操作", key="show_watch_toggle")

    st.divider()
    with st.expander("プロジェクトを開く／保存", expanded=False):
        st.caption("保存したZIPを読み込むと、PDFと作業状態を復元します。")
        project_file = st.file_uploader("プロジェクトZIPまたはJSON", type=["zip", "json"], key="project_upload")
        if project_file and st.button("プロジェクトを読み込む", width="stretch"):
            st.session_state._pending_project_bytes = project_file.getvalue()
            st.rerun()
        if st.session_state.get("project_load_error"):
            st.error(st.session_state.project_load_error)
            st.session_state.project_load_error = ""
        if st.session_state.versions:
            render_project_download_button("プロジェクトZIPを保存", "project_save_sidebar")
            st.caption("ページを閉じる前に保存してください。未保存の編集があると、ブラウザーが確認を表示します。")

    with st.expander("文書・分析の管理", expanded=False):
        st.caption(f"読み込み済み: {len(st.session_state.versions)}文書 · 同時比較は最大4文書")
        upload_pdfs = st.file_uploader("追加するPDF", type=["pdf"], accept_multiple_files=True, key="pdf_upload")
        if upload_pdfs and st.button("PDFを読み込む", type="primary", width="stretch"):
            known = {v["sha256"] for v in st.session_state.versions}
            current = st.session_state.versions.copy()
            errors = []
            for file in upload_pdfs:
                raw = file.getvalue()
                digest = hashlib.sha256(raw).hexdigest()
                if digest in known:
                    existing = next(v for v in current if v["sha256"] == digest)
                    existing["pdf_bytes"] = existing.get("pdf_bytes") or raw
                    continue
                try:
                    version = extract_pdf(file.name, raw)
                    current.append(version)
                    known.add(digest)
                except Exception as exc:
                    errors.append(f"{file.name}: {exc}")
            st.session_state.versions = current
            st.session_state.topic_primary = {}
            st.session_state.topic_membership = {}
            st.session_state.topic_scores = {}
            if errors:
                for error in errors: st.error(error)
            st.rerun()

        if st.session_state.versions:
            with st.expander("版名を編集"):
                for version in st.session_state.versions:
                    version["name"] = st.text_input(version["filename"], value=version["name"], key=f"version_name_{version['id']}")
            st.session_state.topic_count = st.number_input(
                "自動発見トピック数 n", min_value=2, max_value=30,
                value=min(30, max(2, int(st.session_state.topic_count))), step=1)
            st.session_state.topic_membership_threshold = st.slider(
                "自動マルチタグの確率しきい値", .10, .50, float(st.session_state.topic_membership_threshold), .01,
                help="しきい値以上のタグを付けます。主トピックは常に含め、最大4タグまで表示します。")
            mode = st.selectbox("分類方式", ["automatic", "user"],
                                index=0 if st.session_state.active_topic_mode == "automatic" else 1,
                                format_func=lambda x: "自動発見（NMF）" if x == "automatic" else "ユーザー定義（訂正例から学習）")
            if mode != st.session_state.active_topic_mode:
                st.session_state.active_topic_mode = mode
                st.session_state.topic_primary = {}
            st.session_state.threshold = st.slider(
                "段落対応の類似度しきい値", .08, .55, float(st.session_state.threshold), .01,
                help="低い類似度の段落は既存段落へ無理に結び付けず、新設行にします。")
            if st.button("トピック分類を更新", width="stretch"):
                with st.spinner("ローカルで分類しています…"):
                    recalculate_topics()
            if st.button("全データをクリア", width="stretch"):
                st.session_state.clear_on_next_run = True
                if AUTOSAVE_PATH.exists(): AUTOSAVE_PATH.unlink()
                st.rerun()


versions = st.session_state.versions
if not versions:
    st.info("左側から比較するPDFを2つ以上読み込んでください。本文はこのPC内で処理されます。")
    st.markdown("#### はじめに\n1. 比較するPDFを追加します。\n2. 基準版と比較版（最大3版）を選びます。\n3. トピック分類、段落位置、メモ、Watchを編集します。\n4. プロジェクトZIPとして保存すると、PDFを含むすべての作業状態を復元できます。")
    st.stop()

version_ids = [v["id"] for v in versions]
if st.session_state.get("base_id") not in version_ids: st.session_state.base_id = version_ids[0]
if st.session_state.get("base_select") not in version_ids: st.session_state.base_select = st.session_state.base_id
base = next(v for v in versions if v["id"] == st.session_state.base_id)
if len(versions) > 1:
    options = [x for x in version_ids if x != base["id"]]
    current_targets = [x for x in st.session_state.get("target_select", st.session_state.target_ids) if x in options][:3]
    if not current_targets: current_targets = options[:min(3, len(options))]
    st.session_state.target_select = current_targets
    top = st.columns([1, 3])
    with top[0]:
        st.session_state.base_id = st.selectbox("段落順の基準版", version_ids,
                                                 format_func=lambda x: next(v["name"] for v in versions if v["id"] == x), key="base_select")
    base = next(v for v in versions if v["id"] == st.session_state.base_id)
    options = [x for x in version_ids if x != base["id"]]
    current_targets = [x for x in st.session_state.target_select if x in options][:3] or options[:min(3, len(options))]
    st.session_state.target_select = current_targets
    with top[1]:
        selected_ids = st.multiselect("同時に表示する比較版（最大3版）", options, max_selections=3,
                                      format_func=lambda x: next(v["name"] for v in versions if v["id"] == x), key="target_select")
    st.session_state.target_ids = selected_ids
    targets = [v for v in versions if v["id"] in selected_ids]
else:
    st.warning("比較するにはPDFを2つ以上読み込んでください。")
    targets = []

if versions and not st.session_state.topic_primary:
    with st.spinner("初回のトピック分類を実行しています…"): recalculate_topics()
if versions and st.session_state.last_saved_content_fingerprint is None:
    st.session_state.last_saved_content_fingerprint = project_content_fingerprint()
if not st.session_state.topic_summary and st.session_state.active_topic_mode == "user":
    st.info("ユーザー定義モードは、トピックとユーザー訂正例を登録すると分類を学習します。")

if targets:
    rows, links = make_multi_alignment(base, targets, float(st.session_state.threshold),
                                        st.session_state.alignment_overrides, st.session_state.placement_overrides,
                                        st.session_state.manual_order, st.session_state.topic_membership,
                                        st.session_state.topic_primary)
    st.session_state.last_links = links
else:
    rows, links = [], {}

# Work areas
if active_view == "比較":
    watch_by_paragraph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for watch_item in st.session_state.watch:
        watch_by_paragraph[watch_item.get("paragraph_id", "")].append(watch_item)
    if st.session_state.active_topic_mode == "user":
        tag_options = {topic["id"]: topic["name"] for topic in st.session_state.custom_topics}
    else:
        tag_options = {topic["id"]: topic["label"] for topic in st.session_state.topic_summary}
    if not targets:
        st.info("比較版を選択してください。")
    else:
        compared_versions = [base] + targets
        scope_ids = [version["id"] for version in compared_versions]
        old_scope = st.session_state.get("_visible_scope_ids") or []
        old_visible = set(st.session_state.get("visible_version_ids") or [])
        if old_scope:
            visible_ids = (old_visible & set(scope_ids)) | (set(scope_ids) - set(old_scope))
        else:
            visible_ids = set(scope_ids)
        st.session_state.visible_version_ids = [vid for vid in scope_ids if vid in visible_ids]
        st.session_state._visible_scope_ids = scope_ids
        visible_ids = st.multiselect(
            "表示する版（選択中の比較版から表示／非表示を切替）", scope_ids,
            key="visible_version_ids",
            format_func=lambda x: next(v["name"] + (" · 基準版" if v["id"] == base["id"] else "") for v in compared_versions if v["id"] == x))
        visible_versions = [v for v in compared_versions if v["id"] in visible_ids]
        visible_targets = [v for v in targets if v["id"] in visible_ids]
        if not visible_versions:
            st.info("表示する版を1つ以上選択してください。")
        else:
            headers = st.columns(len(visible_versions))
            for index, version in enumerate(visible_versions):
                headers[index].markdown(f"### {version['name']}" + (" · 基準版" if version["id"] == base["id"] else ""))
            st.caption(f"基準 {len(base['paragraphs'])}段落 · 比較版 {len(visible_targets)}版を表示 · {len(rows)}表示行")
            if len(visible_targets) > 1: st.caption("基準版の差分ハイライトは、表示中の先頭比較版との比較です。")
            st.markdown('<span class="badge b-add">新設</span> <span class="badge b-del">削除</span> <span class="badge b-change">統合候補</span> <span class="badge b-move">移動</span>', unsafe_allow_html=True)
            st.markdown('<span class="search-hit">検索語</span> · <span class="manual-highlight">保存した蛍光ペン</span>', unsafe_allow_html=True)
            jump_number = str(st.session_state.get("jump_para_pending", "")).strip()
            leftmost_version = visible_versions[0]
            jump_found = False
            for row in rows:
                lp = row["left"]
                all_right = [p for target in visible_targets for p in row["right"].get(target["id"], [])]
                if not (lp and base["id"] in visible_ids) and not all_right: continue
                leftmost_paras = ([lp] if leftmost_version["id"] == base["id"] and lp
                                  else row["right"].get(leftmost_version["id"], []))
                is_jump_row = bool(jump_number and any(str(p.get("number") or "") == jump_number for p in leftmost_paras))
                corpus = " ".join(([lp["text"], lp.get("memo", ""), str(lp.get("number") or "")] if lp else []) +
                                   [p["text"] + " " + p.get("memo", "") + " " + str(p.get("number") or "") for p in all_right])
                if search and search.casefold() not in corpus.casefold() and not is_jump_row: continue
                row_topics = {topic for p in ([lp] if lp else []) + all_right for topic in paragraph_topics(p["id"], st.session_state.topic_membership, st.session_state.topic_primary)}
                if selected_topic and selected_topic not in row_topics and not is_jump_row: continue
                if show_changes and visible_targets and row["kind"] == "matched" and lp and all(p["text"] == lp["text"] for p in all_right) and not is_jump_row: continue
                columns = st.columns(len(visible_versions), gap="small")
                for column_index, version in enumerate(visible_versions):
                    with columns[column_index]:
                        if version["id"] == base["id"]:
                            if lp:
                                reference_target = visible_targets[0] if visible_targets else None
                                other = " ".join(p["text"] for p in row["right"].get(reference_target["id"], [])) if reference_target else lp["text"]
                                left_text, _ = token_diff(lp["text"], other, search,
                                                          st.session_state.manual_highlights.get(lp["id"], []))
                                jump_target = is_jump_row and leftmost_version["id"] == base["id"] and str(lp.get("number") or "") == jump_number
                                jump_found = jump_found or jump_target
                                render_compare_paragraph(lp, left_text, base["id"], base["name"], row["kind"], True,
                                                         show_sections, show_watch, show_watch_toggle, watch_by_paragraph,
                                                         manual_tag_mode, tag_options, manual_highlight_mode, hide_memo, edit_memo, jump_target)
                            else:
                                st.markdown('<div class="empty-panel">基準版に対応する文言なし</div>', unsafe_allow_html=True)
                        else:
                            rps = row["right"].get(version["id"], [])
                            if not rps:
                                label = "この版に追加なし" if row["kind"] == "added" else "この版で削除／対応なし"
                                st.markdown(f'<div class="empty-panel">{label}</div>', unsafe_allow_html=True)
                            for rp in rps:
                                jump_target = is_jump_row and leftmost_version["id"] == version["id"] and str(rp.get("number") or "") == jump_number
                                jump_found = jump_found or jump_target
                                _, right_text = token_diff(lp["text"] if lp else "", rp["text"], search,
                                                           st.session_state.manual_highlights.get(lp["id"], []) if lp else [],
                                                           st.session_state.manual_highlights.get(rp["id"], []))
                                render_compare_paragraph(rp, right_text, version["id"], version["name"], row["kind"], False,
                                                         show_sections, show_watch, show_watch_toggle, watch_by_paragraph,
                                                         manual_tag_mode, tag_options, manual_highlight_mode, hide_memo, edit_memo, jump_target)
            if jump_number:
                if jump_found:
                    st.html("""<script>const target=document.getElementById('para-jump-target');if(target)target.scrollIntoView({behavior:'smooth',block:'start'});</script>""", unsafe_allow_javascript=True)
                    st.session_state.jump_para_pending = ""
                else:
                    st.sidebar.warning(f"表示中の一番左の版に段落番号 {jump_number} が見つかりません。")
                    st.session_state.jump_para_pending = ""
        with st.expander("対応付け・配置・順番を手動で修正"):
            target_ids = [v["id"] for v in targets]
            if target_ids:
                if st.session_state.get("alignment_target") not in target_ids:
                    st.session_state.alignment_target = target_ids[0]
                edit_target_id = st.selectbox("編集する比較版", target_ids, format_func=lambda x: next(v["name"] for v in targets if v["id"] == x), key="alignment_target")
                edit_target = next(v for v in targets if v["id"] == edit_target_id)
                para_labels = {p["id"]: f"Para. {p.get('number') or '—'} · p.{p['page']} · {p['text'][:72]}" for p in edit_target["paragraphs"]}
                base_labels = {p["id"]: f"Para. {p.get('number') or '—'} · {p['text'][:72]}" for p in base["paragraphs"]}
                if st.session_state.get("placement_edit_pid") not in para_labels:
                    st.session_state.placement_edit_pid = next(iter(para_labels))
                context_pid = st.session_state.placement_edit_pid
                placement_context = f"{edit_target_id}:{context_pid}"
                if st.session_state.get("_placement_context") != placement_context:
                    stored_link = st.session_state.alignment_overrides.get(context_pid, "__AUTO__")
                    st.session_state.placement_link = stored_link
                    stored_place = st.session_state.placement_overrides.get(context_pid, {})
                    if base["paragraphs"]:
                        st.session_state.placement_anchor = stored_place.get("anchor_id", base["paragraphs"][0]["id"])
                    st.session_state.placement_position = stored_place.get("position", "after")
                    st.session_state.placement_display_order = st.session_state.manual_order.get(
                        context_pid, next(i + 1 for i, p in enumerate(edit_target["paragraphs"]) if p["id"] == context_pid))
                    st.session_state._placement_context = placement_context
                with st.form("paragraph_placement_form"):
                    edit_pid = st.selectbox("配置を変更する段落", list(para_labels), format_func=lambda x: para_labels[x], key="placement_edit_pid")
                    current_link = st.session_state.alignment_overrides.get(edit_pid)
                    map_options = {"__AUTO__": "自動判定", **base_labels, "__ADD__": "新設行（基準段落と対応させない）"}
                    link_default = current_link if current_link in map_options else "__AUTO__"
                    if st.session_state.get("placement_link") not in map_options:
                        st.session_state.placement_link = link_default
                    new_link = st.selectbox("基準段落との対応", list(map_options), index=list(map_options).index(link_default), format_func=lambda x: map_options[x], key="placement_link")
                    current_place = st.session_state.placement_overrides.get(edit_pid, {})
                    anchor_options = {p["id"]: f"Para.  {p.get('number') or '—'} · {p['text'][:55]}" for p in base["paragraphs"]}
                    current_anchor = current_place.get("anchor_id") or (base["paragraphs"][0]["id"] if base["paragraphs"] else "")
                    if new_link == "__ADD__":
                        if st.session_state.get("placement_anchor") not in anchor_options:
                            st.session_state.placement_anchor = current_anchor
                        anchor_id = st.selectbox("新設行を置く基準段落", list(anchor_options), index=list(anchor_options).index(current_anchor) if current_anchor in anchor_options else 0, format_func=lambda x: anchor_options[x], key="placement_anchor")
                        position = st.radio("位置", ["after", "before"], index=0 if current_place.get("position", "after") == "after" else 1, horizontal=True, format_func=lambda x: "後" if x == "after" else "前", key="placement_position")
                    else:
                        anchor_id, position = "", "after"
                    original_order = next(i + 1 for i, p in enumerate(edit_target["paragraphs"]) if p["id"] == edit_pid)
                    display_order = st.number_input("比較版内の表示順", min_value=1, max_value=max(1, len(edit_target["paragraphs"])), value=int(st.session_state.manual_order.get(edit_pid, original_order)), step=1, key="placement_display_order")
                    if st.form_submit_button("配置を保存"):
                        if new_link == "__AUTO__": st.session_state.alignment_overrides.pop(edit_pid, None)
                        else: st.session_state.alignment_overrides[edit_pid] = new_link
                        if new_link == "__ADD__": st.session_state.placement_overrides[edit_pid] = {"anchor_id": anchor_id, "position": position}
                        else: st.session_state.placement_overrides.pop(edit_pid, None)
                        original_index = {p["id"]: i + 1 for i, p in enumerate(edit_target["paragraphs"])}
                        effective = sorted(edit_target["paragraphs"],
                                           key=lambda p: (st.session_state.manual_order.get(p["id"], original_index[p["id"]]),
                                                          original_index[p["id"]]))
                        new_order = [p["id"] for p in effective]
                        new_order.remove(edit_pid)
                        new_order.insert(int(display_order) - 1, edit_pid)
                        for rank, para_id in enumerate(new_order, 1): st.session_state.manual_order[para_id] = rank
                        if new_order == [p["id"] for p in edit_target["paragraphs"]]:
                            for para in edit_target["paragraphs"]: st.session_state.manual_order.pop(para["id"], None)
                        st.rerun()

if active_view == "トピックと学習":
    st.subheader("トピックと学習データ")
    st.caption("自動発見の表示名変更と、ユーザー定義トピックへの訂正例学習は別々に管理します。")
    if st.button("推奨の14マルチタグを追加して分類方式を切替"):
        existing_ids = {topic["id"] for topic in st.session_state.custom_topics}
        st.session_state.custom_topics.extend(
            {**topic} for topic in DEFAULT_TOPIC_TAGS if topic["id"] not in existing_ids)
        st.session_state.active_topic_mode = "user"
        with st.spinner("推奨タグを設定しています…"): recalculate_topics()
        st.rerun()
    st.caption("推奨タグを使う場合、段落に複数の正しいタグを選んで学習例として保存します。最初は正誤例がないため、分類結果は未確定です。")
    if st.session_state.active_topic_mode == "automatic":
        st.markdown("#### 自動発見トピック")
        for topic in st.session_state.topic_summary:
            with st.container(border=True):
                topic_id, old_label = topic["id"], topic["label"]
                new_label = st.text_input("トピック名", value=old_label, key=f"rename_{topic_id}")
                st.caption(f"{topic['count']}段落 · 特徴語: " + "、".join(topic.get("keywords", [])))
                if new_label.strip() and new_label != old_label:
                    duplicates = [x["label"] for x in st.session_state.topic_summary if x["id"] != topic_id]
                    if new_label.strip() in duplicates:
                        st.error("トピック名は重複できません。")
                        continue
                    st.session_state.topic_renames[topic_id] = new_label.strip()
                    for pid, label in list(st.session_state.topic_primary.items()):
                        if label == old_label: st.session_state.topic_primary[pid] = new_label.strip()
                    for pid, labels in list(st.session_state.topic_membership.items()):
                        st.session_state.topic_membership[pid] = [new_label.strip() if x == old_label else x for x in labels]
                    for pid, scores in list(st.session_state.topic_scores.items()):
                        if old_label in scores: scores[new_label.strip()] = scores.pop(old_label)
                    topic["label"] = new_label.strip()
        st.caption("名前の変更は表示ラベルだけに反映されます。")
    else:
        st.markdown("#### ユーザー定義トピック")
        with st.form("add_custom_topic"):
            new_topic_name = st.text_input("新しいトピック名（例：気候資金、適応）")
            new_topic_description = st.text_input("説明（任意）")
            if st.form_submit_button("トピックを追加") and new_topic_name.strip():
                if not any(t["name"].casefold() == new_topic_name.strip().casefold() for t in st.session_state.custom_topics):
                    st.session_state.custom_topics.append({"id": uuid.uuid4().hex[:12], "name": new_topic_name.strip(),
                                                          "description": new_topic_description.strip()})
                    st.session_state.topic_primary = {}
                    st.rerun()
        if st.session_state.custom_topics:
            with st.form("custom_topic_rename_form"):
                renamed_topics = {topic["id"]: {
                    "name": st.text_input("トピック名", value=topic["name"], key=f"custom_topic_{topic['id']}"),
                    "description": st.text_input("説明", value=topic.get("description", ""), key=f"custom_topic_desc_{topic['id']}")}
                    for topic in st.session_state.custom_topics}
                if st.form_submit_button("トピック名を保存"):
                    proposed = [renamed_topics[t["id"]]["name"].strip() or t["name"] for t in st.session_state.custom_topics]
                    if len({name.casefold() for name in proposed}) != len(proposed):
                        st.error("トピック名は重複できません。")
                    else:
                        for topic, proposed_name in zip(st.session_state.custom_topics, proposed):
                            topic["name"] = proposed_name
                            topic["description"] = renamed_topics[topic["id"]]["description"].strip()
                        with st.spinner("トピック名を反映しています…"): recalculate_topics()
                        st.rerun()
            for topic in list(st.session_state.custom_topics):
                if topic.get("description"): st.caption(topic["description"])
                if st.button(f"「{topic['name']}」を削除", key=f"delete_topic_{topic['id']}"):
                    tid = topic["id"]
                    st.session_state.custom_topics = [t for t in st.session_state.custom_topics if t["id"] != tid]
                    for sample in st.session_state.training_data: sample["topics"] = [x for x in sample.get("topics", []) if x != tid]
                    for pid, tag_ids in list(st.session_state.manual_topic_overrides.items()):
                        st.session_state.manual_topic_overrides[pid] = [tag_id for tag_id in tag_ids if tag_id != tid]
                    st.session_state.topic_correct_labels = [x for x in st.session_state.get("topic_correct_labels", []) if x != tid]
                    recalculate_topics()
                    st.rerun()
            st.markdown("#### 分類を確認・訂正して学習データに追加")
            all_paras = all_paragraphs(versions)
            pmap = {p["id"]: p for p in all_paras}
            vmap = {v["id"]: v for v in versions}
            valid_pids = list(pmap)
            if st.session_state.get("topic_correct_pid") not in pmap:
                st.session_state.topic_correct_pid = valid_pids[0]
            correction_pid = st.session_state.topic_correct_pid
            saved_sample = next((sample for sample in st.session_state.training_data
                                 if sample.get("paragraph_id") == correction_pid), None)
            topic_id_by_name = {topic["name"]: topic["id"] for topic in st.session_state.custom_topics}
            suggested_ids = (saved_sample.get("topics", []) if saved_sample else
                             [topic_id_by_name[name] for name in st.session_state.topic_membership.get(correction_pid, [])
                              if name in topic_id_by_name])
            if st.session_state.get("_topic_correction_context") != correction_pid:
                st.session_state.topic_correct_labels = suggested_ids
                st.session_state._topic_correction_context = correction_pid
            selected_pid = st.selectbox("判定する段落", list(pmap), format_func=lambda pid: f"{vmap[next(v['id'] for v in versions if any(p['id']==pid for p in v['paragraphs']))]['name']} · 段落 {pmap[pid].get('number') or '—'} · {pmap[pid]['text'][:95]}", key="topic_correct_pid")
            selected_para = pmap[selected_pid]
            with st.form("topic_correction_form"):
                st.caption(selected_para["text"])
                current_labels = [x for x in st.session_state.topic_membership.get(selected_pid, []) if x in {t["name"] for t in st.session_state.custom_topics}]
                selected_topic_ids = st.multiselect("正しいトピック（該当なしは空欄）", [t["id"] for t in st.session_state.custom_topics],
                                                     default=[t["id"] for t in st.session_state.custom_topics if t["name"] in current_labels],
                                                     format_func=lambda tid: next(t["name"] for t in st.session_state.custom_topics if t["id"] == tid), key="topic_correct_labels")
                st.write("モデルの推定: " + " · ".join(f"{name} {prob:.0%}" for name, prob in st.session_state.topic_scores.get(selected_pid, {}).items()) if st.session_state.topic_scores.get(selected_pid) else "まだ推定されていません")
                if st.form_submit_button("この正誤を学習データに保存"):
                    sample = {"paragraph_id": selected_pid,
                              "version_id": next(v["id"] for v in versions if any(p["id"] == selected_pid for p in v["paragraphs"])),
                              "text": selected_para["text"], "topics": selected_topic_ids,
                              "reviewed_at": datetime.now(timezone.utc).isoformat()}
                    old_samples = [s for s in st.session_state.training_data if s["paragraph_id"] != selected_pid]
                    st.session_state.training_data = old_samples + [sample]
                    st.session_state.manual_topic_overrides[selected_pid] = list(selected_topic_ids)
                    with st.spinner("訂正例を反映して学習しています…"): recalculate_topics()
                    st.success("この判定を教師データに保存し、分類を更新しました。")
                    st.rerun()
            st.caption(f"保存済みの確認例: {len(st.session_state.training_data)}段落。確率は推定値で、例が少ない場合は本文との類似度を参考表示します。")
            if st.session_state.topic_summary:
                st.markdown("#### 分類状況")
                for topic in st.session_state.topic_summary:
                    st.write(f"**{topic['label']}** · {topic['count']}段落 · {topic.get('quality', '')}")
        else:
            st.info("先にユーザー定義トピックを追加してください。")

if active_view == "Watch付箋":
    st.subheader("Watch付箋")
    version_choices = [v["id"] for v in versions]
    if st.session_state.get("watch_version") not in version_choices:
        st.session_state.watch_version = version_choices[0]
    selected_watch_version = next(v for v in versions if v["id"] == st.session_state.watch_version)
    watch_para_choices = [p["id"] for p in selected_watch_version["paragraphs"]]
    if st.session_state.get("watch_paragraph") not in watch_para_choices:
        st.session_state.watch_paragraph = watch_para_choices[0]
    with st.form("watch_add"):
        w_version_id = st.selectbox("対象の版", [v["id"] for v in versions], format_func=lambda x: next(v["name"] for v in versions if v["id"] == x), key="watch_version")
        w_version = next(v for v in versions if v["id"] == w_version_id)
        w_pid = st.selectbox("対象段落", [p["id"] for p in w_version["paragraphs"]], format_func=lambda x: next(f"段落 {p.get('number') or '—'} · {page_ref(p)} · {p['text'][:90]}" for p in w_version["paragraphs"] if p["id"] == x), key="watch_paragraph")
        w_para = next(p for p in w_version["paragraphs"] if p["id"] == w_pid)
        st.caption(w_para["text"])
        clauses = [x.strip() for x in re.split(r"(?<=[.;?!])\s+|(?<=:)\s+", w_para["text"]) if x.strip()]
        quote_choices = ["段落全体"] + clauses
        old_quotes = st.session_state.get("watch_quotes")
        if not isinstance(old_quotes, list): old_quotes = []
        st.session_state.watch_quotes = [quote for quote in old_quotes if quote in quote_choices]
        selected_quotes = st.multiselect("付箋にする文言を選択（未選択なら段落全体）", quote_choices, key="watch_quotes") or []
        quote_override = st.text_input("文言の範囲を手入力で調整（任意）", key="watch_quote_override")
        note = st.text_input("メモ", key="watch_note")
        tags = st.text_input("タグ（任意、カンマ区切り）", key="watch_tags")
        priority = st.selectbox("優先度", ["通常", "要確認", "重要"], key="watch_priority")
        if st.form_submit_button("Watchに追加"):
            quote = (quote_override or "").strip() or (w_para["text"] if not selected_quotes or "段落全体" in selected_quotes else " … ".join(selected_quotes))
            item = {"version_id": w_version_id, "version_name": w_version["name"], "paragraph_id": w_pid,
                    "paragraph_number": w_para.get("number"), "page": w_para["page"], "quote": quote.strip(),
                    "note": note.strip(), "tags": [x.strip() for x in tags.split(",") if x.strip()], "priority": priority}
            if not any(x["version_id"] == w_version_id and x["paragraph_id"] == w_pid and x["quote"] == item["quote"] for x in st.session_state.watch):
                st.session_state.watch.append(item)
            st.rerun()
    for idx, item in enumerate(st.session_state.watch):
        with st.container(border=True):
            st.markdown(f"**{item['priority']}** · {item['version_name']} · 段落 {item.get('paragraph_number') or '—'} · p.{item['page']}")
            st.markdown(f'<div class="watch-item">{html.escape(item["quote"])}</div>', unsafe_allow_html=True)
            if item.get("note"): st.write("メモ: " + item["note"])
            if item.get("tags"): st.caption("タグ: " + ", ".join(item["tags"]))
            counterparts = st.session_state.get("last_links", {}).get(item["paragraph_id"], [])
            if counterparts:
                lookup = {p["id"]: p for p in all_paragraphs(versions)}
                st.write("対応候補: " + " · ".join(f"{lookup[x].get('number') or '—'}: {lookup[x]['text'][:120]}" for x in counterparts if x in lookup))
            if st.button("付箋を削除", key=f"watch_del_{idx}"):
                st.session_state.watch.pop(idx)
                st.rerun()
    if st.session_state.watch:
        watch_csv = io.StringIO(newline="")
        writer = csv.DictWriter(watch_csv, fieldnames=["version_name", "version_id", "paragraph_id", "paragraph_number", "page", "quote", "note", "tags", "priority"])
        writer.writeheader()
        for item in st.session_state.watch: writer.writerow({**item, "tags": ",".join(item["tags"])})
        st.download_button("WatchをCSVで保存", watch_csv.getvalue().encode("utf-8-sig"), "watch_list.csv", "text/csv")

if active_view == "メモ":
    st.subheader("メモを登録・表示")
    memo_version_choices = [v["id"] for v in versions]
    if st.session_state.get("memo_version") not in memo_version_choices:
        st.session_state.memo_version = memo_version_choices[0]
    selected_memo_version = next(v for v in versions if v["id"] == st.session_state.memo_version)
    memo_para_choices = [p["id"] for p in selected_memo_version["paragraphs"]]
    if st.session_state.get("memo_paragraph") not in memo_para_choices:
        st.session_state.memo_paragraph = memo_para_choices[0]
    memo_version_id = st.selectbox("メモを編集する版", [v["id"] for v in versions], format_func=lambda x: next(v["name"] for v in versions if v["id"] == x), key="memo_version")
    memo_version = next(v for v in versions if v["id"] == memo_version_id)
    memo_pid = st.selectbox("Para.", [p["id"] for p in memo_version["paragraphs"]], format_func=lambda x: next(f"Para. {p.get('number') or '—'} · p.{p['page']} · {p['text'][:95]}" for p in memo_version["paragraphs"] if p["id"] == x), key="memo_paragraph")
    memo_para = next(p for p in memo_version["paragraphs"] if p["id"] == memo_pid)
    st.markdown(f"**英文原文**\n\n{memo_para['text']}")
    with st.form("memo_edit"):
        memo_text = st.text_area("メモ", value=memo_para.get("memo", ""), height=150, key="memo_text")
        if st.form_submit_button("メモを保存"):
            memo_para["memo"] = memo_text.strip()
            inline_key = f"memo_inline_{memo_pid}"
            if inline_key in st.session_state: st.session_state[inline_key] = memo_para["memo"]
            st.rerun()
    st.download_button("全段落のメモCSVテンプレート", csv_memos(versions, blank=True), "memo_template.csv", "text/csv")
    memo_file = st.file_uploader("メモを入力したCSVを読み込む", type=["csv"], key="memo_csv")
    if memo_file and st.button("CSVのメモを反映"):
        updated, errors = load_memo_csv(memo_file, versions)
        st.success(f"{updated}段落のメモを反映しました。")
        for error in errors: st.warning(error)
    st.download_button("現在のメモをCSVで保存", csv_memos(versions), "memos.csv", "text/csv")

if active_view == "出力・保存":
    st.subheader("比較HTML・全セッション保存")
    st.markdown("プロジェクトZIPは、PDF本体、抽出本文、分類設定・訂正例、手動配置・順番、Watch、メモ、表示選択をまとめて保存します。")
    if targets:
        site = make_html_site(base, targets, rows, st.session_state.topic_primary, st.session_state.topic_membership,
                              st.session_state.topic_scores, st.session_state.watch,
                              st.session_state.manual_highlights)
        st.download_button("現在の複数版比較をHTMLで保存", site.encode("utf-8"), "comparison.html", "text/html", type="primary")
        package = io.BytesIO()
        with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("comparison.html", site.encode("utf-8"))
            for version in versions:
                if version.get("pdf_bytes"): archive.writestr(f"pdfs/{bundled_pdf_name(version)}", version["pdf_bytes"])
            archive.writestr("project.json", json.dumps(project_object(), ensure_ascii=False, indent=2).encode("utf-8"))
            archive.writestr("memos.csv", csv_memos(versions))
        st.download_button("HTML＋PDF＋全セッションをZIP保存", package.getvalue(), "pdf_comparison_package.zip", "application/zip")
    else:
        st.info("比較版を選ぶとHTMLを出力できます。プロジェクトZIPは比較版がなくても保存できます。")
    if versions:
        render_project_download_button("全セッションのプロジェクトZIPを保存", "project_save_export")
    st.caption("自動保存: 無効 · 作業を残す場合はプロジェクトZIPをダウンロードしてください。")

if st.session_state.versions:
    current_fingerprint = project_content_fingerprint()
    unsaved_changes = current_fingerprint != st.session_state.get("last_saved_content_fingerprint")
    if unsaved_changes:
        st.sidebar.warning("未保存の変更があります。ページを閉じる前にプロジェクトZIPを保存してください。")
    else:
        st.sidebar.caption("プロジェクトは現在の内容と一致しています。")
    guard_script = """<script>
window.__pdfCompareProjectUnsaved = __DIRTY__;
if (!window.__pdfCompareBeforeUnloadInstalled) {
  window.addEventListener("beforeunload", function (event) {
    if (window.__pdfCompareProjectUnsaved) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  window.__pdfCompareBeforeUnloadInstalled = true;
}
</script>""".replace("__DIRTY__", "true" if unsaved_changes else "false")
    st.html(guard_script, unsafe_allow_javascript=True)

if st.session_state.versions and st.session_state.autosave_enabled:
    try:
        temporary_save = AUTOSAVE_PATH.with_suffix(".zip.tmp")
        temporary_save.write_bytes(project_bundle())
        temporary_save.replace(AUTOSAVE_PATH)
    except OSError as exc:
        st.sidebar.warning(f"ローカル自動保存に失敗: {exc}")
