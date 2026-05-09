# -*- coding: utf-8 -*-
import qa_generate as q


def test_split_by_headings():
    # 1) 短文本不切
    short = "hello world"
    assert q.split_by_headings(short, 4500) == [short]

    # 2) 有标题, 按标题断
    doc = (
        "# 第一章\n" + "A" * 3000 + "\n"
        "## 第二节\n" + "B" * 3000 + "\n"
        "# 第二章\n" + "C" * 3000 + "\n"
    )
    chunks = q.split_by_headings(doc, 4500)
    # 第一章(~3010) + 第二节(~3010) > 4500 -> 第一章单独一个 chunk
    assert len(chunks) >= 2
    # 所有 chunk 拼回原文
    assert "".join(chunks) == doc
    print(f"split_by_headings: {len(doc)} chars -> {len(chunks)} chunks, sizes={[len(c) for c in chunks]}")

    # 3) 无标题的长文本 -> 硬切
    no_heading = "X" * 15000
    chunks2 = q.split_by_headings(no_heading, 4500)
    assert len(chunks2) >= 3
    assert "".join(chunks2) == no_heading
    print(f"no-heading hard split: {len(chunks2)} chunks")

    # 4) 空文本
    assert q.split_by_headings("", 4500) == [""]
    print("split_by_headings ok")


def test_dedup():
    records = [
        {"question": "太保冷静期多少天？", "answers": "21天"},
        {"question": "太保冷静期多少天?", "answers": "21天"},  # 全半角问号不同
        {"question": "太保冷静期多少天", "answers": "21天"},   # 无问号
        {"question": "太保投保年龄限制？", "answers": "0-65"},
    ]
    kept, dup = q._dedup_by_question(records)
    assert len(kept) == 2, kept
    assert dup == 2
    assert kept[0]["question"] == "太保冷静期多少天？"
    assert kept[1]["question"] == "太保投保年龄限制？"
    print("_dedup_by_question ok")


def test_chunking_threshold():
    # 确认常量存在且合理
    assert q.CHUNK_THRESHOLD == 5000
    assert q.CHUNK_SIZE == 4500
    assert q.QA_PER_CHUNK == 5
    assert q.QA_PER_SHORT_DOC == 6
    assert q.LLM_TIMEOUT == 300
    print("constants ok")


if __name__ == "__main__":
    test_split_by_headings()
    test_dedup()
    test_chunking_threshold()
    print("ALL OK")
