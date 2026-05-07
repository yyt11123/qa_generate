"""
QA 对生成脚本 (markdown 版)
输入: 单个 .md 文件 (保险公司业务手册)
输出: qa_pairs.jsonl, 每行一条 QA 对, 格式对齐老师测试集

用法:
    python qa_generate.py --input "保司文件2.0/万通/万通缴费指引/内部缴费指引.md" --output qa_pairs.jsonl
    python qa_generate.py --input xxx.md --output qa_pairs.jsonl --start-id 100

依赖:
    pip install openai tqdm python-dotenv
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# ============== 配置区 ==============
API_KEY = os.environ.get("VOLC_API_KEY")
BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
MODEL = "ark-code-latest"

# 每个 md 文档生成多少个 QA 对
QA_PER_DOC = 8
# 重试次数
MAX_RETRIES = 3
# 内容太短的 md 跳过 (避免空文档/目录页)
MIN_CHARS = 200
# ===================================


# ============== 分类规则 (基于文件名关键词匹配) ==============
# 顺序敏感: 优先匹配前面的规则
CATEGORY_RULES = [
    ("缴费", ["缴费", "繳費", "保费", "保費", "付款", "繳款"]),
    ("核保", ["核保", "健康"]),
    ("理赔", ["理赔", "理賠", "赔偿", "賠償"]),
    ("产品", ["产品", "產品", "条款", "條款", "保单", "保單"]),
    ("行政规则", ["行政", "操作", "流程", "服务", "服務", "投保", "操作指引", "管理"]),
    ("案例", ["案例"]),
    ("优惠推广", ["优惠", "優惠", "推廣", "推广", "禮遇", "礼遇", "预缴", "預繳"]),
]
DEFAULT_CATEGORY = "一般查询"

# 保司文件2.0 下不是保司名的特殊目录
_NON_COMPANY_DIRS = {"优惠文件", "非标准文本文件"}
# 日期目录名, 如 "24年12月"
_DATE_DIR_RE = re.compile(r"^\d{2}年\d{1,2}月$")


def classify_doc(doc_filename: str) -> str:
    """根据文件名关键词匹配业务分类"""
    name = doc_filename.lower()
    for category, keywords in CATEGORY_RULES:
        for kw in keywords:
            if kw.lower() in name:
                return category
    return DEFAULT_CATEGORY


def extract_company_from_path(md_path: Path, base_dir: str = "保司文件2.0") -> str:
    """
    从 md 文件路径里提取保司名 (即 base_dir 下的第一级子目录)
    例: .../保司文件2.0/万通/xxx/yyy.md -> 万通
    特殊处理:
      - 非标准文本文件/保司名/... -> 保司名
      - 优惠文件/日期/保司名.pdf-uuid/... -> 保司名
    """
    parts = md_path.parts
    if base_dir in parts:
        idx = parts.index(base_dir)
        if idx + 1 < len(parts):
            candidate = parts[idx + 1]

            if candidate == "非标准文本文件" and idx + 2 < len(parts):
                return parts[idx + 2]

            if candidate == "优惠文件":
                # 路径形如: 优惠文件/24年12月/万通.pdf-uuid/full.md
                parent_name = md_path.parent.name
                company = parent_name.split(".pdf-", 1)[0] if ".pdf-" in parent_name else parent_name
                if company:
                    return company

            return candidate

    # 兜底: 从父目录链中找到第一个有意义的目录名
    # 跳过 parents[0] (文档名目录本身), 从 grandparent 开始找
    for p in list(md_path.parents)[1:]:
        if not p.name or p.name in _NON_COMPANY_DIRS or _DATE_DIR_RE.match(p.name):
            continue
        if ".pdf-" in p.name:
            continue
        return p.name

    return "未知保司"


def extract_doc_name_from_path(md_path: Path) -> str:
    """
    从 md 路径中提取规范的"文档名"
    实际数据中, md 文件常常都叫 full.md, 真正的文档名在父目录上, 形如:
        內部繳費指引.pdf-f6385a61-99f1-4c89-95ef-947fae60dcc3/full.md
        富饶千秋产品手册_2025_01/富饶千秋产品手册_2025_01.md
    策略: 优先使用父目录名, 并去掉常见的后缀杂质 (如 .pdf-uuid)
    """
    parent_name = md_path.parent.name
    # 如果父目录名形如 "xxx.pdf-{uuid}", 去掉 .pdf 及之后的部分
    cleaned = re.sub(
        r"\.pdf-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        "",
        parent_name,
        flags=re.IGNORECASE,
    )

    parts = md_path.parts
    # 优惠文件下的文档名额外携带有意义的上下文: 保司_优惠_日期
    if "优惠文件" in parts:
        grandparent = md_path.parents[1].name
        if _DATE_DIR_RE.match(grandparent):
            return f"{cleaned}_优惠_{grandparent}"

    # 兜底: 如果父目录名也没意义(比如就叫 "万通" 这种, 即文件直接在保司目录下),
    # 回退用文件名 stem
    if not cleaned or cleaned == md_path.parents[1].name:
        cleaned = md_path.stem
    return cleaned


# ============== Prompt ==============
SYSTEM_PROMPT = """你是一个专业的保险知识 QA 数据标注员。你正在基于真实的保险公司业务手册(核保/理赔/缴费/行政等)生成 QA 对,用于评估检索系统在保险知识库上的表现。

## 使用场景

这些 QA 是**保险公司新员工/业务员**在工作中查手册时会问的问题。不是消费者闲聊,不是法律咨询,而是**业务工作场景下的快速查询**。

## 问题书写要求

### 1. 必须以保险公司中文名作为前缀

例如: "万通缴费可以使用现金吗?" / "保诚投保人变更流程是怎样的?" / "富卫如何申请保单假期?"

公司名将通过下面的 user prompt 给你, 必须严格使用给定的公司名作为问题开头。

### 2. 问题风格: 简短书面、关键词式

不是网络口语("咋办、啥情况"),也不是法律条款式("根据本规定..."),而是**业务员查手册时直接问出关键词**:

✅ 好: "万通缴费可以使用信用卡吗?"
✅ 好: "保诚投保人变更需要什么材料?"
✅ 好: "富卫如何操作保单价值提取?"
✅ 好: "万通使用港币缴付美元保单的汇率怎么算?"

❌ 不要: "我想问一下万通能不能用现金缴费啊?"  (太啰嗦)
❌ 不要: "投保人欲变更保单权益人时应当遵循何种程序?"  (太书面)
❌ 不要: "万通缴费现金?"  (太碎片)

### 3. 长度

8 - 25 个汉字之间最佳。

## 答案书写要求(重要,和之前不同!)

**答案必须是从原文中直接摘取的片段,不要改写、不要归纳、不要翻译繁体字!**

具体要求:
- 直接复制原文中能回答问题的句子或段落
- 保留原文的繁体字、专业术语、格式标点
- 如果原文是列表/编号格式,保留编号
- 不要用普通话改写繁体内容
- 不要加"根据..."这种导引语
- 答案长度: 一般 30-300 字, 太短(< 10 字)说明问题指向不明确, 太长(> 500 字)说明问题太宽泛

## 分类字段

每条 QA 必须打一个分类标签, 从这 8 个里选一个最贴切的:
- 缴费: 缴费方式、币种、汇率、信用卡、行政费等
- 核保: 健康核保、疾病评估、核保要求、体检等
- 理赔: 理赔流程、所需材料、理赔时效等
- 产品: 产品条款、保障范围、保单内容、产品手册等
- 行政规则: 投保流程、保单变更、退保、保单服务、操作指引等
- 案例: 具体案例分析
- 优惠推广: 保费折扣、优惠活动、限时礼遇、预缴优惠等
- 一般查询: 不属于以上任何一类的通用问题

## 严格规则

1. 答案必须能在原文中直接找到 (字面匹配, 不能编造)
2. 问题必须独立可理解, 禁止"上述/本条款/前文"等指代词
3. 一份文档生成的多个问题之间, 主题要尽量分散, 不要重复问同一个点

## 输出格式

严格输出 JSON 数组, 不要任何额外说明、不要 markdown 代码块。每个元素如下:
{
  "question": "<带保险公司名前缀的问题>",
  "answer": "<原文片段, 保留原文格式和繁体字>",
  "category": "<7 个分类之一>",
  "evidence": "<原文中答案所在的更完整段落, 用于人工核验>"
}
"""


USER_PROMPT_TEMPLATE = """保险公司名: {company}
文档名称: {doc_filename}
默认分类(可参考但不必拘泥): {default_category}

文档内容:
```
{md_content}
```

请基于以上文档内容, 生成 {n} 个高质量 QA 对。

注意:
1. 每个问题必须以"{company}"开头
2. 答案必须从原文直接摘取, 保留繁体字
3. 多个问题主题要分散
4. 直接输出 JSON 数组, 不要任何额外文字
"""


def extract_json_array(text: str):
    """从模型输出中提取 JSON 数组, 容错处理 markdown 代码块和非法转义"""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"模型输出中未找到 JSON 数组: {text[:200]}")
    text = text[start : end + 1]
    # 修复 LLM 产生的非法 JSON 转义 (如 \\% \\$ 等)
    # JSON 仅允许 \\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX 这些转义
    # 把反斜杠后跟非法字符的反斜杠转义为双反斜杠
    text = re.sub(
        r'\\(?!["\\/bfnrtu])',
        r"\\\\",
        text,
    )
    return json.loads(text)


def call_llm(client: OpenAI, system: str, user: str) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            wait = 2**attempt
            print(f"[warn] 第 {attempt+1} 次调用失败: {e}, {wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"调用 LLM 失败 {MAX_RETRIES} 次: {last_err}")


VALID_CATEGORIES = {"缴费", "核保", "理赔", "产品", "行政规则", "案例", "优惠推广", "一般查询"}


def validate_qa(item: dict, company: str) -> tuple:
    """基本字段校验, 返回 (是否有效, 原因)"""
    required = {"question", "answer", "category", "evidence"}
    if not required.issubset(item.keys()):
        return False, "字段缺失"
    if not all(isinstance(item[k], str) and item[k].strip() for k in required):
        return False, "字段为空"
    # 问题必须以保司名开头
    if not item["question"].lstrip().startswith(company):
        return False, f"问题未以「{company}」开头"
    # 过滤指代词
    bad_refs = ["本条款", "本条", "上述", "该条", "如前所述", "前文", "该项"]
    if any(ref in item["question"] for ref in bad_refs):
        return False, "包含指代词"
    # category 必须合法
    if item["category"] not in VALID_CATEGORIES:
        return False, f"非法分类「{item['category']}」"
    # 答案不能太短 (太短说明 LLM 偷懒了)
    if len(item["answer"].strip()) < 5:
        return False, "答案过短"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入的 .md 文件路径")
    parser.add_argument("--output", required=True, help="输出的 qa_pairs.jsonl 路径")
    parser.add_argument(
        "--qa-per-doc",
        type=int,
        default=QA_PER_DOC,
        help=f"每个文档生成多少 QA, 默认 {QA_PER_DOC}",
    )
    parser.add_argument(
        "--start-id", type=int, default=1, help="qa_id 起始编号, 跨多文档批处理时使用"
    )
    parser.add_argument(
        "--base-dir",
        default="保司文件2.0",
        help="zip 解压根目录名, 用来从路径里识别保司名",
    )
    args = parser.parse_args()

    md_path = Path(args.input)
    if not md_path.exists():
        print(f"[错误] 找不到文件: {md_path}")
        return

    # 1. 读 md 内容
    md_content = md_path.read_text(encoding="utf-8")
    if len(md_content) < MIN_CHARS:
        print(f"[跳过] 文档过短 ({len(md_content)} 字符 < {MIN_CHARS}): {md_path.name}")
        return

    # 2. 提取保司名 + 文档名 + 分类
    company = extract_company_from_path(md_path, base_dir=args.base_dir)
    doc_name = extract_doc_name_from_path(
        md_path
    )  # 规范化的文档名 (用于 source_doc / chunk_id)
    doc_filename = md_path.name  # 实际文件名 (常常是 full.md)
    default_category = classify_doc(
        doc_name
    )  # 用规范化文档名做分类匹配, 而不是 full.md

    print(f"文件: {doc_filename}  (实际路径: {md_path})")
    print(f"文档名: {doc_name}")
    print(f"保司: {company}")
    print(f"预设分类: {default_category}")
    print(f"内容长度: {len(md_content)} 字符")
    print()

    # 3. 安全检查: 输出文件已存在
    output_path = Path(args.output)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[警告] 输出文件 {args.output} 已存在且非空。")
        print(f"  - 如果是批处理累加, 输入 y 继续")
        print(f"  - 如果是同一文档重跑, 输入 n 后先删除旧文件")
        choice = input("继续追加? (y/n): ").strip().lower()
        if choice != "y":
            print(f"已退出。删除旧文件: Remove-Item {args.output}")
            return

    # 4. 调用 LLM 生成
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        company=company,
        doc_filename=doc_name,  # 用规范化的文档名传给 LLM, 不是 full.md
        default_category=default_category,
        md_content=md_content,
        n=args.qa_per_doc,
    )

    print("正在调用 LLM 生成 QA, 请稍等...")
    raw = call_llm(client, SYSTEM_PROMPT, user_prompt)

    # 5. 解析
    try:
        items = extract_json_array(raw)
    except Exception as e:
        print(f"[错误] 解析 JSON 失败: {e}")
        print("LLM 返回原文:")
        print(raw[:1000])
        return

    # 6. 校验 + 写盘
    qa_id = args.start_id
    written = 0
    rejected = []
    with open(args.output, "a", encoding="utf-8") as fout:
        for item in items:
            ok, reason = validate_qa(item, company)
            if not ok:
                rejected.append((item.get("question", "<无>")[:50], reason))
                continue
            record = {
                "qa_id": f"qa_{qa_id:05d}",
                "category": item["category"].strip(),
                "question": item["question"].strip(),
                "answer": item["answer"].strip(),
                "source_doc": doc_name,  # 用规范化的文档名, 不是 full.md
                "source_company": company,
                "source_chunk_id": f"{company}_{doc_name}",
                "evidence": item["evidence"].strip(),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            qa_id += 1
            written += 1

    # 7. 报告
    print()
    print(f"=== 完成 ===")
    print(f"成功写入: {written} 条")
    print(f"被过滤: {len(rejected)} 条")
    for q, reason in rejected:
        print(f"  [skip] {reason}: {q}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
