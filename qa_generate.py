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
# 单次 LLM 调用最大输出 tokens, 避免超长 HTML 表格导致 JSON 被截断
MAX_TOKENS = 8192
# 单次 LLM 请求超时秒数 (504/连接超时防御)
LLM_TIMEOUT = 300

# ---- 分段出题 ----
# 文档 >= CHUNK_THRESHOLD 字符时启用分段 (未达阈值就整篇出题)
CHUNK_THRESHOLD = 5000
# 目标切片大小 (字符数). 会在最近的 markdown 标题处断开, 实际 chunk 可能稍大/稍小.
CHUNK_SIZE = 4500
# 每个 chunk 出多少 QA
QA_PER_CHUNK = 5
# 短文档(< CHUNK_THRESHOLD) 一次性出多少 QA
QA_PER_SHORT_DOC = 6
# ===================================


# ============== 枚举 ==============
VALID_CATEGORIES = {"缴费", "核保", "理赔", "产品", "行政规则", "案例", "一般查询"}
VALID_PROMPT_TYPES = {
    "概念查询",
    "条件判断",
    "列表枚举",
    "流程查询",
    "数值查询",
    "逻辑推理",
}

# LLM 偶发把标签繁体化/自造近义词, 校验前统一归一到简体标准值.
# key 做大小写/首尾空白归一后匹配 (见 _canon_label).
CATEGORY_MAP = {
    # 繁体 -> 简体
    "繳費": "缴费",
    "核保": "核保",
    "理賠": "理赔",
    "產品": "产品",
    "行政規則": "行政规则",
    "案例": "案例",
    "一般查詢": "一般查询",
    # LLM 自造近义词 -> 最接近的标准分类
    "保单管理": "行政规则",
    "保單管理": "行政规则",
    "保单服务": "行政规则",
    "保單服務": "行政规则",
    "保单变更": "行政规则",
    "保單變更": "行政规则",
    "投保": "行政规则",
    "投保流程": "行政规则",
    "退保": "行政规则",
    "保费": "缴费",
    "保費": "缴费",
    "缴款": "缴费",
    "繳款": "缴费",
    "付款": "缴费",
    "健康核保": "核保",
    "健康": "核保",
    "理赔申请": "理赔",
    "理賠申請": "理赔",
    "索偿": "理赔",
    "索償": "理赔",
    "产品条款": "产品",
    "產品條款": "产品",
    "条款": "产品",
    "條款": "产品",
    "保障": "产品",
    "案例分析": "案例",
    "其他": "一般查询",
    "其它": "一般查询",
}

PROMPT_TYPE_MAP = {
    # 繁体 -> 简体
    "概念查詢": "概念查询",
    "條件判斷": "条件判断",
    "列表枚舉": "列表枚举",
    "流程查詢": "流程查询",
    "數值查詢": "数值查询",
    "邏輯推理": "逻辑推理",
    # 近义/简写 -> 标准
    "概念": "概念查询",
    "判断": "条件判断",
    "判斷": "条件判断",
    "是否": "条件判断",
    "枚举": "列表枚举",
    "枚舉": "列表枚举",
    "列表": "列表枚举",
    "清单": "列表枚举",
    "清單": "列表枚举",
    "流程": "流程查询",
    "步骤": "流程查询",
    "步驟": "流程查询",
    "数值": "数值查询",
    "數值": "数值查询",
    "数字": "数值查询",
    "數字": "数值查询",
    "计算": "数值查询",
    "計算": "数值查询",
    "推理": "逻辑推理",
    "逻辑": "逻辑推理",
    "邏輯": "逻辑推理",
}


def _canon_label(s: str) -> str:
    """标签比对前的轻度归一: 去首尾空白"""
    return (s or "").strip()


def normalize_category(raw: str) -> str:
    """LLM 输出的 category 规范化到 VALID_CATEGORIES. 返回值未必合法, 交给 validate_qa 校验"""
    v = _canon_label(raw)
    if v in VALID_CATEGORIES:
        return v
    return CATEGORY_MAP.get(v, v)


def normalize_prompt_type(raw: str) -> str:
    v = _canon_label(raw)
    if v in VALID_PROMPT_TYPES:
        return v
    return PROMPT_TYPE_MAP.get(v, v)


# ============== 字面子串匹配(忽略空白差异,不做繁简/全半角归一) ==============
_WS_RE = re.compile(r"\s+")


def _normalize_for_match(s: str) -> str:
    """去掉所有空白,用于宽松子串匹配"""
    return _WS_RE.sub("", s or "")


def is_literal_substring(fragment: str, source: str) -> bool:
    """fragment 是否为 source 的字面子串(允许空白差异,不允许繁简互转)"""
    if not fragment or not source:
        return False
    return _normalize_for_match(fragment) in _normalize_for_match(source)


# ============== Markdown 表格识别 ==============
# 匹配形如 | --- | :---: | ---: | 的分隔行
_MD_TABLE_SEP_RE = re.compile(r"\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|")


def detect_support_type(evidence: str) -> str:
    """由 evidence 判定支撑类型;本轮不生成 img,默认 text,命中表格分隔行则 table"""
    if _MD_TABLE_SEP_RE.search(evidence or ""):
        return "table"
    return "text"


# ============== 动态出题数量 ==============
def decide_qa_count(
    md_length: int,
    chars_per_qa: int = 2000,
    min_n: int = 3,
    max_n: int = 15,
) -> int:
    """按 md 字符长度决定生成 QA 数量"""
    return max(min_n, min(max_n, md_length // chars_per_qa))


# ============== 分段 (按 markdown 标题边界切片) ==============
# 识别 markdown 一级~六级标题的行首标记
_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+\S")


def split_by_headings(text: str, target_size: int) -> list:
    """
    把文本按接近 target_size 字符切片, 优先在 markdown 标题处断开.
    策略:
      1. 按标题位置把全文分成若干"小节"
      2. 顺序拼接小节; 当下一小节加进来会让 chunk 超过 target_size*1.5 时, 切一刀
      3. 如果单个小节已经超过 target_size*1.5(罕见: 巨表格), 按字符数硬切
    保证所有 chunk 串起来能 "".join 回原文.

    返回值至少是 [text] (即使完全没有标题).
    """
    if not text:
        return [text]

    # 收集标题起点 (包括隐式的 0)
    heading_starts = [m.start() for m in _HEADING_RE.finditer(text)]
    if not heading_starts or heading_starts[0] != 0:
        heading_starts = [0] + heading_starts
    heading_starts.append(len(text))

    # 按标题切成小节
    sections = [
        text[heading_starts[i] : heading_starts[i + 1]]
        for i in range(len(heading_starts) - 1)
    ]

    hard_cap = int(target_size * 1.5)
    chunks = []
    buf = ""

    def flush():
        nonlocal buf
        if buf:
            chunks.append(buf)
            buf = ""

    for sec in sections:
        # 单个小节就巨大 -> 先冲掉当前 buf, 再把它按字符硬切
        if len(sec) > hard_cap:
            flush()
            for start in range(0, len(sec), target_size):
                chunks.append(sec[start : start + target_size])
            continue

        # 正常累加, 超过 target_size 就 flush
        if buf and len(buf) + len(sec) > target_size:
            flush()
        buf += sec

    flush()
    return chunks if chunks else [text]


# ============== 跨 chunk 去重 ==============
def _canon_question(q: str) -> str:
    """问题归一, 用于去重: 去空白 + 去结尾标点"""
    s = re.sub(r"\s+", "", q or "")
    return s.rstrip("?？。.,，；;!！")


def _dedup_by_question(records: list) -> tuple:
    """按归一化 question 去重, 首次出现保留. 返回 (去重后 list, 去重掉的条数)"""
    seen = set()
    kept = []
    for r in records:
        key = _canon_question(r.get("question", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept, len(records) - len(kept)


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


# ============== 公司关键词 (用于扁平/非标目录结构的兜底识别) ==============
# canonical -> [别名/繁体/英文关键词], 不区分大小写
COMPANY_KEYWORDS = [
    ("万通", ["万通", "萬通", "yflife", "yf life"]),
    ("保诚", ["保诚", "保誠", "prudential"]),
    ("富卫", ["富卫", "富衛", "fwd"]),
    ("永明", ["永明", "sun life", "sunlife"]),
    ("立桥", ["立桥", "立橋"]),
    ("宏利", ["宏利", "manulife"]),
    ("友邦", ["友邦", "aia"]),
    ("安盛", ["安盛", "axa"]),
    ("忠意", ["忠意", "generali"]),
    (
        "太保",
        [
            "中国太保",
            "中國太保",
            "太保寿险",
            "太保壽險",
            "太保",
            "世代悅享",
            "世代悦享",
            "世代",
            "颐养天年",
            "頤養天年",
            "颐养",
            "頤養",
            "臻享",
            "公司簡介",
            "公司简介",
            "cpic",
        ],
    ),
]
UNKNOWN_COMPANY = "未知保司"


def extract_company_from_path(md_path: Path, base_dir: str = "保司文件2.0") -> str:
    """
    从 md 文件路径里提取保司名 (即 base_dir 下的第一级子目录)
    例: .../保司文件2.0/万通/xxx/yyy.md -> 万通
    如果路径里找不到 base_dir, 退回到取父目录的父目录名
    """
    parts = md_path.parts
    if base_dir in parts:
        idx = parts.index(base_dir)
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # 兜底: 用父目录的父目录名 (因为结构通常是 保司名/文档名/xxx.md)
    if len(md_path.parents) >= 2:
        return md_path.parents[1].name
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
    # uuid 形如 f6385a61-99f1-4c89-95ef-947fae60dcc3 (8-4-4-4-12)
    cleaned = re.sub(
        r"\.pdf-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        "",
        parent_name,
        flags=re.IGNORECASE,
    )
    # 兜底: 如果父目录名也没意义(比如就叫 "万通" 这种, 即文件直接在保司目录下),
    # 回退用文件名 stem
    if not cleaned or cleaned == md_path.parents[1].name:
        cleaned = md_path.stem
    return cleaned


# ============== Prompt ==============
SYSTEM_PROMPT = """你是一个专业的保险知识 QA 数据标注员,基于真实的保险公司业务手册(核保/理赔/缴费/行政等)生成 QA 对,用于评估检索系统在保险知识库上的表现。

## 使用场景
这些 QA 是**保险公司新员工/业务员**在工作中查手册的问题。不是消费者闲聊、不是法律咨询,而是**业务员查手册时的快速查询**。

## 输入预处理规则(重要)
1. **忽略文本中所有图片链接**(形如 `![](...)` 或 `![alt](URL)`),禁止基于图片出题。
2. 仅基于**纯文本**和 **Markdown 表格**生成 QA;表格可正常利用。
3. **不要猜测原 PDF 页码**,不要输出任何页码字段。

## 问题书写要求
1. 必须以 user prompt 给出的保险公司中文名作为前缀,例: "万通缴费可以使用现金吗?"。
2. 风格: 简短书面、关键词式,禁止口语("咋办")或条款式("根据本规定...")。
3. 长度 8-25 个汉字为佳。

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

每条 QA 必须打一个分类标签, 从这 7 个里选一个最贴切的:
- 缴费: 缴费方式、币种、汇率、信用卡、行政费等
- 核保: 健康核保、疾病评估、核保要求、体检等
- 理赔: 理赔流程、所需材料、理赔时效等
- 产品: 产品条款、保障范围、保单内容、产品手册等
- 行政规则: 投保流程、保单变更、退保、保单服务、操作指引等
- 案例: 具体案例分析
- 一般查询: 不属于以上任何一类的通用问题

## 严格规则
1. answer 必须在原文中逐字可查(字面匹配,不可编造、不可繁简互转)
2. question 必须独立可理解,禁止"上述/本条款/前文/该项"等指代词
3. 同一文档多个问题主题要分散
4. 忽略图片链接;只利用纯文本和 Markdown 表格

## 输出格式
严格输出 JSON 数组,不要任何额外说明、不要 markdown 代码块。

### ⚠️ 字段完整性(最高优先级,违反者整条丢弃)
每个 JSON 对象**必须、且只能**包含以下 **5 个字段**, **一个都不能少,一个都不能多**:

1. `question`
2. `answer`
3. `category`
4. `prompt_type`   ← **最常被遗漏,在每条 QA 里都显式写出来!**
5. `evidence`       ← **也经常被遗漏,必须有!**

在生成每一条 QA 之前,请在心里**逐一检查**上面 5 个 key 是否都存在。
**强制模板**(占位符请替换为实际内容,键名和顺序保持不变):
```json
{
  "question": "...",
  "answer": "...",
  "category": "...",
  "prompt_type": "...",
  "evidence": "..."
}
```

禁止输出任何不在上面 5 个 key 之列的字段(如 `id`, `source`, `note` 等)。
禁止把 `prompt_type` 合并进 `category`,二者是独立字段。
"""


USER_PROMPT_TEMPLATE = """保险公司名: {company}
文档名称: {doc_filename}
默认业务分类(可参考,最终仍按内容判定): {default_category}

⚠️ 本文档为繁体中文,answer / evidence 必须原样保留繁体字,禁止繁→简转换。

文档内容(已是纯文本/Markdown;遇到 `![](...)` 图片链接请直接跳过):
```
{md_content}
```

请基于以上文档内容生成 {n} 个高质量 QA 对。

要求:
1. 每个问题必须以"{company}"开头
2. answer 必须从原文逐字摘取,保留繁体字与原格式(繁简体零容忍)
3. 多个问题主题分散
4. 直接输出 JSON 数组,不要任何额外文字、不要 markdown 代码块
5. category / prompt_type 的取值**必须用系统提示里给出的简体标准词**,不要自行翻成繁体或自造新词

## ⚠️ 最终示范 ⚠️
**你输出的 JSON 数组中, 每一个对象必须 100% 和下面这个例子长得一模一样, 包含完整的 5 个 key(question/answer/category/prompt_type/evidence), 一个都不能省略, 一个都不能多!**

```json
[
  {{
    "question": "{company}冷静期是多少天?",
    "answer": "冷靜期為21天",
    "category": "产品",
    "prompt_type": "数值查询",
    "evidence": "保單持有人可於收到保單後21天內行使冷靜期權利,此冷靜期為21天,期間可申請退保。"
  }},
  {{
    "question": "{company}投保人变更需要哪些文件?",
    "answer": "1. 更改保單擁有權申請書(A20表格)\\n2. 新舊保單持有人的身份證明文件副本\\n3. 住址證明",
    "category": "行政规则",
    "prompt_type": "列表枚举",
    "evidence": "客戶必須遞交下列文件:1. 更改保單擁有權申請書(A20表格);2. 新舊保單持有人的身份證明文件副本;3. 住址證明..."
  }}
]
```

注意示范里:
- `category` 写的是简体"产品"/"行政规则", 不是繁体"產品"/"行政規則"
- `prompt_type` 写的是简体"数值查询"/"列表枚举", 不是繁体"數值查詢"
- 5 个 key **全部出现**, 没有省略任何一个
- answer 里保留了原文的繁体字(冷靜期/擁有權/擔保)

现在请按同样的结构,基于前面的文档内容,输出 {n} 个 QA 的 JSON 数组。
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
                max_tokens=MAX_TOKENS,
                timeout=LLM_TIMEOUT,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            wait = 2**attempt
            print(f"[warn] 第 {attempt+1} 次调用失败: {e}, {wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"调用 LLM 失败 {MAX_RETRIES} 次: {last_err}")


VALID_CATEGORIES = {"缴费", "核保", "理赔", "产品", "行政规则", "案例", "一般查询"}


def validate_qa(item: dict, company: str, md_content: str) -> tuple:
    """基本字段校验, 返回 (是否有效, 原因)"""
    required = {"question", "answer", "category", "evidence"}
    if not required.issubset(item.keys()):
        return False, "字段缺失"
    if not all(isinstance(item[k], str) and item[k].strip() for k in required):
        return False, "字段为空或非字符串"

    # 繁体/近义标签归一到标准简体 (原地覆盖, 让下游写盘时已经是规范值)
    item["category"] = normalize_category(item["category"])
    item["prompt_type"] = normalize_prompt_type(item["prompt_type"])

    q = item["question"].strip()
    a = item["answer"].strip()

    if not q.startswith(company):
        return False, f"问题未以「{company}」开头"

    bad_refs = ["本条款", "本条", "上述", "该条", "如前所述", "前文", "该项"]
    if any(ref in q for ref in bad_refs):
        return False, "问题包含指代词"

    if item["category"] not in VALID_CATEGORIES:
        return False, f"非法业务分类「{item['category']}」"
    if item["prompt_type"] not in VALID_PROMPT_TYPES:
        return False, f"非法题型「{item['prompt_type']}」"

    if not is_literal_substring(a, md_content):
        return False, "answer 非原文字面子串(可能被改写或繁简互转)"

    return True, "ok"


def generate_qa_for_md(
    md_path: Path,
    client: OpenAI,
    qa_per_doc=None,
    base_dir: str = "保司文件2.0",
) -> dict:
    """
    处理单个 md 文件的核心入口, 供 batch_driver 复用.

    流程:
      - 短文档(< CHUNK_THRESHOLD): 整篇一次出 QA_PER_SHORT_DOC 条
      - 长文档: 在 markdown 标题边界切成 CHUNK_SIZE 的若干 chunk,
                每 chunk 独立调用 LLM 出 QA_PER_CHUNK 条, 最后按 question 去重合并

    qa_per_doc 显式传入时, 表示"整篇一次出 qa_per_doc 条", 绕过切片(兼容老入口).

    返回:
        {
            "status": "ok" | "skipped_short",
            "records": [...],
            "rejected": [...],
            "doc_info": {..., "n_chunks", "n_qa_requested", ...}
        }
    """
    md_content = md_path.read_text(encoding="utf-8")
    doc_length = len(md_content)

    if doc_length < MIN_CHARS:
        return {
            "status": "skipped_short",
            "records": [],
            "rejected": [],
            "doc_info": {
                "md_length": doc_length,
                "doc_filename": md_path.name,
            },
        }

    company = extract_company_from_path(md_path, base_dir=base_dir)
    doc_name = extract_doc_name_from_path(md_path)
    default_category = classify_doc(doc_name)

    # 决定处理模式
    if qa_per_doc is not None:
        # 显式指定 => 整篇一次出指定数量(保留旧行为, 兼容 CLI 的 --qa-per-doc)
        chunks = [md_content]
        per_chunk = qa_per_doc
        mode = "explicit"
    elif doc_length < CHUNK_THRESHOLD:
        # 短文档: 整篇一次
        chunks = [md_content]
        per_chunk = QA_PER_SHORT_DOC
        mode = "short"
    else:
        # 长文档: 切片
        chunks = split_by_headings(md_content, CHUNK_SIZE)
        per_chunk = QA_PER_CHUNK
        mode = "chunked"

    all_records = []
    all_rejected = []
    # 校验仍按"整文原文"做字面匹配, 避免片段边界把 answer 跨 chunk 误杀
    for i, chunk in enumerate(chunks, 1):
        user_prompt = USER_PROMPT_TEMPLATE.format(
            company=company,
            doc_filename=doc_name,
            default_category=default_category,
            md_content=chunk,
            n=per_chunk,
        )
        try:
            raw = call_llm(client, SYSTEM_PROMPT, user_prompt)
            items = extract_json_array(raw)
        except Exception as e:
            # 单个 chunk 失败只丢这个 chunk, 不影响别的 chunk
            all_rejected.append(
                {
                    "reason": f"chunk_{i}/{len(chunks)} 调用失败: {type(e).__name__}: {str(e)[:200]}",
                    "question": "",
                    "answer": "",
                    "evidence": chunk[:120],
                    "category": None,
                    "prompt_type": None,
                }
            )
            continue

        for item in items:
            ok, reason = validate_qa(item, company, md_content)
            if not ok:
                all_rejected.append(
                    {
                        "reason": reason,
                        "question": (item.get("question") or "")[:120],
                        "answer": (item.get("answer") or "")[:300],
                        "evidence": (item.get("evidence") or "")[:300],
                        "category": item.get("category"),
                        "prompt_type": item.get("prompt_type"),
                    }
                )
                continue
            evidence = item["evidence"].strip()
            all_records.append(
                {
                    "category": item["category"].strip(),
                    "prompt_type": item["prompt_type"].strip(),
                    "question": item["question"].strip(),
                    "answers": item["answer"].strip(),
                    "document": f"{doc_name}.pdf",
                    "page": "N/A",
                    "support_type": detect_support_type(evidence),
                    "source_company": company,
                    "source_chunk_id": f"{company}_{doc_name}_c{i}",
                    "evidence": evidence,
                }
            )

    # 跨 chunk 去重 (按 question, 首次出现保留)
    deduped, dup_count = _dedup_by_question(all_records)

    return {
        "status": "ok",
        "records": deduped,
        "rejected": all_rejected,
        "doc_info": {
            "company": company,
            "doc_name": doc_name,
            "doc_filename": md_path.name,
            "md_length": doc_length,
            "default_category": default_category,
            "mode": mode,
            "n_chunks": len(chunks),
            "n_qa_per_chunk": per_chunk,
            "n_qa_requested": per_chunk * len(chunks),
            "n_duplicates_removed": dup_count,
        },
    }


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

    # 安全检查: 输出文件已存在
    output_path = Path(args.output)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[警告] 输出文件 {args.output} 已存在且非空。")
        print(f"  - 如果是批处理累加, 输入 y 继续")
        print(f"  - 如果是同一文档重跑, 输入 n 后先删除旧文件")
        choice = input("继续追加? (y/n): ").strip().lower()
        if choice != "y":
            print(f"已退出。删除旧文件: Remove-Item {args.output}")
            return

    # 显式传了 --qa-per-doc 才固定 n, 否则按文档长度动态决定
    qa_per_doc = args.qa_per_doc if args.qa_per_doc != QA_PER_DOC else None

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print("正在调用 LLM 生成 QA, 请稍等...")
    try:
        result = generate_qa_for_md(
            md_path, client, qa_per_doc=qa_per_doc, base_dir=args.base_dir
        )
    except Exception as e:
        print(f"[错误] 生成失败: {e}")
        return

    if result["status"] == "skipped_short":
        info = result["doc_info"]
        print(
            f"[跳过] 文档过短 ({info['md_length']} 字符 < {MIN_CHARS}): "
            f"{info['doc_filename']}"
        )
        return

    info = result["doc_info"]
    print(f"文件: {info['doc_filename']}  (实际路径: {md_path})")
    print(f"文档名: {info['doc_name']}")
    print(f"保司: {info['company']}")
    print(f"预设分类: {info['default_category']}")
    print(f"内容长度: {info['md_length']} 字符")
    print(f"请求 QA 数量: {info['n_qa_requested']}")
    print()

    qa_id = args.start_id
    written = 0
    with open(args.output, "a", encoding="utf-8") as fout:
        for rec in result["records"]:
            rec_with_id = {"qa_id": f"qa_{qa_id:05d}", **rec}
            fout.write(json.dumps(rec_with_id, ensure_ascii=False) + "\n")
            qa_id += 1
            written += 1

    rejected = result["rejected"]
    print()
    print(f"=== 完成 ===")
    print(f"成功写入: {written} 条")
    print(f"被过滤: {len(rejected)} 条")
    for rej in rejected:
        print(f"  [skip] {rej['reason']}: {rej['question'][:50]}")
        print(f"         answer 片段: {rej['answer'][:80]}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
