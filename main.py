from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import json, math, re
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

deepseek = OpenAI(
    api_key="sk-47caf8ee0df84ba8a712ae60af574e4a",
    base_url="https://api.deepseek.com"
)

# 内存存储：{ "username_paperid": [ {text, tokens, ...}, ... ] }
doc_store: dict = {}


# ══════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════

def split_text(text: str, chunk_size=500, overlap=50) -> list:
    """按字符切片，带重叠"""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def tokenize(text: str) -> list:
    """中英文分词"""
    return re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]', text.lower())


def bm25_score(query_tokens: list, chunk_tokens: list, avg_len: float) -> float:
    """BM25 打分，用于快速初筛"""
    k1, b = 1.5, 0.75
    chunk_tf = Counter(chunk_tokens)
    chunk_len = len(chunk_tokens)
    score = 0.0
    for token in set(query_tokens):
        if token in chunk_tf:
            freq = chunk_tf[token]
            idf = math.log(2)
            num = freq * (k1 + 1)
            den = freq + k1 * (1 - b + b * chunk_len / (avg_len + 1))
            score += idf * num / den
    return score


def ai_rerank(question: str, candidates: list, top_k: int = 5) -> list:
    """
    用 DeepSeek Chat 做语义重排：
    先把候选片段的摘要发给 AI，让它选出最相关的编号，
    再按编号取对应片段。
    """
    # 每个片段取前 120 字作为摘要，避免 prompt 过长
    index_lines = "\n".join(
        f"[{i+1}] {c['text'][:120].strip()}..."
        for i, c in enumerate(candidates)
    )

    prompt = f"""下面是论文片段列表，格式为[编号] 片段内容摘要：

{index_lines}

用户问题：{question}

请从上面的片段中选出与问题最相关的片段编号，只返回编号，用英文逗号分隔，最多{top_k}个。
示例输出：3,7,12
不要输出任何其他内容。"""

    try:
        resp = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0
        )
        raw = resp.choices[0].message.content.strip()
        # 解析编号，容错处理
        indices = []
        for x in re.split(r'[,，\s]+', raw):
            x = x.strip().lstrip('[').rstrip(']')
            if x.isdigit():
                idx = int(x) - 1
                if 0 <= idx < len(candidates):
                    indices.append(idx)
        if indices:
            return [candidates[i] for i in indices[:top_k]]
    except Exception as e:
        print(f"AI rerank 失败，回退 BM25: {e}")

    # AI 失败时降级回 BM25 结果（已排好序）
    return candidates[:top_k]


# ══════════════════════════════════════════
#  数据模型
# ══════════════════════════════════════════

class IndexRequest(BaseModel):
    paper_id: str
    paper_name: str
    content: str
    username: str


class QARequest(BaseModel):
    question: str
    paper_id: str
    username: str
    paper_name: str = ""
    top_k: int = 5


# ══════════════════════════════════════════
#  接口
# ══════════════════════════════════════════

@app.get("/")
def health():
    return {"status": "ScholarMind RAG API running"}


@app.options("/api/index")
async def options_index():
    return {}


@app.options("/api/qa")
async def options_qa():
    return {}


@app.post("/api/index")
async def index_paper(req: IndexRequest):
    """接收论文内容，切片并存入内存"""
    key = f"{req.username}_{req.paper_id}"
    chunks = split_text(req.content)
    if not chunks:
        raise HTTPException(400, "内容为空")

    doc_store[key] = [
        {
            "text": chunk,
            "tokens": tokenize(chunk),
            "paper_id": req.paper_id,
            "paper_name": req.paper_name,
            "chunk_index": i
        }
        for i, chunk in enumerate(chunks)
    ]
    print(f"✅ 索引完成: {req.paper_name}，共 {len(chunks)} 片段")
    return {"status": "ok", "chunks": len(chunks)}


@app.post("/api/qa")
async def rag_qa(req: QARequest):
    """RAG 问答：BM25 初筛 → AI 语义重排 → DeepSeek 生成回答"""

    # 1. 收集候选片段
    candidates = []
    for key, chunks in doc_store.items():
        if not key.startswith(req.username):
            continue
        if req.paper_id and f"_{req.paper_id}" not in key:
            continue
        candidates.extend(chunks)

    if not candidates:
        raise HTTPException(404, "请先索引论文，重新上传后再提问")

    # 2. BM25 初筛：取相关性最高的前 20 个片段（减少 AI rerank 的输入量）
    query_tokens = tokenize(req.question)
    avg_len = sum(len(c["tokens"]) for c in candidates) / max(len(candidates), 1)

    bm25_top = sorted(
        candidates,
        key=lambda c: bm25_score(query_tokens, c["tokens"], avg_len),
        reverse=True
    )[:20]

    # 3. AI 语义重排：从 20 个里选出最相关的 top_k 个
    if len(bm25_top) > req.top_k:
        scored = ai_rerank(req.question, bm25_top, req.top_k)
    else:
        scored = bm25_top[:req.top_k]

    # 4. 拼装上下文
    context = "\n\n".join(
        f"【片段{i+1}·{c['paper_name']}·原文第{c['chunk_index']+1}段】\n{c['text']}"
        for i, c in enumerate(scored)
    )

    system_prompt = f"""你是专业学术论文分析助手。
以下是从论文中精确检索到的最相关片段，请严格基于这些内容回答问题。

{context}

回答要求：
1. 只引用上述片段中的信息，不要杜撰
2. 回答中用【片段N】标注引用来源
3. 用中文回答，专业准确"""

    # 5. 流式生成回答
    def generate():
        # 先发来源信息
        sources = [
            {"name": c["paper_name"], "chunk": c["chunk_index"] + 1}
            for c in scored
        ]
        yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"

        # 再流式生成回答
        try:
            stream = deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": req.question}
                ],
                stream=True,
                temperature=0.3,
                max_tokens=2000
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'text': delta}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'text': f'生成失败：{str(e)}'}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
