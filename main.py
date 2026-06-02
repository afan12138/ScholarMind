"""
ScholarMind RAG API - PostgreSQL 版本
数据库表：users, papers, notes, analysis_cache, action_logs
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from openai import OpenAI
from sqlalchemy import (
    create_engine, Column, String, Integer, Text, Boolean,
    DateTime, ForeignKey, BigInteger, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from datetime import datetime, timedelta
import jwt, bcrypt, json, math, re, os
from collections import Counter
from typing import Optional

# ══════════════════════════════════════════
#  配置
# ══════════════════════════════════════════

DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("DATABASE_PRIVATE_URL")
    or os.environ.get("DATABASE_PUBLIC_URL")
    or "postgresql://postgres:postgres@localhost:5432/scholarmind"
)
# Railway 有时给出 postgres:// 前缀，psycopg3 dialect 用 postgresql+psycopg://
DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

JWT_SECRET = os.environ.get("JWT_SECRET", "scholarmind-super-secret-2025")
JWT_EXPIRE_DAYS = 30

DEEPSEEK_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    "sk-47caf8ee0df84ba8a712ae60af574e4a"
)

# ══════════════════════════════════════════
#  数据库
# ══════════════════════════════════════════

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(32), unique=True, index=True, nullable=False)
    email      = Column(String(120), unique=True, index=True, nullable=False)
    password   = Column(String(128), nullable=False)         # bcrypt hash
    role       = Column(String(16), default="user")          # user | admin
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    papers  = relationship("Paper", back_populates="owner", cascade="all, delete-orphan")
    notes   = relationship("Note",  back_populates="owner", cascade="all, delete-orphan")
    caches  = relationship("AnalysisCache", back_populates="owner", cascade="all, delete-orphan")
    logs    = relationship("ActionLog",     back_populates="user",  cascade="all, delete-orphan")


class Paper(Base):
    __tablename__ = "papers"
    id         = Column(String(64), primary_key=True)        # 前端生成的 id
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name       = Column(String(256), nullable=False)
    size       = Column(BigInteger, default=0)
    file_type  = Column(String(16), default="pdf")
    content    = Column(Text, nullable=True)                 # 提取的文字
    preview    = Column(String(500), nullable=True)
    added_at   = Column(DateTime, default=datetime.utcnow)
    meta       = Column(JSON, nullable=True)                 # _meta: authors, doi, etc.
    from_search = Column(Boolean, default=False)

    owner  = relationship("User", back_populates="papers")
    notes  = relationship("Note", back_populates="paper", cascade="all, delete-orphan")
    caches = relationship("AnalysisCache", back_populates="paper", cascade="all, delete-orphan")


class Note(Base):
    __tablename__ = "notes"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    paper_id   = Column(String(64), ForeignKey("papers.id"), nullable=False, index=True)
    content    = Column(Text, default="")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User",  back_populates="notes")
    paper = relationship("Paper", back_populates="notes")


class AnalysisCache(Base):
    __tablename__ = "analysis_cache"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    paper_id   = Column(String(64), ForeignKey("papers.id"), nullable=False, index=True)
    cache_type = Column(String(32), nullable=False)          # analysis|keywords|score|correct|mindmap
    data       = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User",  back_populates="caches")
    paper = relationship("Paper", back_populates="caches")


class ActionLog(Base):
    __tablename__ = "action_logs"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=True)
    action     = Column(String(64), nullable=False)
    detail     = Column(Text, nullable=True)
    ip         = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="logs")


# RAG 向量存储（内存，重启丢失，仅用于问答）
doc_store: dict = {}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """建表 + 插入默认 admin 和 demo 账号"""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if not db.query(User).filter_by(username="admin").first():
            db.add(User(
                username="admin",
                email="admin@scholar.com",
                password=bcrypt.hashpw(b"admin888", bcrypt.gensalt()).decode(),
                role="admin",
            ))
        if not db.query(User).filter_by(username="demo").first():
            db.add(User(
                username="demo",
                email="demo@scholar.com",
                password=bcrypt.hashpw(b"demo123", bcrypt.gensalt()).decode(),
                role="user",
            ))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


# ══════════════════════════════════════════
#  JWT
# ══════════════════════════════════════════

security = HTTPBearer(auto_error=False)


def create_token(user_id: int, username: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "token 已过期，请重新登录")
    except Exception:
        raise HTTPException(401, "无效的 token")


def current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    if not creds:
        raise HTTPException(401, "未登录")
    payload = decode_token(creds.credentials)
    user = db.query(User).filter_by(id=payload["sub"]).first()
    if not user:
        raise HTTPException(401, "用户不存在")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "需要管理员权限")
    return user


# ══════════════════════════════════════════
#  RAG 工具函数（原有逻辑不变）
# ══════════════════════════════════════════

deepseek = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")


def split_text(text: str, chunk_size=500, overlap=50) -> list:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def tokenize(text: str) -> list:
    return re.findall(r'[a-zA-Z]+|[\u4e00-\u9fff]', text.lower())


def bm25_score(query_tokens: list, chunk_tokens: list, avg_len: float) -> float:
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
    index_lines = "\n".join(
        f"[{i+1}] {c['text'][:120].strip()}..."
        for i, c in enumerate(candidates)
    )
    prompt = f"""下面是论文片段列表：\n{index_lines}\n\n用户问题：{question}\n\n请选出最相关的片段编号，只返回编号，英文逗号分隔，最多{top_k}个。示例：3,7,12"""
    try:
        resp = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30, temperature=0
        )
        raw = resp.choices[0].message.content.strip()
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
        print(f"AI rerank 失败: {e}")
    return candidates[:top_k]


# ══════════════════════════════════════════
#  FastAPI App
# ══════════════════════════════════════════

app = FastAPI(title="ScholarMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    print("✅ 数据库初始化完成")


@app.get("/")
def health():
    return {"status": "ScholarMind API running", "db": "postgresql"}


# ══════════════════════════════════════════
#  Auth 接口
# ══════════════════════════════════════════

class RegisterReq(BaseModel):
    username: str
    email: str
    password: str

class LoginReq(BaseModel):
    username: str      # 用户名或邮箱
    password: str


@app.post("/api/auth/register")
def register(req: RegisterReq, request: Request, db: Session = Depends(get_db)):
    if len(req.username) < 4 or len(req.username) > 32:
        raise HTTPException(400, "用户名长度应为4-32位")
    if len(req.password) < 6:
        raise HTTPException(400, "密码不能少于6位")
    if db.query(User).filter_by(username=req.username).first():
        raise HTTPException(409, "用户名已存在")
    if db.query(User).filter_by(email=req.email).first():
        raise HTTPException(409, "邮箱已被注册")

    hashed = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    user = User(username=req.username, email=req.email, password=hashed, role="user")
    db.add(user)
    db.commit()
    db.refresh(user)

    _log(db, user.id, "register", f"新用户注册：{user.username}", request)
    token = create_token(user.id, user.username, user.role)
    return {"token": token, "username": user.username, "email": user.email, "role": user.role}


@app.post("/api/auth/login")
def login(req: LoginReq, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        (User.username == req.username) | (User.email == req.username)
    ).first()
    if not user or not bcrypt.checkpw(req.password.encode(), user.password.encode()):
        raise HTTPException(401, "用户名或密码错误")

    user.last_login = datetime.utcnow()
    db.commit()
    _log(db, user.id, "login", f"用户登录", request)
    token = create_token(user.id, user.username, user.role)
    return {"token": token, "username": user.username, "email": user.email, "role": user.role}


@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "username": user.username, "email": user.email, "role": user.role, "created_at": user.created_at}


# ══════════════════════════════════════════
#  文库接口
# ══════════════════════════════════════════

class PaperAddReq(BaseModel):
    id: str
    name: str
    size: int
    content: Optional[str] = ""
    file_type: Optional[str] = "search"
    meta: Optional[dict] = None
    from_search: Optional[bool] = False


@app.get("/api/library")
def get_library(user: User = Depends(current_user), db: Session = Depends(get_db)):
    papers = db.query(Paper).filter_by(user_id=user.id).order_by(Paper.added_at.desc()).all()
    return [_paper_to_dict(p) for p in papers]


@app.post("/api/library")
def add_paper(req: PaperAddReq, user: User = Depends(current_user), db: Session = Depends(get_db)):
    is_from_search = req.from_search or False
    safe_content = "" if is_from_search else (req.content or "")[:80000]
    safe_preview = (req.meta.get("title", req.name) if req.meta else req.name)[:300]                    if is_from_search else safe_content[:300]

    existing = db.query(Paper).filter_by(id=req.id, user_id=user.id).first()
    if existing:
        if not is_from_search:
            existing.content = safe_content
            existing.preview = safe_preview
        existing.meta = req.meta
        db.commit()
        return {"status": "updated", "id": req.id}

    # 同名合并只针对普通上传论文
    if not is_from_search:
        same_name = db.query(Paper).filter_by(name=req.name, user_id=user.id).first()
        if same_name:
            same_name.content = safe_content
            same_name.preview = safe_preview
            db.commit()
            return {"status": "updated", "id": same_name.id}

    paper = Paper(
        id=req.id,
        user_id=user.id,
        name=req.name,
        size=req.size,
        file_type=req.file_type or "search",
        content=safe_content,
        preview=safe_preview,
        meta=req.meta,
        from_search=is_from_search,
    )
    db.add(paper)
    db.commit()
    return {"status": "added", "id": req.id}


@app.delete("/api/library/{paper_id}")
def delete_paper(paper_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    paper = db.query(Paper).filter_by(id=paper_id, user_id=user.id).first()
    if not paper:
        raise HTTPException(404, "论文不存在")
    db.delete(paper)
    db.commit()
    return {"status": "deleted"}


class PaperDeleteReq(BaseModel):
    paper_id: str


@app.post("/api/library/delete")
def delete_paper_by_post(req: PaperDeleteReq, user: User = Depends(current_user), db: Session = Depends(get_db)):
    paper = db.query(Paper).filter_by(id=req.paper_id, user_id=user.id).first()
    if not paper:
        raise HTTPException(404, "论文不存在")
    db.delete(paper)
    db.commit()
    return {"status": "deleted"}


# ══════════════════════════════════════════
#  笔记接口
# ══════════════════════════════════════════

class NoteReq(BaseModel):
    paper_id: str
    content: str


@app.get("/api/notes")
def get_notes(user: User = Depends(current_user), db: Session = Depends(get_db)):
    notes = db.query(Note).filter_by(user_id=user.id).all()
    return {n.paper_id: n.content for n in notes}


@app.post("/api/notes")
def save_note(req: NoteReq, user: User = Depends(current_user), db: Session = Depends(get_db)):
    note = db.query(Note).filter_by(user_id=user.id, paper_id=req.paper_id).first()
    if note:
        note.content = req.content
        note.updated_at = datetime.utcnow()
    else:
        note = Note(user_id=user.id, paper_id=req.paper_id, content=req.content)
        db.add(note)
    db.commit()
    return {"status": "saved"}


@app.delete("/api/notes/{paper_id}")
def delete_note(paper_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.query(Note).filter_by(user_id=user.id, paper_id=paper_id).delete()
    db.commit()
    return {"status": "deleted"}


# ══════════════════════════════════════════
#  分析缓存接口
# ══════════════════════════════════════════

class CacheSetReq(BaseModel):
    paper_id: str
    cache_type: str
    data: dict


@app.get("/api/cache/{paper_id}/{cache_type}")
def get_cache(paper_id: str, cache_type: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    row = db.query(AnalysisCache).filter_by(
        user_id=user.id, paper_id=paper_id, cache_type=cache_type
    ).first()
    if not row:
        raise HTTPException(404, "缓存不存在")
    return {"data": row.data}


@app.post("/api/cache")
def set_cache(req: CacheSetReq, user: User = Depends(current_user), db: Session = Depends(get_db)):
    row = db.query(AnalysisCache).filter_by(
        user_id=user.id, paper_id=req.paper_id, cache_type=req.cache_type
    ).first()
    if row:
        row.data = req.data
        row.updated_at = datetime.utcnow()
    else:
        row = AnalysisCache(
            user_id=user.id,
            paper_id=req.paper_id,
            cache_type=req.cache_type,
            data=req.data,
        )
        db.add(row)
    db.commit()
    return {"status": "ok"}


@app.delete("/api/cache/{paper_id}/{cache_type}")
def delete_cache(paper_id: str, cache_type: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.query(AnalysisCache).filter_by(
        user_id=user.id, paper_id=paper_id, cache_type=cache_type
    ).delete()
    db.commit()
    return {"status": "deleted"}


@app.delete("/api/cache/{paper_id}")
def delete_paper_cache(paper_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.query(AnalysisCache).filter_by(user_id=user.id, paper_id=paper_id).delete()
    db.commit()
    return {"status": "deleted"}


# ══════════════════════════════════════════
#  RAG 接口（原有逻辑）
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


@app.post("/api/index")
async def index_paper(req: IndexRequest):
    key = f"{req.username}_{req.paper_id}"
    chunks = split_text(req.content)
    if not chunks:
        raise HTTPException(400, "内容为空")
    doc_store[key] = [
        {"text": chunk, "tokens": tokenize(chunk),
         "paper_id": req.paper_id, "paper_name": req.paper_name, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]
    print(f"✅ 索引完成: {req.paper_name}，共 {len(chunks)} 片段")
    return {"status": "ok", "chunks": len(chunks)}


@app.post("/api/qa")
async def rag_qa(req: QARequest):
    candidates = []
    for key, chunks in doc_store.items():
        if not key.startswith(req.username):
            continue
        if req.paper_id and f"_{req.paper_id}" not in key:
            continue
        candidates.extend(chunks)

    if not candidates:
        raise HTTPException(404, "请先索引论文，重新上传后再提问")

    query_tokens = tokenize(req.question)
    avg_len = sum(len(c["tokens"]) for c in candidates) / max(len(candidates), 1)
    bm25_top = sorted(
        candidates,
        key=lambda c: bm25_score(query_tokens, c["tokens"], avg_len),
        reverse=True
    )[:20]

    scored = ai_rerank(req.question, bm25_top, req.top_k) if len(bm25_top) > req.top_k else bm25_top[:req.top_k]

    context = "\n\n".join(
        f"【片段{i+1}·{c['paper_name']}·原文第{c['chunk_index']+1}段】\n{c['text']}"
        for i, c in enumerate(scored)
    )
    system_prompt = f"""你是专业学术论文分析助手。以下是从论文中精确检索到的最相关片段，请严格基于这些内容回答问题。

{context}

回答要求：
1. 只引用上述片段中的信息，不要杜撰
2. 回答中用【片段N】标注引用来源
3. 用中文回答，专业准确"""

    def generate():
        sources = [{"name": c["paper_name"], "chunk": c["chunk_index"] + 1} for c in scored]
        yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"
        try:
            stream = deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": req.question}
                ],
                stream=True, temperature=0.3, max_tokens=2000
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


# ══════════════════════════════════════════
#  管理员接口
# ══════════════════════════════════════════

@app.get("/api/admin/stats")
def admin_stats(admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    total_users  = db.query(User).count()
    total_papers = db.query(Paper).count()
    total_notes  = db.query(Note).filter(Note.content != "").count()
    admin_count  = db.query(User).filter_by(role="admin").count()
    return {
        "total_users": total_users,
        "total_papers": total_papers,
        "total_notes": total_notes,
        "admin_count": admin_count,
    }


@app.get("/api/admin/users")
def admin_list_users(admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    result = []
    for u in users:
        paper_count = db.query(Paper).filter_by(user_id=u.id).count()
        note_count  = db.query(Note).filter_by(user_id=u.id).count()
        result.append({
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_login": u.last_login.isoformat() if u.last_login else None,
            "paper_count": paper_count,
            "note_count": note_count,
        })
    return result


class AdminAddUserReq(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "user"


@app.post("/api/admin/users")
def admin_add_user(req: AdminAddUserReq, admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    if db.query(User).filter_by(username=req.username).first():
        raise HTTPException(409, "用户名已存在")
    if db.query(User).filter_by(email=req.email).first():
        raise HTTPException(409, "邮箱已存在")
    hashed = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    user = User(username=req.username, email=req.email, password=hashed, role=req.role or "user")
    db.add(user)
    db.commit()
    return {"status": "created", "id": user.id}


@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(user_id: int, admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    if user_id == admin.id:
        raise HTTPException(400, "不能删除自己")
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(404, "用户不存在")
    db.delete(user)
    db.commit()
    return {"status": "deleted"}


class AdminUpdateUserReq(BaseModel):
    role: Optional[str] = None
    password: Optional[str] = None


@app.patch("/api/admin/users/{user_id}")
def admin_update_user(user_id: int, req: AdminUpdateUserReq, admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(404, "用户不存在")
    if req.role:
        user.role = req.role
    if req.password:
        user.password = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    db.commit()
    return {"status": "updated"}


@app.get("/api/admin/logs")
def admin_logs(admin: User = Depends(admin_user), db: Session = Depends(get_db), limit: int = 100):
    logs = db.query(ActionLog).order_by(ActionLog.created_at.desc()).limit(limit).all()
    return [{
        "id": l.id,
        "user_id": l.user_id,
        "username": l.user.username if l.user else "—",
        "action": l.action,
        "detail": l.detail,
        "ip": l.ip,
        "created_at": l.created_at.isoformat(),
    } for l in logs]


@app.get("/api/admin/library")
def admin_library(admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    papers = db.query(Paper).order_by(Paper.added_at.desc()).limit(200).all()
    return [{
        "id": p.id,
        "name": p.name,
        "username": p.owner.username,
        "size": p.size,
        "file_type": p.file_type,
        "added_at": p.added_at.isoformat(),
    } for p in papers]


# ══════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════

def _paper_to_dict(p: Paper) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "size": p.size,
        "file_type": p.file_type,
        "content": p.content or "",
        "preview": p.preview or "",
        "added_at": int(p.added_at.timestamp() * 1000) if p.added_at else 0,
        "meta": p.meta,
        "from_search": p.from_search,
        "type": p.file_type,
    }


def _log(db: Session, user_id: Optional[int], action: str, detail: str, request: Request = None):
    ip = None
    if request:
        ip = request.client.host if request.client else None
    db.add(ActionLog(user_id=user_id, action=action, detail=detail, ip=ip))
    db.commit()
