import os
import re
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from pytube import YouTube

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Please set OPENAI_API_KEY environment variable")
    # 不直接抛异常，让 / 路由仍能打开页面；调用到 AI 时再报错
    pass

client = None
def get_openai_client():
    global client
    if client is None:
        if not OPENAI_API_KEY:
            print("11111")
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        print(OPENAI_API_KEY)
        print("8888")
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("9999")
    return client

app = FastAPI(title="Basic AI Chatbot + YouTube", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 本地开发先放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Data models
# ----------------------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    kind: str = "text"  # text | video_summary | video_qa
    video: Optional[Dict[str, Any]] = None
    citations: Optional[List[Dict[str, Any]]] = None

class VideoQARequest(BaseModel):
    video_id: str
    question: str

# ----------------------------
# Simple in-memory cache
# ----------------------------
TRANSCRIPT_CACHE: Dict[str, Dict[str, Any]] = {}   # video_id -> {title, lang, segments, text, ts}
# 速率限制（超简单按 IP 计数：每 60s 内最多 30 次）
RATE_BUCKET: Dict[str, List[float]] = defaultdict(list)
RATE_WINDOW_SEC = 60
RATE_LIMIT = 30

# ----------------------------
# Helpers
# ----------------------------
YOUTUBE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})",
    re.IGNORECASE
)

def is_youtube_url(text: str) -> bool:
    return YOUTUBE_RE.search(text) is not None

def extract_video_id(text: str) -> Optional[str]:
    m = YOUTUBE_RE.search(text.strip())
    return m.group(1) if m else None

def get_video_title(video_id: str) -> Optional[str]:
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        return yt.title
    except Exception:
        return None

def get_transcript_segments(video_id: str) -> List[Dict[str, Any]]:
    """
    return segments: list of {text, start, duration}
    优先中文/英文字幕
    """
    preferred = ["zh-Hans", "zh-Hant", "zh-CN", "zh-TW", "en"]
    ytt = YouTubeTranscriptApi()
    try:
        fetched = ytt.fetch(video_id, languages=preferred)
        return fetched.to_raw_data()
    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="Uploader disabled subtitles.")
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail="No transcript in requested languages.")
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video unavailable/private.")
    except Exception as e:
        # 典型：被 YouTube 同意页/风控挡住
        raise HTTPException(status_code=502, detail=f"Transcript fetch failed: {type(e).__name__}: {e}")

def human_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def join_text(segments: List[Dict[str, Any]]) -> str:
    return "\n".join(f"[{human_time(seg['start'])}] {seg['text']}" for seg in segments)

def build_summary(text: str, title: Optional[str]) -> str:
    client = get_openai_client()
    sys = "You are a helpful assistant. Summarize the video transcript into 3–4 concise sentences. Avoid fluff."
    user = f"Title: {title or 'N/A'}\nTranscript (truncated if too long):\n{text[:12000]}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.2,
        max_tokens=220,
    )
    return resp.choices[0].message.content.strip()

def select_relevant_segments(question: str, segments: List[Dict[str, Any]], topk: int = 10) -> List[Dict[str, Any]]:
    """
    轻量检索：按关键词命中 + 简单分数（长度惩罚）。
    """
    q = question.lower()
    scores = []
    for i, seg in enumerate(segments):
        t = seg["text"].lower()
        score = 0
        # 命中词计分（可再加分词/同义词）
        for token in set(re.findall(r"\w+", q)):
            if token and token in t:
                score += 1
        # 轻微惩罚段落过长
        score -= len(seg["text"]) / 3000.0
        if score > 0:
            scores.append((score, i))
    scores.sort(key=lambda x: x[0], reverse=True)
    idxs = [i for _, i in scores[:topk]]
    idxs = sorted(set(idxs))
    return [segments[i] for i in idxs]


# ---- New helpers for context selection ----
def spread_segments(segments: List[Dict[str, Any]], maxn: int = 40) -> List[Dict[str, Any]]:
    """
    Evenly sample up to maxn segments across the whole video to give broad coverage.
    """
    n = len(segments)
    if n <= maxn:
        return segments
    step = max(1, n // maxn)
    out = segments[::step]
    return out[:maxn]

def pick_time_windows(segments: List[Dict[str, Any]], timestr_list: List[str], neighbors: int = 6) -> List[Dict[str, Any]]:
    """
    For each timestamp in timestr_list (e.g., ['5:30', '57:32']), find the single nearest segment
    (by absolute difference of `start`), then include a small neighborhood of segments around it
    to preserve continuity (±neighbors segments). This yields more complete, contiguous context
    that actually spans the asked time (including slightly after it), instead of mixing distant snippets.
    """
    def to_sec(ts: str) -> int:
        parts = ts.split(":")
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(s)
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        return 0

    if not segments:
        return []

    targets = [to_sec(t) for t in timestr_list if ":" in t]
    idxs = set()

    # map each timestamp to closest index
    for t in targets:
        closest_i = min(range(len(segments)), key=lambda i: abs(int(segments[i].get("start", 0)) - t))
        lo = max(0, closest_i - neighbors)
        hi = min(len(segments), closest_i + neighbors + 1)
        for i in range(lo, hi):
            idxs.add(i)

    # preserve original order
    ordered = [segments[i] for i in sorted(idxs)]
    return ordered

def answer_question(question: str, title: Optional[str], segments: List[Dict[str, Any]]) -> str:
    client = get_openai_client()
    context = "\n".join(f"[{human_time(s['start'])}] {s['text']}" for s in segments)
    sys = (
        "You are answering questions about a YouTube video using provided transcript text. "
        "Use only the text you are given (it may be the full transcript or selected excerpts). "
        "Do NOT claim you lack access to the full transcript; if context seems partial, preface with 'Based on the provided excerpts,'. "
        "Cite timestamps inline like [mm:ss]. If the answer is not present in the provided text, say you don't know. "
        "When timestamps are provided in the question, prioritize citing the nearest transcript snippets to those times."
    )
    user = f"Video Title: {title or 'N/A'}\nExcerpts:\n{context}\n\nQuestion: {question}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.2,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip()

def rate_limit_ok(ip: str) -> bool:
    now = time.time()
    bucket = RATE_BUCKET[ip]
    # 清理窗口外
    while bucket and now - bucket[0] > RATE_WINDOW_SEC:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT:
        return False
    bucket.append(now)
    return True

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def index():
    return FileResponse("templates/index.html")

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    # 极简限流（可关掉）
    print("11111")
    ip = request.client.host if request.client else "unknown"
    if not rate_limit_ok(ip):
        raise HTTPException(status_code=429, detail="Too many requests, please slow down.")
    print("222222")

    msg = req.message.strip()
    print("333333")

    # 场景 1：识别 YouTube 链接 -> 拉字幕 -> 总结
    if is_youtube_url(msg):
        print("444444")

        video_id = extract_video_id(msg)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL.")
        print("4445")
        if video_id not in TRANSCRIPT_CACHE:
            print("44455")
            print(video_id)
            segments = get_transcript_segments(video_id)
            print("44455")

            title = get_video_title(video_id)
            text = join_text(segments)
            TRANSCRIPT_CACHE[video_id] = {
                "title": title,
                "segments": segments,
                "text": text,
                "ts": time.time(),
            }
        print("4446")

        data = TRANSCRIPT_CACHE[video_id]
        try:
            summary = build_summary(data["text"], data["title"])
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Summarization failed: {type(e).__name__}")

        return ChatResponse(
            reply=summary,
            kind="video_summary",
            video={"id": video_id, "title": data["title"]},
            citations=None,
        )

    # 普通聊天（任务 1）
    try:
        print("55555")
        c = get_openai_client()
        print("6666")

        completion = c.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a concise, helpful assistant."},
                {"role":"user","content": msg},
            ],
            temperature=0.6,
            max_tokens=200,
        )
        print("6666")
        reply = completion.choices[0].message.content.strip()
        print(reply)
        return ChatResponse(reply=reply or "…", kind="text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {type(e).__name__}")

@app.post("/video_qa", response_model=ChatResponse)
def video_qa(body: VideoQARequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not rate_limit_ok(ip):
        raise HTTPException(status_code=429, detail="Too many requests, please slow down.")

    vid = body.video_id.strip()
    if vid not in TRANSCRIPT_CACHE:
        # 没缓存就尝试抓一次
        segments = get_transcript_segments(vid)
        title = get_video_title(vid)
        text = join_text(segments)
        TRANSCRIPT_CACHE[vid] = {
            "title": title,
            "segments": segments,
            "text": text,
            "ts": time.time(),
        }
    data = TRANSCRIPT_CACHE[vid]

    # 选片段 + 回答（更智能的上下文选择）
    q = body.question.strip().lower()
    segments_all = data["segments"]

    # 如果问题包含时间戳，优先围绕时间戳取窗口
    import re as _re
    ts_list = _re.findall(r"(?:\b\d{1,2}:\d{2}\b|\b\d{1,2}:\d{2}:\d{2}\b)", q)
    picks: List[Dict[str, Any]] = []
    if ts_list:
        # take nearest index and include ±neighbors for continuity
        picks.extend(pick_time_windows(segments_all, ts_list, neighbors=6))

    # 关键词检索补充
    picks.extend(select_relevant_segments(body.question, segments_all, topk=10))

    # 如果问题是总览型（main points / 概要 / 总结 / overall 等），取全局均匀采样
    overview_triggers = ["main point", "main points", "overall", "summary", "summarize", "概述", "概要", "总结", "大意", "要点", "重点"]
    if any(t in q for t in overview_triggers) or (len(q) <= 12 and "?" not in q):
        picks.extend(spread_segments(segments_all, maxn=40))

    # 兜底：若仍为空，则取均匀采样
    if not picks:
        picks = spread_segments(segments_all, maxn=30)

    # 去重并保持原始顺序
    seen_ids = set()
    ordered = []
    for seg in segments_all:
        if seg in picks and id(seg) not in seen_ids:
            ordered.append(seg)
            seen_ids.add(id(seg))
    # allow a few more when a timestamp guided the selection
    cap = 80 if ts_list else 50
    picks = ordered[:cap]

    answer = answer_question(body.question, data["title"], picks)

    # citations：返回时间戳与摘要文本
    cites = [{"time": human_time(s["start"]), "text": s["text"]} for s in picks]

    return ChatResponse(
        reply=answer,
        kind="video_qa",
        video={"id": vid, "title": data["title"]},
        citations=cites,
    )