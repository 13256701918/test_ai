
# Basic AI Chatbot 🤖 (with YouTube Support)

一个基于 **FastAPI + OpenAI API** 的简易 AI 聊天机器人，支持单轮对话和网页聊天界面。

## 功能特性
- **Python 后端 (FastAPI)**  
  提供一个 `/chat` API，用于接收用户输入并返回 AI 回复。
- **AI 响应**  
  使用 OpenAI GPT 模型（默认 `gpt-4o-mini`），也可以替换为其他模型。
- **前端页面**  
  `templates/index.html` 提供基础聊天界面：输入框 + 消息显示区。
- **简洁易跑**  
  仅需几个文件，配置简单。
- **YouTube 视频解析与问答**  
  自动识别 YouTube 链接、抓取字幕（优先中/英）、生成 3–4 句总结、基于完整字幕进行问答并按 `[mm:ss–mm:ss]` 引用；若官方字幕不可用则 Whisper 兜底转写。

---

## 安装与运行

### 1. 克隆仓库
```bash
git clone https://github.com/13256701918/test_ai.git
cd test_ai

2. 创建虚拟环境（可选）

python3 -m venv .venv
source .venv/bin/activate

3. 安装依赖

pip install -r requirements.txt

4. 设置环境变量

在终端里配置 OpenAI API key（你面试用的 key 可直接用）：

export OPENAI_API_KEY="sk-xxxxxxx"

5. 启动服务

uvicorn app:app --reload --port 8000

6. 打开前端

浏览器访问：

http://127.0.0.1:8000/


---

## YouTube 功能说明

- **使用方式**：前端聊天框直接粘贴视频 URL，或在后端调用相应接口（与 /chat 一致，会自动检测 URL）。
- **字幕抓取**：基于 `youtube-transcript-api>=1.2.2` 的 `YouTubeTranscriptApi().fetch()`；优先语言 `['zh-Hans','zh-Hant','zh-CN','zh-TW','en']`；必要时进行翻译。
- **兜底转写**：若字幕被禁用/被地区合规页拦截/不存在，使用 `pytube` 下载音频 + OpenAI `whisper-1` 转写（需要 `OPENAI_API_KEY`）。
- **时间戳问答**：支持问题中包含时间戳（如 `5:30` / `01:02:03`），后端会选取该时间点附近的连续片段，生成更贴近的答案与引用。
- **引用格式**：返回的 `citations` 使用范围格式，例如 `05:26–05:32`，并标注是否覆盖提问时间点。

---

## 配置项 / 环境变量

- `OPENAI_API_KEY`：用于 GPT 回复与 Whisper 转写。
- `YT_COOKIE_STR`（可选）：分号分隔的一行 cookie 字符串，后端会解析为 dict 提高字幕抓取成功率。
- `cookies.txt`（可选）：Netscape 格式的 cookies 文件，放在项目根目录，YouTube 字幕抓取会自动使用。

---

## 依赖版本建议

- `youtube-transcript-api>=1.2.2`
- `openai>=1.51.0`
- `httpx>=0.27`
- `pytube>=15`

---

## 常见问题 (FAQ)

- 只返回到 [05:26] 但我问的是 5:30？  
  —— 字幕片段以开始时间计时，README 说明我们已改为范围显示并保证该片段覆盖 5:30。
- 提示 `no element found` / `ParseError`？  
  —— 通常是被 YouTube 同意页/风控拦截，建议提供 `YT_COOKIE_STR` 或 `cookies.txt`；否则会自动走 Whisper。
- 速度较慢？  
  —— Whisper 兜底会下载音频并转写，耗时更长；可自行关闭或按需限制。

⸻

文件结构

.
├── app.py              # FastAPI 后端
├── requirements.txt    # Python 依赖
├── templates/
│   └── index.html      # 聊天前端页面
├── .gitignore
└── README.md

**前端无需新增页面**，直接在现有聊天框粘贴链接即可触发视频流程。


⸻

截图


⸻

后续改进
	•	✅ 支持多轮对话（保存上下文）
	•	✅ 加入 HuggingFace / 本地模型选项
	•	✅ 部署到 Vercel / Render / Docker 化

⸻

License

MIT
