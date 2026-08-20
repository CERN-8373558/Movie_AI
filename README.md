# Movie_AI

基于 FastAPI + LangChain + Milvus 的电影智能推荐微服务。
通过语义检索与多模态大模型实现自然语言电影推荐、图片识别电影、相似电影推荐等功能。
网站地址： http://www.ciaohello.icu/ 测试账户： user/user
## 功能特性

| 接口 | 说明 |
| ---- | ---- |
| `POST /chat` | 自然语言电影推荐，DeepSeek 智能体基于向量检索结果生成推荐 |
| `POST /image/recognize` | 多模态识图：传入图片 URL + 文字，Qwen-VL 识别图片中的电影 |
| `POST /movie/recognize` | 多模态识图：直接上传图片文件识别电影 |
| `POST /vector/sync` | 从 Java Spring Boot 后端拉取全部电影，向量化后写入 Milvus |
| `POST /movie/similar` | 根据电影 ID 返回 Top-35 相似电影 |

## 技术架构

- **FastAPI** 提供 REST 接口
- **LangChain / LangGraph** 构建智能体（Agent）
- **Milvus Lite**（嵌入式向量库）存储 1024 维电影向量，COSINE 距离检索
- **Qwen 3.5 Plus / Qwen-VL**（DashScope OpenAI 兼容接口）多模态识别
- **text-embedding-v3** 文本向量化
- **DeepSeek V4 Pro** 推荐对话智能体

数据流：

```
Java Spring Boot ──(/FILMES/ALL)──> 向量化(text-embedding-v3) ──> Milvus Lite (movies)
用户请求 ──> 向量检索(Top-K) ──> LLM 智能体 ──> 推荐结果
```

## 快速开始

### 1. 环境要求

- Python 3.14+
- `.env` 配置文件（见下方环境变量）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件：

```bash
QWEN_API_KEY=你的阿里云百炼API_KEY
DEEPSEEK_API_KEY=你的DeepSeek_API_KEY
```

### 4. 启动服务

```bash
python movie_is_all_you_need/Contrllor.py
```

服务监听 `0.0.0.0:8085`。

### 5. 健康检查

```bash
curl -s http://127.0.0.1:8085/
```

## 接口示例

### 自然语言推荐

```bash
curl -X POST http://127.0.0.1:8085/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "推荐一部悬疑烧脑的国产电影"}'
```

### 同步电影向量

```bash
curl -X POST http://127.0.0.1:8085/vector/sync
```

### 相似电影推荐

```bash
curl -X POST http://127.0.0.1:8085/movie/similar \
  -H "Content-Type: application/json" \
  -d '{"movie_id": 1}'
```

## 目录结构

```
movie_is_all_you_need/
├── Contrllor.py        # FastAPI 入口与路由
├── Model.py            # 模型初始化（Qwen / DeepSeek / 嵌入）
├── Service.py          # 聊天与图片识别业务逻辑
├── VectorRepo.py       # Milvus 向量库封装与同步
├── SimilarMovie.py     # 相似电影推荐
├── MovieRecognition.py # 图片识别电影
├── ViewVector.py       # 向量查看工具
└── movie.db/           # Milvus Lite 本地数据（不入库）
```

## 常见问题

- **Windows 中文路径读取问题**：本项目图片处理已通过 `np.fromfile` + `cv2.imdecode` 规避 `cv2.imread` 的中文路径 bug。
- **Milvus 文件锁冲突**：Milvus Lite 在 Windows 上需避免多进程同时打开同一数据库。
