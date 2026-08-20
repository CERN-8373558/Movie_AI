# 图片识电影 — VL 视觉理解模块

## 1. 定位

"图片识电影"功能的第一阶段：纯视觉理解。

发一张电影画面（海报/剧照/截图）给千问 VL 模型，模型分析画面内容，返回结构化的解析结果。

**不查数据库、不接 Milvus**，只靠 VL 模型的视觉能力做判断。

## 2. 输入

| 字段 | 类型 | 说明 |
|------|------|------|
| `image_url` | string | 图片URL（或后续扩展为 base64 / multipart） |
| `text` | string（可选） | 用户附带的文字描述，如"这是哪部电影" |

## 3. 输出

```json
{
  "analysis": "对画面内容的详细解析，包括人物、场景、色调、文字信息",
  "movie_guess": [
    {"name": "疑似片名1", "year": "年份", "confidence": "高/中/低", "reason": "依据"},
    {"name": "疑似片名2", "year": "年份", "confidence": "高/中/低", "reason": "依据"}
  ],
  "genre_guess": ["科幻", "悬疑"],
  "era_guess": "2010年代",
  "visual_features": {
    "has_text": true,
    "text_content": "画面中的文字内容",
    "has_actor": true,
    "actor_desc": "演员特征描述",
    "color_tone": "暖色调/冷色调/暗黑/明亮",
    "scene_type": "室内/室外/城市/自然/太空/..."
  }
}
```

## 4. 处理流程

```
用户发图片 + 可选文字
        │
        ▼
┌─────────────────────┐
│ qwen3-vl-plus       │  根据画面内容做多维度分析
│ - 识别演员/场景       │
│ - 读取画面文字        │
│ - 判断色调/年代/类型   │
│ - 综合猜测电影        │
└──────────┬──────────┘
           │
           ▼
      结构化 JSON 响应
```

## 5. 技术选型

- 模型：`qwen3-vl-plus`（DashScope OpenAI 兼容 API）
- 框架：FastAPI 现有服务，新增端点
- 输出格式：JSON（Pydantic 模型约束）

## 6. 与现有代码的关系

- **不动** `Model.py` 中现有的 `model_image`（qwen3.5-plus）
- **不动** `Service.py` 中现有的 `ImageService`
- **不动** `Contrllor.py` 中现有的 `/image/recognize`
- **新增** 模型定义、服务类、端点

## 7. 后续扩展方向

本期只做 VL 视觉理解。后续可以：
- 接入 Embedding → Milvus 检索电影库
- 接入演员识别 API
- 多图联合识别
- 对话式追问（"这个演员还演过什么"）
