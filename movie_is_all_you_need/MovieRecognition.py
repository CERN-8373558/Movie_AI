import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from Model import agent_vision
from VectorRepo import VectorRepo

SYSTEM_PROMPT_VISION = """
你是一个专业的电影识别助手"影小AI"。用户会发给你一张电影相关图片（海报、剧照、截图），你需要仔细分析画面内容，然后以 JSON 格式返回结果。

【分析步骤】
1. 观察画面中的人物：有没有知名演员？面部特征、服装风格、发型、妆容有什么特点？
2. 观察文字信息：画面上有没有片名、台词、字幕、海报文字？这是最重要的线索，优先提取。
3. 观察场景：室内还是室外？什么时代背景？什么国家/地区？
4. 观察色调和风格：暖色调/冷色调、明亮/黑暗、写实/风格化、胶片感/数字感
5. 观察构图和道具：有什么标志性的物品、建筑、车辆、武器？

【输出格式】
必须且只能返回以下 JSON，不要包含任何其他文字：

{
  "analysis": "对画面的详细解析，2-4句话，涵盖关键视觉信息",
  "movie_guess": [
    {
      "name": "电影片名（中文）",
      "year": "上映年份",
      "confidence": "高/中/低",
      "reason": "一句话说明猜测依据"
    }
  ],
  "genre_guess": ["类型1", "类型2"],
  "era_guess": "年代描述（如1990年代、2010年代、古典时期、近未来）",
  "visual_features": {
    "has_text": true,
    "text_content": "画面中出现的文字内容，没有则填'无'",
    "has_actor": true,
    "actor_desc": "可辨认的演员及其特征描述，没有则填'无明显可辨认演员'",
    "color_tone": "暖色调/冷色调/暗黑/明亮/复古/自然",
    "scene_type": "室内/室外/城市/自然/太空/战争/科幻/古装/现代"
  }
}

【规则】
- movie_guess 列出1-3个最可能的电影，按置信度从高到低排列
- 如果画面中有明确的片名文字，confidence 应为"高"
- 如果完全无法判断，movie_guess 可以为空数组，但 analysis 仍需描述画面
- 不是电影画面时（如表情包、纯文字、生活照），analysis 中说明，movie_guess 留空
- 只返回 JSON，不要任何其他文字
"""


class MovieRecognitionService:
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT_VISION
        self.vector_repo = VectorRepo()

    def recognize(self, image_url: str, text: str = "请识别这张图片中的电影") -> dict:
        print("\n" + "=" * 50)
        print("[图片识电影] 收到请求")

        # 1. VL 模型分析图片
        print("[1/3] 调用 qwen3-vl-plus 分析图片...")
        vl_result = self._call_vl(image_url, text)
        guesses = vl_result.get("movie_guess", [])
        print(f"      分析完成，猜测 {len(guesses)} 部电影")
        for g in guesses:
            print(f"      - 《{g['name']}》({g['year']}) 置信度:{g['confidence']}")
        print(f"      画面解析: {vl_result.get('analysis', '')[:100]}...")

        # 2. 只取置信度最高的一部
        if guesses:
            best = guesses[0]
            print(f"\n[2/3] 取置信度最高的「{best['name']}」在电影库中搜索...")
            matched = self._search_movie(best)
            best["matched_movie"] = matched[0] if matched else None
            print(f"      匹配结果: {best['matched_movie']['name'] if best['matched_movie'] else '无'}")

            # 3. 搜相似电影
            if best["matched_movie"]:
                m = best["matched_movie"]
                print(f"      → 搜「{m['name']}」的相似电影 (加权:类型>导演>简介>片名)...")
                similar = self._get_similar(m)
                print(f"      共找到 {len(similar)} 部")
            else:
                similar = []
            best["similar_movies"] = similar

            vl_result["movie_guess"] = best
        else:
            vl_result["movie_guess"] = None

        print("=" * 50)
        print("[图片识电影] 完成\n")
        return vl_result

    def _call_vl(self, image_url: str, text: str) -> dict:
        msg = HumanMessage(content=[
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ])

        result = agent_vision.invoke({
            "messages": [
                SystemMessage(content=self.system_prompt),
                msg,
            ]
        })

        raw = result["messages"][-1].content
        return self._parse_json(raw)

    def _search_movie(self, guess: dict) -> list[dict]:
        """
        向量搜索 + 文本匹配双路召回。
        向量搜索语义容易漂移（如"猜火车"→"乘火车去旅行"），
        文本匹配用 like 兜底，确保片名包含搜索词的电影不被漏掉。
        """
        name = guess.get("name", "")
        year = guess.get("year", "")
        reason = guess.get("reason", "")

        # 向量搜索：丰富查询 + 纯片名
        rich_query = f"{name} {year} {reason}"
        vec_results = self.vector_repo.search(rich_query, top_k=10)
        vec_results += self.vector_repo.search(name, top_k=10)

        # 文本匹配：片名包含搜索词
        text_results = self._text_match(name)

        # 合并去重，向量结果优先排序，文本结果补到末尾
        seen = set()
        merged = []
        for r in vec_results:
            rid = r["id"]
            if rid not in seen:
                seen.add(rid)
                merged.append({**r, "_source": "向量"})
        for r in text_results:
            rid = r["id"]
            if rid not in seen:
                seen.add(rid)
                merged.append({**r, "_source": "文本"})

        matched = []
        for r in merged[:5]:
            print(f"      [{r['id']}] 《{r['name']}》 [{r['_source']}] 相似度:{r.get('similarity', '-')}")
            matched.append({
                "id": r["id"],
                "name": r["name"],
                "director": r["director"],
                "type": r["type"],
                "release_date": r["release_date"],
                "description": r.get("description", ""),
            })
        return matched

    def _text_match(self, name: str) -> list[dict]:
        """文本匹配：在 Milvus 中用 like 按片名搜索"""
        try:
            results = self.vector_repo.client.query(
                collection_name=self.vector_repo.collection_name,
                filter=f'name like "%{name}%"',
                output_fields=["id", "name", "director", "type", "release_date", "description"],
                limit=10,
            )
            return [
                {
                    "id": r["id"],
                    "name": r.get("name", ""),
                    "director": r.get("director", ""),
                    "type": r.get("type", ""),
                    "release_date": r.get("release_date", ""),
                    "description": r.get("description", ""),
                    "similarity": 1.0,  # 文本精确匹配视为最高相似
                }
                for r in results
            ]
        except Exception:
            return []

    def _get_similar(self, matched_movie: dict) -> list[dict]:
        """
        按加权策略搜索相似电影：类型 > 导演 > 简介 > 片名。
        避免纯向量相似度被单个词带偏。
        """
        try:
            movie_type = matched_movie.get("type", "")
            director = matched_movie.get("director", "")
            description = matched_movie.get("description", "")
            name = matched_movie.get("name", "")

            weighted_query = f"{movie_type} {movie_type} {movie_type} {movie_type} {movie_type} " \
                             f"{director} {director} {director} " \
                             f"{description} {description} " \
                             f"{name}"

            results = self.vector_repo.search(weighted_query, top_k=12)
            similar = []
            for r in results:
                if r["name"] != name:
                    similar.append({
                        "id": r["id"],
                        "name": r["name"],
                        "director": r["director"],
                        "type": r["type"],
                        "release_date": r["release_date"],
                        "similarity": r["similarity"],
                    })
            return similar[:10]
        except Exception:
            return []

    def _parse_json(self, raw: str) -> dict:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "analysis": raw,
            "movie_guess": [],
            "genre_guess": [],
            "era_guess": "",
            "visual_features": {
                "has_text": False,
                "text_content": "",
                "has_actor": False,
                "actor_desc": "",
                "color_tone": "",
                "scene_type": "",
            },
        }
