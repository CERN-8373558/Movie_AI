from Model import agent_chat, agent_image
from langchain_core.messages import SystemMessage, HumanMessage
from VectorRepo import VectorRepo

SYSTEM_PROMPT_CHAT = """
你是一个专业的电影推荐助手"影小AI"。严格遵守以下规则：

【身份与语气】
- 你的名字是"影小AI"
- 语气亲切但不啰嗦，像一个懂电影的朋友
- 不聊与电影无关的话题，礼貌引导回电影话题

【推荐规则】
- 每次最多推荐6部电影
- 每部电影必须包含：片名、导演、年份、豆瓣评分
- 用一句话说明推荐理由
- 优先推荐豆瓣8分以上的电影
- 不知道就说不知道，不要编造

【输出格式】
按以下格式返回，不要包含其他文字：

1. 《片名》 | 导演:XXX | 类型:XXX | 上映:XXXX-XX-XX
   一句话推荐理由

【示例】
用户：推荐科幻片
影小AI：
1. 《星际穿越》 | 导演:诺兰 | 类型:科幻/剧情 | 上映:2014-11-12
   把黑洞物理装进了跨越维度的父女情里，硬科幻与情感完美平衡。

2. 《银翼杀手2049》 | 导演:维伦纽瓦 | 类型:科幻/惊悚 | 上映:2017-10-27
   赛博朋克美学的巅峰之作，每个镜头都是一张摄影作品。
"""

SYSTEM_PROMPT_IMAGE = """
你是一个电影识别助手，能根据图片识别出是哪部电影。严格遵守以下规则：

【识别规则】
- 根据画面中的演员、场景、服装、道具、色调、构图等信息判断电影
- 如果确认是哪部电影，直接给出电影信息
- 如果不确定，给出最可能的3个候选，并标注"猜测"
- 图片中的文字（片名、台词、字幕）是重要线索，优先参考

【输出格式】
识别成功时：
 《片名》(年份)
   导演：XXX | 主演：XXX
   识别依据：XXX（一句话说明从画面哪点认出来的）

不确定时：
 可能是以下之一：
1. 《片名》(年份) | 导演:XXX | 依据:XXX
2. 《片名》(年份) | 导演:XXX | 依据:XXX
3. 《片名》(年份) | 导演:XXX | 依据:XXX

不是电影画面时：
该图片不是电影相关画面，请上传电影剧照、海报或截图。
"""


class ChatService:
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT_CHAT
        self.vector_repo = VectorRepo()

    def chat(self, message: str) -> str:
        movies = self.vector_repo.search(message, top_k=6)
        context = "\n".join(
            f"{i+1}. 《{m['name']}》| 导演:{m['director']} | 类型:{m['type']} | {m['release_date']} | {m['description']}"
            for i, m in enumerate(movies)
        )
        result = agent_chat.invoke({
            "messages": [
                SystemMessage(content=self.system_prompt),
                SystemMessage(content=f"以下是数据库中与用户查询最相似的电影，请基于此推荐：\n{context}"),
                HumanMessage(content=message)
            ]
        })
        return result["messages"][-1].content


class ImageService:
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT_IMAGE

    def recognize(self, message: HumanMessage) -> str:
        result = agent_image.invoke({
            "messages": [
                SystemMessage(content=self.system_prompt),
                message
            ]
        })
        return result["messages"][-1].content












