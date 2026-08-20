import os
import base64
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from starlette.middleware.cors import CORSMiddleware

from Service import ChatService, ImageService
from VectorRepo import VectorRepo
from SimilarMovie import SimilarMovie
from MovieRecognition import MovieRecognitionService


app = FastAPI();
app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_methods=["*"],
      allow_headers=["*"],
  )
chat_service = ChatService();
image_service = ImageService();
movie_recognition_service = MovieRecognitionService();


class ChatReq(BaseModel):
    message: str

class ChatResp(BaseModel):
    reply: str


class ImageReq(BaseModel):
    text: str
    image_url: str

class ImageResp(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    reply = chat_service.chat(req.message)
    return ChatResp(reply=reply)


@app.post("/image/recognize", response_model=ImageResp)
def image_recognize(req: ImageReq):
    msg = HumanMessage(content=[
        {"type": "text", "text": req.text},
        {"type": "image_url", "image_url": {"url": req.image_url}},
    ])
    reply = image_service.recognize(msg)
    return ImageResp(reply=reply)


class SyncResp(BaseModel):
    count: int

@app.post("/vector/sync", response_model=SyncResp)
def vector_sync():
    repo = VectorRepo()
    count = repo.sync_from_springboot(
        base_url="http://localhost:8080",
        token="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIiwiaWF0IjoxNzgwODA5OTUyLCJleHAiOjE3ODE0MTQ3NTJ9.ItqP66oun4MIDaKupMvGizSEJQsXs2srl4QgZc4dvgE"
    )
    return SyncResp(count=count)


similar_service = SimilarMovie()

class SimilarReq(BaseModel):
    movie_id: int

class SimilarResp(BaseModel):
    movie_id: int
    similar_ids: list[int]

@app.post("/movie/similar", response_model=SimilarResp)
def movie_similar(req: SimilarReq):
    ids = similar_service.get_similar(req.movie_id)
    return SimilarResp(movie_id=req.movie_id, similar_ids=ids)


@app.post("/movie/recognize")
async def movie_recognize(
    image: UploadFile = File(...),
    text: str = Form("请识别这张图片中的电影"),
):
    image_bytes = await image.read()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{image.content_type};base64,{b64}"
    result = movie_recognition_service.recognize(data_url, text)
    return result


import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8085)
