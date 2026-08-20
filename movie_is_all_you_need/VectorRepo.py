import os
import requests
from pymilvus import MilvusClient, DataType
from openai import OpenAI

_client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class VectorRepo:
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), "movie.db")
        self.client = MilvusClient(self.db_path)
        self.collection_name = "movies"
        self._init_collection()
        self.client.load_collection(self.collection_name)

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=1024,
            datatype=DataType.FLOAT_VECTOR,
            metric_type="COSINE",
        )

    def _embed(self, text: str) -> list[float]:
        resp = _client.embeddings.create(
            model="text-embedding-v3",
            input=text,
        )
        return resp.data[0].embedding

    def sync_from_springboot(self, base_url: str, token: str, start: int = 0) -> int:
        resp = requests.get(
            f"{base_url}/FILMES/ALL",
            headers={"token": token},
            timeout=30,
        )
        resp.raise_for_status()
        movies = resp.json()
        total = len(movies)
        print(f"共 {total} 部电影，从第 {start + 1} 部开始...")

        for i in range(start, total):
            m = movies[i]
            text = f"{m['name']} {m.get('director','')} {m.get('actors','')} {m.get('type','')} {m.get('region','')} {m.get('description','')}"
            vector = self._embed(text)
            self.client.upsert(
                collection_name=self.collection_name,
                data=[{
                    "id": m["id"],
                    "vector": vector,
                    "name": m.get("name", ""),
                    "cover": m.get("cover", ""),
                    "director": m.get("director", ""),
                    "actors": m.get("actors", ""),
                    "type": m.get("type", ""),
                    "region": m.get("region", ""),
                    "language": m.get("language", ""),
                    "release_date": m.get("releaseDate", ""),
                    "duration": m.get("duration", 0),
                    "description": m.get("description", ""),
                }],
            )
            print(f"  [{i + 1}/{total}] {m['name']}")

        return total - start

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        vector = self._embed(query)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=top_k,
            output_fields=["name", "director", "actors", "type", "region",
                           "release_date", "duration", "description"],
        )
        return [
            {
                "id": hit["id"],
                "name": hit["entity"]["name"],
                "director": hit["entity"]["director"],
                "actors": hit["entity"]["actors"],
                "type": hit["entity"]["type"],
                "region": hit["entity"]["region"],
                "release_date": hit["entity"]["release_date"],
                "duration": hit["entity"]["duration"],
                "description": hit["entity"]["description"],
                "similarity": round(hit["distance"], 2),
            }
            for hit in results[0]
        ]

    def count(self) -> int:
        return self.client.query(
            collection_name=self.collection_name,
            filter="id >= 0",
            output_fields=["count(*)"],
        )[0]["count(*)"]


if __name__ == "__main__":
    repo = VectorRepo()
    count = repo.sync_from_springboot(
        base_url="http://localhost:8080", 
        token="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIiwiaWF0IjoxNzgwODA5OTUyLCJleHAiOjE3ODE0MTQ3NTJ9.ItqP66oun4MIDaKupMvGizSEJQsXs2srl4QgZc4dvgE"
    )
    print(f"同步完成，共 {count} 部电影")
