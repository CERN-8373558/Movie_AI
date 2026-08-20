from VectorRepo import VectorRepo


class SimilarMovie:
    def __init__(self):
        self.repo = VectorRepo()

    def get_similar(self, movie_id: int, top_k: int = 35) -> list[int]:
        # 拿目标电影的向量
        result = self.repo.client.get(
            collection_name=self.repo.collection_name,
            ids=[movie_id],
            output_fields=["vector"],
        )
        if not result:
            return []

        vec = result[0]["vector"]

        # 用向量搜索最相似的 top_k+1 部（+1 因为会搜到自己）
        results = self.repo.client.search(
            collection_name=self.repo.collection_name,
            data=[vec],
            limit=top_k + 1,
            output_fields=["id", "name"],
        )

        # 排除自己
        ids = []
        for hit in results[0]:
            if hit["id"] != movie_id:
                ids.append(hit["id"])
                sim = round(1 - hit['distance'], 4)  # COSINE: distance = 1 - 余弦相似度
                print(f"  [{hit['id']}] 《{hit['entity']['name']}》 相似度:{sim}")

        return ids[:top_k]


if __name__ == "__main__":
    s = SimilarMovie()
    ids = s.get_similar(33100)
    print(f"\n《挽救计划》最相似的7部电影ID: {ids}")
