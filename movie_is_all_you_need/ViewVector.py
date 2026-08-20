from VectorRepo import VectorRepo

repo = VectorRepo()

total = repo.count()
print(f"===== 向量库内容 (共 {total} 部) =====")

if total == 0:
    print("向量库为空，请先运行 VectorRepo.py 初始化")
    exit()

# 文本信息
results = repo.client.query(
    collection_name=repo.collection_name,
    filter="id >= 0",
    limit=1000,
    output_fields=["id", "name", "director", "type", "release_date", "description"],
)

for r in results:
    desc = r.get('description') or ''
    print(f"  [{r['id']}] 《{r['name']}》| 导演:{r.get('director', '?')} | {r.get('type', '?')} | {r.get('release_date', '?')}")
    print(f"      {desc[:80]}")

    # 拿这条的向量
    vec = repo.client.get(
        collection_name=repo.collection_name,
        ids=[r["id"]],
        output_fields=["vector"],
    )[0].get("vector", [])
    print(f"      向量: [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}, ...] (共{len(vec)}维)")
    print()
