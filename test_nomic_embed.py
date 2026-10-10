from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

resp = client.embeddings.create(
    model="yorkieoh10/text-embedding-nomic-embed-text-v1.5",
    input="脓毒症患者ICU生命体征数据、乳酸、SOFA评分"
)
print(f"✅ 向量维度：{len(resp.data[0].embedding)}")
