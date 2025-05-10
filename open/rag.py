"""
Bito - 比 Copilot 还多些创新
Amazon CodeWhisperer - 代码补全，免费。AWS 相关的编程能力卓越。其它凑合
Cursor - AI first 的 IDE。
Tabnine - 代码补全，个人基础版免费
Tongyi Lingma -- 代码补全，免费。阿里云相关。
"""
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    '''从 PDF 文件中（按指定页码）提取文字'''
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' '+text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    return paragraphs
paragraphs = extract_text_from_pdf("llama2.pdf", min_line_length=10)

for para in paragraphs[:3]:
    print(para+"\n")

"""
比如文档中包含很长的专有名词，关键字检索往往更精准而向量检索容易引入概念混淆。
混合检索的核心是，综合文档 
 在不同检索算法下的排序名次（rank），为其生成最终排序。
https://github.com/Raudaschl/rag-fusion
https://github.com/FlagOpen/FlagEmbedding

一个最常用的算法叫 Reciprocal Rank Fusion（RRF）
FAISS: Meta 开源的向量检索引擎 https://github.com/facebookresearch/faiss
Pinecone: 商用向量数据库，只有云服务 https://www.pinecone.io/
Milvus: 开源向量数据库，同时有云服务 https://milvus.io/
Weaviate: 开源向量数据库，同时有云服务 https://weaviate.io/
Qdrant: 开源向量数据库，同时有云服务 https://qdrant.tech/
PGVector: Postgres 的开源向量检索引擎 https://github.com/pgvector/pgvector
RediSearch: Redis 的开源向量检索引擎 https://github.com/RediSearch/RediSearch
ElasticSearch 也支持向量检索 https://www.elastic.co/enterprise-search/vector-search
"""

class RAG_Bot:
    def __init__(self, vector_db, llm_api, n_results=2):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        # 1. 检索
        search_results = self.vector_db.search(user_query, self.n_results)

        # 2. 构建 Prompt
        prompt = build_prompt(
            prompt_template, info=search_results['documents'][0], query=user_query)

        # 3. 调用 LLM
        response = self.llm_api(prompt)
        return response

# 创建一个RAG机器人
bot = RAG_Bot(
    vector_db,
    llm_api=get_completion
)

user_query = "llama 2有对话版吗？"

response = bot.chat(user_query)

print(response)


model = "text-embedding-3-large"
dimensions = 128

query = "国际争端"

# 且能支持跨语言
# query = "global conflicts"

documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]

query_vec = get_embeddings([query],model=model,dimensions=dimensions)[0]
doc_vecs = get_embeddings(documents,model=model,dimensions=dimensions)

print("Dim: {}".format(len(query_vec)))

print("Cosine distance:")
for vec in doc_vecs:
    print(cos_sim(query_vec, vec))

print("\nEuclidean distance:")
for vec in doc_vecs:
    print(l2(query_vec, vec))