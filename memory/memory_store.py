import time
from pymilvus import Collection

class AgentMemory:
    def __init__(self, collection_name="agent_memory"):
        self.col = Collection(collection_name)

    def insert_memory(self, embedding, text, mem_type="episodic"):
        ts = int(time.time() * 1000)
        self.col.insert(
            [
                [None],
                [embedding],
                [mem_type],
                [text],
                [ts]
            ]
        )

    def search(self, embedding, top_k=5, mem_type=None):
        expr = f'mem_type == "{mem_type}"' if mem_type else ""
        return self.col.search(
            data=[embedding],
            anns_field="embedding",
            param={"ef":64},
            limit=top_k,
            expr=expr
        )

