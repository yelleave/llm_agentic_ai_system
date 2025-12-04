from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

def milvus_client():
    connections.connect(
        alias="default",
        host="127.0.0.1",
        port=1953,
    )

def create_memory_collection():
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
    type_field = FieldSchema(name="mem_type", dtype=DataType.VARCHAR, max_length=32)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048)
    ts_field = FieldSchema(name="ts", dtype=DataType.INT64)

    schema = CollectionSchema(fields=[id_field, vector_field, type_field, text_field, ts_field])
    collection = Collection("agent_memory", schema=schema)

    collection.create_index(
        field_name = "embedding",
        index_params = {
            "index_type": "HNSW",
            "metric_type": "cosine",
            "params": {"M": 32, "efConstruction": 200},
        }
    )