import streamlit as st
import os
import ray
import warnings
from pathlib import Path
from imrag.constant import EMBEDDING_DIMENSIONS, MAX_CONTEXT_LENGTHS
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np
from ray.data import ActorPoolStrategy
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import openai
import time
import psycopg
from pgvector.psycopg import register_vector

from functools import partial

from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

# from dotenv import load_dotenv
# load_dotenv()

NEW_DATA_LOADED = False
DB_CONNECTION_STRING = "postgresql://localhost:5432/postgres?user=postgres"


def init_module(is_reload_required):
    # Initialize cluster
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        runtime_env={
            "env_vars": {
                "OPENAI_API_KEY": "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",
            },
        }
    )
    ray.cluster_resources()

    IMGS_DIR = Path("data/")
    ds = ray.data.from_items(
        [{"path": path.as_posix()} for path in IMGS_DIR.rglob("*.jpg") if not path.is_dir()]
    )
    print(f"{ds.count()} images")

    class ExtractText:
        def __init__(self):
            self.ocr_model = ocr_predictor(pretrained=True)

        def __call__(self, batch):
            doc = DocumentFile.from_images(batch["path"])
            text = self.ocr_model(doc)
            return {"source": batch["path"], "text": [text.render()]}

    # Extract texts from images
    texts_ds = ds.map_batches(ExtractText, compute=ActorPoolStrategy(size=1))

    # ample = texts_ds.take(ds.count())
    # print("dumps print: ", json.dumps(sample, indent=2))

    class EmbedTexts:
        def __init__(self, model_name="thenlper/gte-base"):
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"device": "cpu", "batch_size": 4},
            )

        def __call__(self, batch):
            embeddings = self.embedding_model.embed_documents(batch["text"])
            return {
                "text": batch["text"],
                "source": batch["source"],
                "embeddings": embeddings,
            }

    embedded_texts = texts_ds.map_batches(
        EmbedTexts,
        fn_constructor_kwargs={"model_name": "thenlper/gte-base"},
        compute=ActorPoolStrategy(size=1),
    )

    # sample = embedded_texts.take(1)
    # print(f"embedding size: {len(sample[0]['embeddings'])}")

    class StoreEmbeddings:
        def __call__(self, batch):
            with psycopg.connect(DB_CONNECTION_STRING, password="postgres") as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    for text, source, embedding in zip(batch["text"], batch["source"], batch["embeddings"]):
                        cur.execute("INSERT INTO infographic (text, source, embedding) VALUES (%s, %s, %s)",
                                    (text, source, embedding,), )
            return {}

    embedded_texts.map_batches(
        StoreEmbeddings,
        batch_size=4,
        num_cpus=1,
        compute=ActorPoolStrategy(size=1),
    ).count()


# ***************END OF INIT_MODULE***************************#

def semantic_search(query, embedding_model, num_infographics):
    question_embedding = np.array(embedding_model.embed_query(query))
    with psycopg.connect(DB_CONNECTION_STRING, password="postgres") as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source, text FROM infographic ORDER BY embedding <=> %s LIMIT %s",
                (question_embedding, num_infographics),
            )
            rows = cur.fetchall()
            semantic_context = [
                {"id": row[0], "source": row[1], "text": row[2]} for row in rows
            ]
    return semantic_context


def generate_response(
        llm,
        temperature=0.0,
        system_content="",
        assistant_content="",
        user_content="",
        max_retries=1,
        retry_interval=60,
):
    """Generate response from an LLM."""
    retry_count = 0
    while retry_count <= max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=llm,
                temperature=temperature,
                stream=False,
                api_key="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",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": user_content},
                ],
            )
            return response["choices"][-1]["message"]["content"]
        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)
            retry_count += 1
    return ""


def get_response_to_user_query():
    # Retrieve context
    query = " whose is vendee of this sale deed was made and what is the size of land?"
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-base")
    num_infographics = 4
    results = semantic_search(query, embedding_model, num_infographics)
    context = [item["text"] for item in results]

    # Generate response
    response = generate_response(
        llm="gpt-4",
        temperature=0.0,
        system_content="Answer the query using the context provided. Be succinct.",
        user_content=f"query: {query}, context: {context}",
    )
    print("query: ", query)
    # Stream response
    for content in response:
        print(content, end="", flush=True)

    sources = [item["source"] for item in results]
    print(sources)


# ***************get_response_to_user_query***************************#
# init_module(True)
get_response_to_user_query()
# main()
# query_summarize()
