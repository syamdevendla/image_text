import streamlit as st
import os
import ray
import warnings
from pathlib import Path
# from imrag.constant import EMBEDDING_DIMENSIONS, MAX_CONTEXT_LENGTHS
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
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

# import streamlit as st
# st.set_page_config("Hello")

# from dotenv import load_dotenv
# load_dotenv()

NEW_DATA_LOADED = False
DB_CONNECTION_STRING = "postgresql://localhost:5432/postgres?user=postgres"


def init_pdf_module(is_reload_required):
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

    PDF_DIR = Path("data_pdf/")
    ds = ray.data.from_items(
        [{"path": path.as_posix()} for path in PDF_DIR.rglob("*.pdf") if not path.is_dir()]
    )
    print(f"{ds.count()} images")

    class ExtractPdfText:
        def __init__(self):
            pass

        def __call__(self, batch):
            print("Entered PyPDFLoader 1:", batch["path"])
            loader = PyPDFLoader(str(batch["path"]), extract_images=True)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            chunks = text_splitter.split_documents(data)
            return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]

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

    class StoreEmbeddings:
        def __call__(self, batch):
            with psycopg.connect(DB_CONNECTION_STRING, password="postgres") as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    for text, source, embedding in zip(batch["text"], batch["source"], batch["embeddings"]):
                        cur.execute("INSERT INTO infographic (text, source, embedding) VALUES (%s, %s, %s)",
                                    (text, source, embedding,), )
            return {}

    # Split texts of pdfs
    texts_ds = ds.flat_map(ExtractPdfText, compute=ActorPoolStrategy(size=1))

    # sample = texts_ds.take(ds.count())
    # print("dumps print: ", json.dumps(sample, indent=2))

    embedded_texts = texts_ds.map_batches(
        EmbedTexts,
        fn_constructor_kwargs={"model_name": "thenlper/gte-base"},
        compute=ActorPoolStrategy(size=1),
    )

    # sample = embedded_texts.take(1)
    # print(f"embedding size: {len(sample[0]['embeddings'])}")

    embedded_texts.map_batches(
        StoreEmbeddings,
        batch_size=4,
        num_cpus=1,
        compute=ActorPoolStrategy(size=1),
    ).count()


# ***************END OF INIT_MODULE***************************#


def semantic_search(query, embedding_model, num_of_chunks):
    question_embedding = np.array(embedding_model.embed_query(query))
    with psycopg.connect(DB_CONNECTION_STRING, password="postgres") as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source, text FROM infographic ORDER BY embedding <=> %s LIMIT %s",
                (question_embedding, num_of_chunks),
            )
            rows = cur.fetchall()
            print("rows: ", len(rows))
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


class QueryAgent:
    def __init__(self, embedding_model_name="thenlper/gte-base",
                 llm="meta-llama/Llama-2-70b-chat-hf", temperature=0.0,
                 max_context_length=4096, system_content="", assistant_content=""):
        # Embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Context length (restrict input length to 50% of total length)
        max_context_length = int(0.5 * max_context_length)

        # LLM
        self.llm = llm
        self.temperature = temperature
        # self.context_length = max_context_length - get_num_tokens(system_content + assistant_content)
        self.system_content = system_content
        self.assistant_content = assistant_content

    def __call__(self, query, num_chunks=5, stream=True):
        # Get sources and context
        context_results = semantic_search(
            query=query,
            embedding_model=self.embedding_model,
            num_of_chunks=num_chunks)

        # Generate response
        context = [item["text"] for item in context_results]
        sources = [item["source"] for item in context_results]
        user_content = f"query: {query}, context: {context}"

        answer = generate_response(
            llm=self.llm,
            temperature=self.temperature,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=user_content)

        # Result
        result = {
            "question": query,
            "sources": sources,
            "answer": answer,
            "llm": self.llm,
            "context": context,
        }
        return result


def get_response_to_user_query():
    # Retrieve context
    query = "whats the website mentioned to Get a quote for the completion of  template "
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-base")
    num_infographics = 5
    results = semantic_search(query, embedding_model, num_infographics)
    context = [item["text"] for item in results]
    print("context: ", context)

    # llm="gpt-4",
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


def use_agent(input_query):
    #llm = "meta-llama/Llama-2-7b-chat-hf"
    llm = "gpt-4"
    agent = QueryAgent(
        embedding_model_name="thenlper/gte-base",
        llm=llm,
        system_content="Answer the query using the context provided. Be succinct.")
    response = agent(query=input_query)
    #print("query: ", input_query)
    #answers = [item["answer"] for item in response]
    #print(answers)
    #sources = [item["sources"] for item in response]
    #print(sources)
    print("\n\n", json.dumps(response, indent=2))


# ***************get_response_to_user_query***************************#

# init_pdf_module(True)
query = "whats the website mentioned to Get a quote for the completion of  template "
use_agent(query)

# get_response_to_user_query()
# main()
# query_summarize()
