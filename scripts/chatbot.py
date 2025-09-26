# scripts/chatbot.py
import os
import traceback
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# --------------------------
# CONFIG
# --------------------------
MODEL_PATH = os.getenv("LLAMA_MODEL", "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
VECTOR_DB_PATH = os.getenv("CHROMA_PATH", "vectorstore/chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --------------------------
# PROMPT TEMPLATE
# --------------------------
CUSTOM_PROMPT = """
You are an AI assistant specialized in oceanography and ARGO float data.

Use the retrieved context to answer the user query in detail.
If the context does not contain enough information, say:
"I don’t have enough ARGO data to answer that confidently."

Question: {question}

Context:
{context}

Answer:
"""

prompt = PromptTemplate(
    template=CUSTOM_PROMPT,
    input_variables=["context", "question"],
)

# --------------------------
# VECTOR DB + EMBEDDINGS
# --------------------------
print(f"{Fore.CYAN} Loading embeddings: {EMBED_MODEL}{Style.RESET_ALL}")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

print(f"{Fore.CYAN} Connecting to Chroma at: {VECTOR_DB_PATH}{Style.RESET_ALL}")
vectorstore = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --------------------------
# LLaMA MODEL
# --------------------------
print(f"{Fore.CYAN} Loading LLaMA model from: {MODEL_PATH}{Style.RESET_ALL}")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=6,
    n_batch=256,
    max_tokens=256,
    temperature=0.2,
    verbose=False,
)

# --------------------------
# RETRIEVAL QA CHAIN
# --------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,   # keep sources for float info
)

# --------------------------
# MERGE EVERYTHING INTO ONE PARAGRAPH
# --------------------------
FINAL_PROMPT = PromptTemplate(
    template="""
Take the following ARGO analysis and summary, and merge them into a single clear paragraph.
The paragraph should read smoothly, not have headings, and must also include which floats and time periods were used.

Analysis:
{answer}

Short summary:
{summary}

Floats considered:
{floats}

Final paragraph:
""",
    input_variables=["answer", "summary", "floats"],
)

# --------------------------
# INTERACTIVE CHAT LOOP
# --------------------------
print(f"\n{Fore.BLUE}Ocean ARGO Chatbot is ready!")
print(" Type your question below. Type 'exit' or 'quit' to leave.")
print("=" * 70 + Style.RESET_ALL)

while True:
    try:
        query = input(f"\n{Fore.YELLOW}Your question: {Style.RESET_ALL}").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print(f"\n{Fore.GREEN}Exiting chatbot. Goodbye!{Style.RESET_ALL}")
            break

        # Step 1: RetrievalQA
        result = qa.invoke({"query": query})
        answer = result["result"]
        sources = result.get("source_documents", [])

        # Step 2: Extract float metadata
        floats_used = []
        for s in sources:
            meta = s.metadata
            floats_used.append(
                f"Float {meta.get('platform_number')} ({meta.get('year_month')})"
            )
        floats_summary = ", ".join(floats_used) if floats_used else "None"

        # Step 3: Short summary
        short_summary = llm.invoke(
            f"Summarize in 2 sentences: {answer}"
        )

        # Step 4: Merge into one final paragraph
        final_paragraph = llm.invoke(
            FINAL_PROMPT.format(answer=answer, summary=short_summary, floats=floats_summary)
        )

        # Output
        print(f"\n{Fore.GREEN}{final_paragraph}{Style.RESET_ALL}\n")
        print("=" * 70)

    except KeyboardInterrupt:
        print(f"\n\n{Fore.RED}Interrupted by user. Exiting.{Style.RESET_ALL}")
        break
    except Exception as e:
        print(f"\n{Fore.RED}Error occurred:{Style.RESET_ALL}")
        print(str(e))
        traceback.print_exc()
        print("=" * 70)
        continue
