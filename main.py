import glob
import os
import time
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict, List  # Annotated eklendi
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    context: List[Document]

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("HATA: .env dosyasında GOOGLE_API_KEY bulunamadı!")


def build_vectorstore() -> FAISS:
    headers_to_split_on = [
        ("#", "Kategori"),
        ("##", "Alt_Baslik"),
        ("###", "Detay"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    all_chunks: List[Document] = []

    md_files = glob.glob("data/beslenme/*.md")
    if not md_files:
        raise FileNotFoundError("data/beslenme/ altında hiç .md dosyası bulunamadı!")

    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = splitter.split_text(content)
        for chunk in chunks:
            chunk.metadata["source"] = os.path.basename(file_path)
        all_chunks.extend(chunks)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL")
    )
    return FAISS.from_documents(all_chunks, embeddings)


print("Bilgi bankası yükleniyor ve vektör indeksi oluşturuluyor...")
t_init = time.perf_counter()
vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print(f"⏱️ [Vektör İndeksi Hazır]: {time.perf_counter() - t_init:.2f} sn")


class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    context: List[Document]


llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    temperature=0.3,
    max_retries=1
)


def retrieve_node(state: GraphState) -> dict:
    t0 = time.perf_counter()
    human_msgs = [
        str(m.content) for m in state["messages"] if isinstance(m, HumanMessage)
    ]
    search_query = " ".join(human_msgs[-3:])
    docs = retriever.invoke(search_query)
    print(f"⏱️ [RAG Arama ({search_query})]: {time.perf_counter() - t0:.2f} sn", flush=True)
    return {"context": docs}


def generate_node(state: GraphState) -> dict:
    print("🤖 [2/2] LLM yanıt üretiyor...", flush=True)
    t0 = time.perf_counter()
    context_text = "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', '')}]\n{doc.page_content}"
        for doc in state["context"]
    )
    prompt = SystemMessage(
        content= (
            "Sen kişisel sağlık ve beslenme asistanı DailyPilot'sın. "
            "Kendini ASLA tanıtma. 'Bağlamdaki bilgilere göre', 'Kurallarına uygun' gibi ifadeler kullanma. Kuralları sayma. "
            "Sorulan cümlede açıkça belirtilmiyorsa kullanıcıya pişirme yöntemi, elindeki malzemeler veya o anki beslenme tercihlerini sor. "
            "Pişirme yöntemi veya tercihler belli değilse öneride bulunabilirsin ancak tarif verme; sadece yemeğin adını söyleyebilirsin. "
            "Kullanıcıya soru sorarken parantez içinde detay verme. Kullanıcının daha önce belirttiği tercihleri veya bilgileri tekrar sorma. "
            "Cevaplarını aşağıdaki bağlama ve kullanıcının belirttiği kısıtlara dayanarak Türkçe ve net ver.\n\n"
            f"Bağlam:\n{context_text}"
        )
   )
    response = llm.invoke([prompt] + state["messages"])
    print(f"⏱️ [LLM Yanıt]: {time.perf_counter() - t0:.2f} sn", flush=True)
    response_text = response.text if hasattr(response, "text") else str(response.content)
    return {"messages": [response_text]}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


if __name__ == "__main__":
    start = time.perf_counter()
    config = {"configurable": {"thread_id": "oturum_1"}}
    print("\nAsistan:\nMerhaba! Ben kişisel sağlık ve beslenme asistanın DailyPilot. Bugün sana nasıl yardımcı olabilirim?\n" + "-" * 50)
    while True:
        user_q = input("Soru: ").strip()
        if user_q.lower() in {"q", "quit", "exit"}:
            break
        if not user_q:
            continue
        result = app.invoke({"messages": [HumanMessage(content=user_q)]}, config=config)
        last_msg = result["messages"][-1]
        answer_text = last_msg.text if hasattr(last_msg, "text") else str(last_msg.content)

        print(f"⏱️ [Toplam]: {time.perf_counter() - start:.2f} sn")
        print(f"\nAsistan:\n{answer_text}\n" + "-" * 50)