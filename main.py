from fastapi import FastAPI
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

app = FastAPI()

class Question(BaseModel):
    question: str
    image: str = None

@app.on_event("startup")
def build_index():
    global qa
    with open("data/tds_posts.txt") as f:
        text = f.read()
    doc = Document(page_content=text)
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents([doc])
    db = FAISS.from_documents(chunks, OpenAIEmbeddings())
    qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo"), retriever=db.as_retriever())

@app.post("/api/")
async def ask(q: Question):
    answer = qa.run(q.question)
    return {"answer": answer, "links": []}

