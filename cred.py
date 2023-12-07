from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import re
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



DB_FAISS_PATH = r"/home/ec2-user/chatbot/kotak/vectordb"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
if user say Hi,hello just you say hello!!, Hi!!
If user say bye, just say that have a Nice day!! 

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#Conversation QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       memory=memory,
                                       combine_docs_chain_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = r"/home/ec2-user/chatbot/llama_model/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


# import required lib
from typing import Optional
import chainlit as cl
import csv

# Define fetch the password from .CSV file
def fetch_user_from_csv(username: str, password: str) -> Optional[cl.AppUser]:
    with open(r'/home/ec2-user/chatbot/Conversational_chatbot/new2/Upgrade/authentication.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            csv_username, csv_password, csv_role = row
            if username == csv_username and password == csv_password:
                return cl.AppUser(username=csv_username, role=csv_role, provider="credentials")
    return None

# Define the password authentication callback
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.AppUser]:
    return fetch_user_from_csv(username, password)


# Define the on_chat_start callback
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, this is Gensu, your intelligent assistant. Tell me what is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

# Define the on_message callback
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    res = await chain.acall(message.content, callbacks=[cb])
    
    # Check if 'answer' key is present in the result
    if 'answer' in res:
        answer = res["answer"]
        msg = cl.Message(content=f"➡️ {answer}")
        await msg.send()
    else:
        # Handle the case where 'answer' key is not present
        msg = cl.Message(content="No answer found.")
        await msg.send()

