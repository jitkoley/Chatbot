from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Optional
import csv

DB_FAISS_PATH = r"/home/ec2-user/chatbot/kotak/vectordb"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that I don't know, don't try to make up an answer.

 
Chat History: {chat_history}
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



#Conversation QA Chain
def retrieval_qa_chain(llm, prompt, db):
    # ConversationBufferMemory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,output_key='answer')
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       memory=memory,
                                       combine_docs_chain_kwargs={'prompt': prompt},
                                       return_source_documents=True
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


# Initialize the Chainlit client
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

# Chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    #chain_qa = model.qa_bot() #
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, this is Gensu, your intelligent assistant. Tell me what is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    # chain_qa = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    res = await chain.acall(message.content, callbacks=[cb])
   # res2 = await chain_qa.acall(message, callbacks=[cb])
    source = res["source_documents"]
    data = str(source)
    with open('/home/ec2-user/chatbot/Upgrade/sources.txt', 'w') as file:
        a = file.write(data)
    with open('/home/ec2-user/chatbot/Upgrade/sources.txt', 'r') as file:
        text = file.read()
    text = text.replace(r'\n', '').strip()
    if source:
        answer = res["answer"] + f"\n\n➡️ Sources:" + text
        msg = cl.Message(content=f"➡️ {answer}")
        await msg.send()
    else:
        # Handle the case where 'answer' key is not present
        msg = cl.Message(content="No answer found.")
        await msg.send()
