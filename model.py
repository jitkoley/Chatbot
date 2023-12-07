from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import re
#
from langchain.chains import ConversationalRetrievalChain

###############################################
# from langchain.document_loaders import S3FileLoader
# from langchain.document_loaders import S3FileLoader
################################################

#load the data from s3 bucket
# loader = S3FileLoader("genaibasechatbot", "s3://genaibasechatbot/Sreeram_Intern.pdf")
# loader.load()
##################################################


DB_FAISS_PATH = r"/home/ec2-user/chatbot/kotak/vectordb"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

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

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = r"/home/ec2-user/chatbot/llama_model/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 652,
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

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, this is Gensu, your intelligent assistant. Tell me what is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    # cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]
    
    data = str(sources)
    """
    print("******************")
    sources = str(sources[0])
        sources = sources.replace(r"\n"," ")
    for doc in sources:
        b = doc.page_content.replace(r'\n', ' ').strip()
    """
    
    with open('/home/ec2-user/chatbot/Upgrade/sources.txt', 'w') as file:
        a = file.write(data)
    with open('/home/ec2-user/chatbot/Upgrade/sources.txt', 'r') as file:
        text = file.read()
    text = text.replace(r'\n', '').strip()
    

    if sources:
        answer += f"\n\nSources:" + text
    else:
        answer += "\n\nNo sources found"

    await cl.Message(content=answer, disable_human_feedback=True).send()

