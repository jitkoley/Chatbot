from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os
#..................
from langchain.document_loaders import S3DirectoryLoader
from langchain.document_loaders import S3FileLoader
import boto3
import os

# 
def download_all_pdfs_from_s3(bucket_name, s3_directory, local_directory):
    s3 = boto3.client(
        's3',
        aws_access_key_id="AKIA2HTZEBEIIJ3F", # my AWS Access key
        aws_secret_access_key="t5GQ1nlV12LWsPGVjY+jIREfTNhR1iOm0u2" # my AWS secret key
    )

    # List objects in the specified S3 bucket directory
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_directory)

    # Extract the keys (file paths) for PDF files
    pdf_keys = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf')]

    # Download each PDF file to the local directory
    for pdf_key in pdf_keys:
        local_path = os.path.join(local_directory, os.path.basename(pdf_key))
        s3.download_file(bucket_name, pdf_key, local_path)
        print(f"Downloaded: {pdf_key} to {local_path}")

if __name__ == "__main__":
    bucket_name = "genaibasechatbot" # my Aws bucket name
    s3_directory = "dataset/" # AWS bucket directory path
    local_directory = "/home/ec2-user/chatbot/Conversational_chatbot/new2/Upgrade/data" # my local machine directory path

    # Create the local directory if it doesn't exist
    os.makedirs(local_directory, exist_ok=True)

    # Download all PDF files from S3 to the local directory
    download_all_pdfs_from_s3(bucket_name, s3_directory, local_directory)


DATA_PATH = r"/home/ec2-user/chatbot/Conversational_chatbot/new2/Upgrade/data"
DB_FAISS_PATH = r"/home/ec2-user/chatbot/Conversational_chatbot/new2/Upgrade/vectorstore"

# Create vector database
def create_vector_db():
    documents = []
    for file in os.listdir(DATA_PATH):  # loading pdf file
        if file.endswith(".pdf"):
            pdf_path = DATA_PATH+"//"+ file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'): # loading doc file
            doc_path = DATA_PATH+"//"+ file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):       # loading txt file
            text_path = DATA_PATH+"//"+ file
            loader = TextLoader(text_path)
            documents.extend(loader.load())
        elif file.endswith('.csv'):      # loading csv file
            csv_path = DATA_PATH + "\\" + file
            loader = CSVLoader(csv_path)
            documents.extend(loader.load())
    # print(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()
