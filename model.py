from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from typing import cast

DB_FAISS_PATH = 'vectorstore/db_faiss'

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
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

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
    cl.user_session.set("runnable", chain)
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Cybersecurity Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    # chain = cl.user_session.get("chain") 
    # cb = cl.AsyncLangchainCallbackHandler(
    #     stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    # )
    # cb.answer_reached = True
    # res = await chain.acall(message.content, callbacks=[cb])
    # this is deprecated any other option to get the answer?
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable
    msg = cl.Message(content="")
    final_answer = ""
    sources = []

    # Stream tokens from the runnable
    async for chunk in runnable.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )]),
    ):
        # Extract the token from the chunk (if it's a string) and stream it
        token = chunk.get("result", "")  # Get the result token if present
        if token:
            await msg.stream_token(token)
        
        # Handle the final result and sources if included in the chunk
        # if "result" in chunk:
            # final_answer += chunk["result"]
        if "source_documents" in chunk:
            sources.extend(chunk["source_documents"])
        else:
            if "result" in chunk:
                final_answer += chunk["result"]
    response = final_answer.strip()

    # Add sources to the response
    if sources:
        response += "\n\nSources:\n"
        for source in sources:
            page_content = source.page_content or "No content available"
            metadata = source.metadata or {}
            metadata_str = "\n".join([f"{key}: {value}" for key, value in metadata.items()])
            response += f"\Source Details : \n{page_content}\nMetadata:\n{metadata_str}\n"
    else:
        response += "\n\nNo sources found."

    # Send the complete response
    await cl.Message(content=response).send()

