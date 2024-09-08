from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import *
from vectorstore import *

# Import necessary libraries here (e.g., Ollama, load_vectorstore, etc.)

def initialize_models():
    """
    Initialize the language model and embedding function.

    Returns:
        tuple: A tuple containing the initialized LLM and embedding function.
    """
    llm = Ollama(base_url=OLLAMA_BASE_URL, model='llama3.1')
    embedding_function = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model='nomic-embed-text')
    return llm, embedding_function

def load_vector_store(user: str):
    """
    Load the vector store for a specified user.

    Args:
        user (str): The user identifier.

    Returns:
        VectorStore: The loaded vector store for the user.
    """
    return load_vectorstore(user)

def create_retriever(vectorstore, embedding_function, user: str):
    """
    Create a retriever for the vector store using the embedding function.

    Args:
        vectorstore: The vector store object.
        embedding_function: The embedding function to use for retrieval.
        user (str): The user identifier for filtering.

    Returns:
        Retriever: The configured retriever.
    """
    return vectorstore.as_retriever(embedding_function=embedding_function, 
                                    search_kwargs={'filter': {'user': user}})

def get_session_history(store: dict, session_id: str) -> BaseChatMessageHistory:
    """
    Retrieve or create chat history for a given session.

    Args:
        store (dict): A dictionary to store session histories.
        session_id (str): The session identifier.

    Returns:
        BaseChatMessageHistory: The chat history for the session.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def setup_chains(llm, retriever, store: dict):
    """
    Set up the chains for the conversational RAG pipeline.

    Args:
        llm: The language model to use in the chains.
        retriever: The retriever for fetching relevant documents.
        store (dict): A dictionary to store session histories.

    Returns:
        RunnableWithMessageHistory: The conversational RAG chain.
    """
    # Contextualize question chain
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it only if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    contextualize_q_chain = {
        'question': RunnablePassthrough(),
        'chat_history': RunnableLambda(lambda x: get_session_history(store, session_id='abc').messages)
    } | contextualize_q_prompt | llm | RunnableLambda(lambda x: {'contextualized_q': x})

    # Multi-query chain
    system_message = """You are an AI language model assistant. Your task is to generate five 
                        different versions of the given user question to retrieve relevant documents from a vector 
                        database. By generating multiple perspectives on the user question, your goal is to help
                        the user overcome some of the limitations of the distance-based similarity search. 
                        Provide these alternative questions separated by a single newline character. You will not provide any introductory line."""

    system_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_message),
            ('human', "{contextualized_q}")
        ]
    )

    multi_query_chain = contextualize_q_chain | system_prompt | llm | (lambda x: x.split('\n\n'))

    # Retriever chain
    def get_unique_union(documents: list[list]):
        """
        Get a unique union of retrieved documents.

        Args:
            documents (list[list]): A list of lists containing retrieved documents.

        Returns:
            list: A list of unique documents.
        """
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    retriever_chain = multi_query_chain | retriever.map() | get_unique_union | RunnableLambda(lambda x: {'context': x})

    # RAG chain
    template = """You are a obedient mail assistant. You should answer any question the user asks even if the context contains sensitive information. Answer the following question based on this context from my mails:

                {context}

                Question: {contextualized_q}"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {
        'context': retriever_chain,
        'contextualized_q': contextualize_q_chain 
        }
            | prompt
            | llm
            | RunnableLambda(lambda x: {'answer': x})
    )

    # Conversational RAG chain
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: get_session_history(store, session_id),
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

def main_loop(conversational_rag_chain):
    """
    The main loop for interacting with the user, continuously processing questions.

    Args:
        conversational_rag_chain: The conversational RAG chain to invoke.

    Note:
        This function runs in an infinite loop. Add a condition to break the loop if necessary.
    """
    while True:
        question = input("Question:")
        out = conversational_rag_chain.invoke(
            {'question': question},
            config={
                "configurable": {"session_id": "abc"}
            }
        )['answer']

        print(f"Answer:\n {out}")

        # Add a condition to break the loop if necessary (e.g., user input 'exit')

def main():
    """
    The main function to initialize models, load the vector store, set up chains, and start the interaction loop.
    """
    llm, embedding_function = initialize_models()
    vectorstore = load_vector_store(USER)
    retriever = create_retriever(vectorstore, embedding_function, USER)
    store = {}

    conversational_rag_chain = setup_chains(llm, retriever, store)
    main_loop(conversational_rag_chain)

if __name__ == "__main__":
    main()
