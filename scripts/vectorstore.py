from psql_functions import psql_cursor, user_exists, log_user_activity, get_last_fetch
from mail_functions import retrieve_mails
from config import *

import sys
import shutil
import time
from datetime import datetime, timedelta
from typing import List

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

def load_vectorstore(username: str):
    """
    Checks last vectorstore update for user, and if no records exists creates new ones.
    Logs latest update for user in Postgres (TABLE user_activity)

    Args:
        username (str): Gmail ID of user.
    """
    def update_vectorstore(username: str, date_lim: datetime, num_threads: int= 3):
        """
        Updates vectorstore for user with all emails received between date_lim and current datetime.

        Args:
            username (str): Gmail ID of user.
            date_lim (datetime): Starting datetime from which emails should be retrieved for processing
            num_threads (int, optional): Number of threads to use for processing the emails in parallel. Defaults to 3.tr
        
        Returns:
            Chroma: The updated vectorstore object.
        """
        def concurrent_vectorstore_load(splits: List, num_threads: int= 3):
            """
            Concurrently processes and updates the vectorstore using multiple threads.

            Args:
                splits (List): The email data to be processed.
                num_threads (int, optional): Number of threads for parallel processing. Defaults to 3.

            Returns:
                Chroma: The updated vectorstore object.
            """
            start = time.time()

            # Split the data into chunks for each thread
            chunk_size = len(splits) // num_threads

            subsets = [(splits[i*chunk_size:(i+1)*chunk_size], metadata[i*chunk_size:(i+1)*chunk_size], i) for i in range(num_threads)]

            # Handle any remaining items that weren't evenly divided
            if len(splits) % num_threads != 0:
                subsets.append((splits[num_threads*chunk_size:], metadata[num_threads*chunk_size:], num_threads-1))

            # Initialize a list to hold all subdirectory paths
            subdirs = []

            print('Initializing Threads')

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                subdirs = list(executor.map(lambda p: process_subset(*p), subsets))
            
            # Aggregate all the subdirectories into the final vectorstore
            final_vectorstore = Chroma(embedding_function=oembed, 
                                        persist_directory=VECTORESTORE_DIR)

            print('Threads Returned')

            print("Collecting Results")

            old_stdout = sys.stdout # backup current stdout
            sys.stdout = open(os.devnull, "w")

            for subdir,sub_vectorstore in subdirs:
                subdir_data=sub_vectorstore._collection.get(include=['documents','metadatas','embeddings'])
                final_vectorstore._collection.add(
                    embeddings=subdir_data['embeddings'],
                    metadatas=subdir_data['metadatas'],
                    documents=subdir_data['documents'],
                    ids=subdir_data['ids']
                )
                try:
                    shutil.rmtree(subdir)
                except:
                    continue

            sys.stdout = old_stdout

            end = time.time()
            print(f"Time taken: {end - start:.4f} seconds")

            return final_vectorstore

        def process_subset(subset_texts: List[str], subset_metadata: List[dict], thread_id: int):
            """
            Processes a subset of emails in a specific thread and creates a vectorstore for it.

            Args:
                subset_texts (List[str]): List of email texts to be processed.
                subset_metadata (List[dict]): Corresponding metadata for the email texts.
                thread_id (int): Identifier for the thread processing this subset.

            Returns:
                Tuple[str, Chroma]: The directory path where the subset's vectorstore is saved and the vectorstore object.
            """
            subset_vectorstore_dir = os.path.join(VECTORESTORE_DIR, f'thread_{thread_id}')
            sub_vectorstore = Chroma.from_texts(texts=subset_texts, 
                                            embedding=oembed,
                                            persist_directory=subset_vectorstore_dir,
                                            metadatas=subset_metadata)
            return (subset_vectorstore_dir,sub_vectorstore)
    
        print("Updating vectorstore....This might take some time.")

        all_mails = retrieve_mails(username,date_lim)
        docs = '\n'.join(all_mails)
        
        oembed = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model="nomic-embed-text")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=600, 
                                                                                chunk_overlap=100)
        splits = text_splitter.split_text(docs)
        metadata = [{'user': username} for _ in range(len(splits))]

        final_vectorstore = concurrent_vectorstore_load(splits,num_threads)

        print("Vectorstore updated")

        log_user_activity(conn,cursor,username)

        return final_vectorstore

    conn, cursor = psql_cursor(dbname = PSQL_DB_NAME,
                        user = PSQL_USER,
                        password = PSQL_PASSWORD,
                        host = PSQL_HOST,
                        port = PSQL_PORT)
    
    oembed = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model="nomic-embed-text")

    if user_exists(cursor, username):
        date_lim = get_last_fetch(cursor,username)

        flag = input(f"Your emails were last fetched on {date_lim}\nDo you want to update them?[y\\n]")
        if flag.lower() == 'n':
            return Chroma(embedding_function=oembed,
                          persist_directory=VECTORESTORE_DIR)
        else:
            return update_vectorstore(username,date_lim = date_lim)
    
    return update_vectorstore(username,date_lim = datetime.now() - timedelta(days=30))


if __name__ == "__main__":
    load_vectorstore(USER)