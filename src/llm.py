from langchain.llms import OpenAI
import pandas as pd
import sqlite3
import numpy as np

from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.embeddings import OpenAIEmbeddings


class RetrievalLLM:
    """A class for retrieving and reranking offers using a large language model (LLM).

    Args:
        data_path (str): The path to the directory containing the data CSV files.
        tables (list[str]): A list of the names of the data CSV files.
        db_name (str): The name of the SQLite database to store the data in.
        openai_api_key (str): OpenAI API key.

    Attributes:
        data_path (str): The path to the directory containing the data CSV files.
        tables (list[str]): A list of the names of the data CSV files.
        db_name (str): The name of the SQLite database to store the data in.
        openai_api_key (str): OpenAI API key.
        db (SQLDatabase): SQLite database connection.
        llm (OpenAI): OpenAI LLM client.
        embeddings (OpenAIEmbeddings): OpenAI embeddings client.
        db_chain (SQLDatabaseChain): The SQL database chain with the LLM integrated.
    """

    def __init__(self, data_path, tables, db_name, openai_api_key):
        self.data_path = data_path
        self.tables = tables
        self.db_name = db_name
        self.openai_api_key = openai_api_key

        dfs = {}
        for table in self.tables:
            dfs[table] = pd.read_csv(f"{self.data_path}/{table}.csv")

        with sqlite3.connect(self.db_name) as local_db:
            for table, df in dfs.items():
                df.to_sql(table, local_db, if_exists="replace")

        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_name}")
        self.llm = OpenAI(
            temperature=0, verbose=True, openai_api_key=self.openai_api_key
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.db_chain = SQLDatabaseChain.from_llm(self.llm, self.db)

    def retrieve_offers(self, prompt):
        """Retrieves offers from the database using the LLM.

        Args:
            prompt (str): The prompt to use to retrieve offers.

        Returns:
            list[str]: A list of retrieved offers.
        """

        retrieved_offers = self.db_chain.run(prompt)
        return None if retrieved_offers == "None" else retrieved_offers

    def get_embeddings(self, documents):
        """Gets the embeddings of a list of documents using the LLM.

        Args:
            documents (list[str]): A list of documents.

        Returns:
            np.ndarray: A NumPy array containing the embeddings of the documents.
        """

        if len(documents) == 1:
            return np.asarray(self.embeddings.embed_query(documents[0]))
        else:
            embeddings_list = []
            for document in documents:
                embeddings_list.append(self.embeddings.embed_query(document))
            return np.asarray(embeddings_list)

    def parse_output(self, retrieved_offers, query):
        """Parses the output of the retrieve_offers() method and returns a DataFrame.

        Args:
            retrieved_offers (list[str]): A list of retrieved offers.
            query (str): The query that was used to retrieve the offers.

        Returns:
            pd.DataFrame: A DataFrame containing the match confidence and offers.
        """

        top_offers = retrieved_offers.split("#")

        query_embedding = self.get_embeddings([query])
        offer_embeddings = self.get_embeddings(top_offers)

        sim_scores = offer_embeddings.dot(query_embedding)
        sim_scores = [p * 100 for p in sim_scores]

        df = (
            pd.DataFrame({"Match Confidence %": sim_scores, "Offers": top_offers})
            .sort_values(by=["Match Confidence %"], ascending=False)
            .reset_index(drop=True)
        )
        df.index += 1
        return df
