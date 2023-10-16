import streamlit as st
from langchain.llms import OpenAI
import pandas as pd
import sqlite3
import os
import numpy as np

from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.embeddings import OpenAIEmbeddings

st.title("Search for offers 🔍")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

def parse_output(retrieved_offers):
    if retrieved_offers == "None":
        return "No relevant offers found."
    else:
        top_offers = retrieved_offers.split("#")

        query_embedding = embeddings.embed_query(query)
        query_embedding = np.asarray(query_embedding)

        offer_embeddings = []
        for offer in top_offers:
            offer_embeddings.append(embeddings.embed_query(offer))
        offer_embeddings = np.asarray(offer_embeddings)

        sim_scores = offer_embeddings.dot(query_embedding)

        df = (
            pd.DataFrame({"Match Confidence": sim_scores, "Offers": top_offers})
            .sort_values(by=["Match Confidence"], ascending=False)
            .reset_index(drop=True)
        )
        return df


with st.form("search_form"):
    query = st.text_input("Search for offers by category, brand, or retailer.")
    submitted = st.form_submit_button("Search")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    else:
        DATA_PATH = "../data"
        keys = ("brand_category", "categories", "offer_retailer")
        dfs = {}
        for k in keys:
            dfs[k] = pd.read_csv(f"{DATA_PATH}/{k}.csv")

        DB_NAME = "offer_db.sqlite"
        with sqlite3.connect(DB_NAME) as local_db:
            for key, df in dfs.items():
                df.to_sql(key, local_db, if_exists="replace")

        db = SQLDatabase.from_uri(f"sqlite:///{DB_NAME}")
        llm = OpenAI(temperature=0, verbose=True, openai_api_key=openai_api_key)
        db_chain = SQLDatabaseChain.from_llm(llm, db)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        if submitted:
            prompt = f"""
    You are given a query and your job is to retrieve relevant offers stored in the `offer_retailer` table under the `OFFER` field. 
    The query might be in mixed case, so search for capitalized versions of the query too.
    Importantly, you might need to use information from other tables in the database namely: `brand_category`, `categories`, `offer_retailer` to retrieve the correct offer(s).
    Do not hallucinate offers. If offer does not exist in the `offer_retailer` table, return the string: `NONE`.
    Else, if you are able to retrieve offers from the `offer_retailer` table, separate each offer with the delimiter `#`. For example, here is what the output should look like: `offer1#offer2#offer3`.
    If the SQLResult is empty, return `None`. Do not generate any offers.
    Here is the query: `{query}`
    """
            retrieved_offers = db_chain.run(prompt)
            if retrieved_offers == "None":
                st.text("No relevant offers found.")
            else:
                st.table(parse_output(retrieved_offers))
