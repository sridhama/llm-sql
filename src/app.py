import streamlit as st
from llm import RetrievalLLM

DATA_PATH = '../data'
TABLES = ('brand_category', 'categories', 'offer_retailer')
DB_NAME = 'offer_db.sqlite'
PROMPT_TEMPLATE = """
                You are given a query and your job is to retrieve relevant offers stored in the `offer_retailer` table under the `OFFER` field. 
                The query might be in mixed case, so search for capitalized versions of the query too.
                Importantly, you might need to use information from other tables in the database namely: `brand_category`, `categories`, `offer_retailer` to retrieve the correct offer(s).
                Do not hallucinate offers. If offer does not exist in the `offer_retailer` table, return the string: `NONE`.
                Else, if you are able to retrieve offers from the `offer_retailer` table, separate each offer with the delimiter `#`. For example, here is what the output should look like: `offer1#offer2#offer3`.
                If the SQLResult is empty, return `None`. Do not generate any offers.
                Here is the query: `{}`
                """

st.title("Search for offers 🔍")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

with st.form("search_form"):
    query = st.text_input("Search for offers by category, brand, or retailer.")
    submitted = st.form_submit_button("Search")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    else:
        retrieval_llm = RetrievalLLM(
            data_path=DATA_PATH,
            tables=TABLES,
            db_name=DB_NAME,
            openai_api_key=openai_api_key,
        )
        if submitted:
            retrieved_offers = retrieval_llm.retrieve_offers(
                PROMPT_TEMPLATE.format(query)
            )
            if not retrieved_offers:
                st.text("No relevant offers found.")
            else:
                st.table(retrieval_llm.parse_output(retrieved_offers, query))