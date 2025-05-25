import os, csv, matplotlib, tiktoken, streamlit as st, pandas as pd
from pandasai import SmartDataframe
from pandasai.connectors import PandasConnector
from pandasai.connectors.yahoo_finance import YahooFinanceConnector
from pandasai.llm import OpenAI, GoogleGemini
from pandasai.helpers.openai_info import get_openai_callback
from pandasai.responses.response_parser import ResponseParser

# ──────────────────────────────────────────────────────────────────────────
# 1) Monkey-patch class: force a valid model (e.g. "models/chat-bison-001")
import google.generativeai as genai

class PatchedGoogleGemini(GoogleGemini):
    def __init__(self, api_key: str, model: str = "models/gemini-2.0-flash-thinking-exp"):
        super().__init__(api_key=api_key)
        self._forced_model = model
        self.api_key = api_key  # so call() can reference it

    # Accept both prompt and context (PandasAI passes two args)
    def call(self, prompt, context=None, **kwargs) -> str:
        text_prompt = str(prompt)  # Convert Prompt-object → raw string
        genai.configure(api_key=self.api_key)
        gmodel = genai.GenerativeModel(self._forced_model)
        response = gmodel.generate_content(text_prompt)
        return response.text
# ──────────────────────────────────────────────────────────────────────────


class OutputParser(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def parse(self, result):
        if result['type'] == "dataframe":
            st.dataframe(result['value'])
        elif result['type'] == 'plot':
            st.image(result["value"])
        else:
            st.write(result['value'])
        return


def setup():
    st.header("Chat with your small and large datasets!", anchor=False, divider="red")
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)


def get_tasks():
    st.sidebar.header("Select a task", divider="rainbow")
    task = st.sidebar.radio(
        "Choose one:",
        (
            "Load from local drive, <200MB",
            "Load from local drive, 200MB+",
            "Load from Google Storage",
            "Yahoo Finance",
        ),
    )
    return task


def get_llm():
    st.sidebar.header("Select a LLM", divider='rainbow')
    llm = st.sidebar.radio("Choose a llm:", ("OpenAI", "Google Gemini"))
    return llm


def calculate_cost(df):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    cost = 0.0005
    strings = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
    token_count = [len(encoding.encode(rows)) for rows in strings]
    total_tokens = sum(token_count)
    st.write('Tokens:', total_tokens)
    st.write('Cost:', total_tokens * cost / 1000)


def main():
    setup()
    task = get_tasks()

    if task == "Load from local drive, <200MB":
        dataset = st.file_uploader("Upload your csv or xlsx file", type=['csv','xlsx'])
        if not dataset:
            st.stop()
        df = pd.read_csv(dataset, low_memory=False)

        calculate_cost(df)
        st.write("Data Preview:")
        st.dataframe(df.head())

        col_desc = st.radio(
            "Do you want to provide column descriptors?",
            ("Yes", "No")
        )
        if col_desc == "Yes":
            addon = st.text_input("Enter your column description, e.g. 'col1': 'unique id'")
        else:
            addon = "None"

        if addon:
            llm_choice = get_llm()

            # ─── PandasAI branch (unchanged) ───
            if llm_choice == "PandasAI":
                connector = PandasConnector({"original_df": df}, field_descriptions=addon)
                sdf = SmartDataframe(connector, {"enable_cache": False})
                prompt1 = st.text_input("Enter your question/prompt.")
                if not prompt1:
                    st.stop()
                response = sdf.chat(prompt1)
                st.write("Response")
                st.write("Loading response, please wait...")
                st.spinner("Processing your request...")
                st.write(response)
                st.divider()
                st.write("🧞‍♂️ Under the hood, the code that was executed:")
                st.code(sdf.last_code_executed)

            # ─── OpenAI branch (unchanged) ───
            elif llm_choice == "OpenAI":
                llm_instance = OpenAI(api_token=OPENAI_API_KEY)
                connector = PandasConnector({"original_df": df}, field_descriptions=addon)
                sdf = SmartDataframe(
                    connector,
                    {"enable_cache": False},
                    config={
                        "llm": llm_instance,
                        "conversational": False,
                        "response_parser": OutputParser
                    }
                )
                prompt2 = st.text_input("Enter your question/prompt.")
                if not prompt2:
                    st.stop()
                st.write("Response")
                with get_openai_callback() as cb:
                    response2 = sdf.chat(prompt2)
                    st.divider()
                    st.write("🧞‍♂️ Under the hood, the code that was executed:")
                    st.code(sdf.last_code_executed)
                    st.divider()
                    st.write("💰 Tokens used and your cost:")
                    st.write(cb)

            # ─── Google Gemini branch (monkey-patched) ───
            elif llm_choice == "Google Gemini":
                # Now we know "models/chat-bison-001" is available—use it:
                chosen_model = "models/gemini-2.0-flash-thinking-exp"  # ← replace with exactly what you saw

                llm_instance = PatchedGoogleGemini(
                    api_key=GOOGLE_API_KEY,
                    model=chosen_model
                )

                connector = PandasConnector(
                    {"original_df": df},
                    field_descriptions=addon
                )

                sdf = SmartDataframe(
                    connector,
                    {"enable_cache": False},
                    config={
                        "llm": llm_instance,
                        "conversational": False,
                        "response_parser": OutputParser
                    }
                )

                prompt3 = st.text_input("Enter your question/prompt.")
                if not prompt3:
                    st.stop()

                st.write("Response")
                response3 = sdf.chat(prompt3)

                st.divider()
                st.write("🧞‍♂️ Under the hood, the code that was executed:")
                st.code(sdf.last_code_executed)

    # … (the rest of your “200MB+”, Google Storage, and Yahoo Finance blocks stay the same) …

    if task == "Load from local drive, 200MB+":
        # (unchanged from before)
        filename = st.text_input(
            "Enter your file path including filename, e.g. /users/xyz/abc.csv, .csv files only"
        )
        if not filename:
            st.stop()
        df_large = pd.read_csv(filename, low_memory=False)
        st.write("Data Preview:")
        st.dataframe(df_large.head())
        col_desc = st.radio(
            "Do you want to provide column descriptors?",
            ("Yes", "No")
        )
        if col_desc == "Yes":
            addon = st.text_input("Enter your column description, e.g. 'col1': 'unique id'")
        else:
            addon = "None"

        if addon:
            llm_instance = OpenAI(api_token=OPENAI_API_KEY)
            connector = PandasConnector({"original_df": df_large}, field_descriptions=addon)
            sdf = SmartDataframe(
                connector,
                {"enable_cache": False},
                config={
                    "llm": llm_instance,
                    "conversational": False,
                    "response_parser": OutputParser
                }
            )
            prompt6 = st.text_input("Enter your question/prompt.")
            if not prompt6:
                st.stop()
            st.write("Response")
            with get_openai_callback() as cb:
                response6 = sdf.chat(prompt6)
                st.divider()
                st.write("🧞‍♂️ Under the hood, the code that was executed:")
                st.code(sdf.last_code_executed)
                st.divider()
                st.write("💰 Tokens used and your cost:")
                st.write(cb)

    if task == "Yahoo Finance":
        # (unchanged from before)
        stock_symbol = st.text_input("Enter a stock symbol, e.g. MSFT.")
        if not stock_symbol:
            st.stop()
        yahoo_connector = YahooFinanceConnector(stock_symbol)
        yahoo_df = SmartDataframe(yahoo_connector, config={"response_parser": OutputParser})
        prompt5 = st.text_input("Enter your prompt.")
        if not prompt5:
            st.stop()
        st.write("Response")
        response5 = yahoo_df.chat(prompt5)
        st.divider()
        st.code((yahoo_df.last_code_executed))


if __name__ == '__main__':
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    PANDASAI_API_KEY = os.environ.get('PANDASAI_API_KEY')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    projectid = os.environ.get('GOOG_PROJECT')
    matplotlib.use("Agg", force=True)
    main()
