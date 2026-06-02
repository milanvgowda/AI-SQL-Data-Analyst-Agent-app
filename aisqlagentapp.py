import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AI SQL Data Analyst Agent", layout="wide")
st.title("🤖 AI SQL Data Analyst Agent")

file = st.file_uploader("Upload CSV File", type=["csv"])

if file:

    df = pd.read_csv(file)

    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    engine = create_engine("sqlite:///data.db")
    df.to_sql("data", engine, if_exists="replace", index=False)

    db = SQLDatabase(engine)

    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key
    )

    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False)

    question = st.text_input("💬 Ask your question")

    if question:

        q = question.lower()
        numeric_df = df.select_dtypes(include=['number'])

        with st.spinner("🤖 AI Thinking..."):

            try:

                # ---------------- ADVANCED ANALYSIS (PYTHON ONLY) ----------------

                if "heatmap" in q or "correlation" in q:

                    corr = numeric_df.corr()

                    fig, ax = plt.subplots()
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)

                    insight = llm.invoke(f"Give insights from this correlation matrix:\n{corr.to_string()}")
                    st.subheader("🧠 AI Insight")
                    st.write(insight.content)

                elif "pairplot" in q:

                    fig = sns.pairplot(numeric_df)
                    st.pyplot(fig)

                    insight = llm.invoke("Explain patterns in this pairplot")
                    st.subheader("🧠 AI Insight")
                    st.write(insight.content)

                elif "distribution" in q:

                    col = None

                    for c in df.columns:
                        clean_c = c.lower().replace("_", "").replace(" ", "")
                        clean_q = q.replace(" ", "")
                        if clean_c in clean_q:
                            col = c
                            break

                    if col is None:
                        st.warning("Please specify column like 'distribution of age'")
                    else:
                        fig = px.histogram(df, x=col)
                        st.plotly_chart(fig)

                        insight = llm.invoke(f"Explain distribution of {col}")
                        st.subheader("🧠 AI Insight")
                        st.write(insight.content)

                elif "box" in q:

                    col = None
                    for c in df.columns:
                        if c.lower() in q:
                            col = c
                            break

                    if col:
                        fig = px.box(df, y=col)
                        st.plotly_chart(fig)

                        insight = llm.invoke(f"Explain outliers in {col}")
                        st.subheader("🧠 AI Insight")
                        st.write(insight.content)
                    else:
                        st.warning("Specify column")

                elif "scatter" in q:

                    cols = numeric_df.columns
                    if len(cols) >= 2:
                        fig = px.scatter(df, x=cols[0], y=cols[1])
                        st.plotly_chart(fig)

                        insight = llm.invoke(f"Explain relationship between {cols[0]} and {cols[1]}")
                        st.subheader("🧠 AI Insight")
                        st.write(insight.content)
                    else:
                        st.warning("Not enough numeric columns")

                # ---------------- BASIC ANALYSIS (SQL ONLY) ----------------

                else:

                    result = db_chain.invoke({"query": question})
                    output = result["result"]

                    st.subheader("🧠 AI Generated SQL")

                    if "SQLQuery:" in output:

                        sql_query = output.split("SQLQuery:")[1].split("SQLResult:")[0].strip()

                        # FIX SQL ERROR (REMOVE ```sql)
                        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

                        st.code(sql_query, language="sql")

                        result_df = pd.read_sql(sql_query, engine)

                        st.subheader("📋 Result Table")
                        st.dataframe(result_df)

                        if result_df.shape[1] >= 2:
                            fig = px.bar(
                                result_df,
                                x=result_df.columns[0],
                                y=result_df.columns[1]
                            )
                            st.subheader("📊 Visualization")
                            st.plotly_chart(fig)

                        insight = llm.invoke(f"Explain insights from this data:\n{result_df.to_string()}")
                        st.subheader("🧠 AI Insight")
                        st.write(insight.content)

                    else:
                        st.write(output)

            except Exception as e:
                st.error(e)