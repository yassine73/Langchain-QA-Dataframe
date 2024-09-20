import streamlit as st
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage


def main():
    st.set_page_config(page_title="Q/A Dataframe 🗂️")
    st.header("Chat with your CSV File 🗂️")
    
    def chain_pandas(file):
        llm = ChatOllama(model="llama2", temperature=0)
        # llm = ChatOpenAI(model="gpt-3.5-turbo")
        
        template = """
        You are a pandas generator that take questions and generates a python pandas code.
        Based on table below. Write a pandas query that would answer the user's question.
        Don't add text, Return only the pandas query as python code.
        Table: {table}
        
        Chat history: {chat_history}
        Question: {question}
        Pandas query:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def to_dataframe(_):
            return pd.DataFrame(file)
        
        return RunnablePassthrough.assign(table=to_dataframe) | prompt | llm | StrOutputParser()
    

    def response_generator(pandas_chain, file):
        llm = ChatOllama(model="llama2", temperature=0.7)
        # llm = ChatOpenAI(model="gpt-3.5-turbo")
        
        template = """
            Based on table below, pandas query, pandas response. Generate a natural language response.
            Show the response without neither pandas code or the way you found it
            Table: {table}
            
            Question: {question}
            Pandas query: {query}
            Pandas response: {response}
            Answer: 
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        def to_dataframe(_):
            return pd.DataFrame(file)
        
        def pd_exec(query):
            df = pd.DataFrame(file)
            return eval(query.replace("`", ""))
        
        return (
            RunnablePassthrough.assign(query=pandas_chain).assign(
                table = to_dataframe,
                response = lambda res: pd_exec(res["query"])
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
    chat_history = []
    user_data = st.file_uploader("Upload File", type=["csv"], accept_multiple_files=False)
    if user_data is not None:
        data = pd.read_csv(user_data)
        user_input = st.text_input("Ask a question about you CSV: ")
        if user_input:
            pd_chain = chain_pandas(data)
            final_chain = response_generator(pd_chain, data)
            answer = final_chain.invoke({"question" : user_input, "chat_history": chat_history})
            st.write(answer)
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))



if __name__ == "__main__":
    main()