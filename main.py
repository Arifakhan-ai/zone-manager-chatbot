import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
import pandas as pd
import sqlite3
from dotenv import load_dotenv # type: ignore
from langchain_openai import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from langchain_chroma import Chroma
from chromadb import PersistentClient
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from pydantic import BaseModel, Field
from typing import List
from langchain_community.tools import QuerySQLDatabaseTool
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from fordllm.utils import TokenFetcher
from openai import OpenAI
import os
from fordllm.utils import TokenFetcher
import streamlit as st



# Streamlit page configuration
st.set_page_config(
    page_title="Zone-Manager Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("Zone-Manager Chatbot")

# Environment Variables Loading
def load_environment_variables():
    """Load environment variables from .env file or cloud environment"""
    if os.path.exists('.env'):
        # Local development
        load_dotenv()
        st.sidebar.success("🔧 Local development mode")
    else:
        # Cloud deployment
        st.sidebar.info("☁️ Cloud deployment mode")
    
    # Get all environment variables
    env_vars = {
        'FORDLLM_CLIENT_ID': os.getenv('FORDLLM_CLIENT_ID'),
        'FORDLLM_CLIENT_SECRET': os.getenv('FORDLLM_CLIENT_SECRET'),
        'LLM_TOKEN_ENDPOINT': os.getenv('LLM_TOKEN_ENDPOINT'),
        'LLM_SCOPE': os.getenv('LLM_SCOPE'),
        'API_HOST': os.getenv('API_HOST'),
        'PROXY_ENDPOINT': os.getenv('PROXY_ENDPOINT'),
        'LLM_API_ENDPOINT': os.getenv('LLM_API_ENDPOINT'),
        'MODEL': os.getenv('MODEL'),
        'LANGFUSE_SECRET_KEY': os.getenv('LANGFUSE_SECRET_KEY'),
        'LANGFUSE_PUBLIC_KEY': os.getenv('LANGFUSE_PUBLIC_KEY'),
        'LANGFUSE_HOST': os.getenv('LANGFUSE_HOST'),
    }
    
    # Check for required variables
    required_vars = ['FORDLLM_CLIENT_ID', 'FORDLLM_CLIENT_SECRET']
    missing_vars = [var for var in required_vars if not env_vars[var]]
    
    if missing_vars:
        st.error(f"⚠️ Missing required environment variables: {', '.join(missing_vars)}")
        st.stop()
    
    return env_vars

# Load environment variables
env_vars = load_environment_variables()

# Set environment variables for fordllm
os.environ['FORDLLM_CLIENT_ID'] = env_vars['FORDLLM_CLIENT_ID']
os.environ['FORDLLM_CLIENT_SECRET'] = env_vars['FORDLLM_CLIENT_SECRET']

# Initialize TokenFetcher
try:
    token_fetcher = TokenFetcher()
    openai_api_key = token_fetcher.token
except Exception as e:
    st.error(f"⚠️ Failed to initialize TokenFetcher: {str(e)}")
    st.stop()

if not openai_api_key:
    st.error("⚠️ OpenAI API key not found! Please check your environment variables.")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(
        api_key=openai_api_key,
        base_url="https://api.pivpn.core.ford.com/fordllmapi/api/v1",
    )
except Exception as e:
    st.error(f"⚠️ Failed to initialize OpenAI client: {str(e)}")
    st.stop()

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4.1"

# Initialize Langfuse
try:
    langfuse = Langfuse(
        public_key=env_vars['LANGFUSE_PUBLIC_KEY'],
        secret_key=env_vars['LANGFUSE_SECRET_KEY'],
        host=env_vars['LANGFUSE_HOST']
    )
    langfuse_handler = CallbackHandler()
except Exception as e:
    st.warning(f"⚠️ Langfuse initialization failed: {str(e)}. Continuing without Langfuse tracking.")
    langfuse_handler = None

# Initialize LLM
try:
    callbacks = [langfuse_handler] if langfuse_handler else []
    llm = ChatOpenAI(
        model="gpt-4.1",
        openai_api_base="https://api.pivpn.core.ford.com/fordllmapi/api/v1",
        openai_api_key=token_fetcher.token,
        temperature=0,
        streaming=True,
        callbacks=callbacks
    )
except Exception as e:
    st.error(f"⚠️ Failed to initialize LLM: {str(e)}")
    st.stop()

# Answer prompt template
answer_prompt = PromptTemplate.from_template(
    """You are a helpful assistant. Given the user's question, SQL query, and SQL result,
         generate a clear and concise final answer based **only on the SQL result**.

         Important:
         - DO NOT assume the current year manually. Use the years shown in the SQL result.
         - DO NOT assume FDC as Floor Days of Capital. FDC/fdc means Forward Days Cover.
         - If data is shown for years like 2024 or 2025, treat those as the actual current or previous years.
         - Never assume it's 2023 or 2022 unless that's reflected in the SQL result.
         - Be factual, do not speculate on possible data errors unless explicitly asked.
         
         When comparing vehicle sales year over year:
         - If both years have sales data, compare the values:
         - Use "increased from X to Y" when 2025 sales > 2024 sales.
         - Use "decreased from X to Y" when 2025 sales < 2024 sales.
         - Use "remained the same at X units" if values are equal.
         - If one year has missing data, say "no data available for comparison".
         Ensure the summary accurately reflects the direction of change based on the numeric values.

         Question: {question}
         SQL Query: {query}
         SQL Result: {result}

         Answer:"""
)

# Few-shot examples
examples =[
  {
    "input": "What is alpine Ford’s objectives for December 2024?",
    "query": "SELECT d.dealer_name, d.objective FROM dealers d WHERE LOWER(d.dealer_name) LIKE 'alpine ford' AND d.objective_month = '2024-12-01' AND objective_group IN ('Car', 'SUV', 'Truck');"
  },
  {
    "input": "what is the raptor sales mix and sales of city Ford on Ranger for Qtr 3 2024?",
    "query": "SELECT d.dealer_name, COUNT(DISTINCT CASE WHEN LOWER(s.series) LIKE '%raptor%' THEN v.vehicle_key END) as raptor_sales, COUNT(DISTINCT CASE WHEN LOWER(s.series) LIKE '%raptor%' THEN v.vehicle_key END) * 100.0 / COUNT(DISTINCT v.vehicle_key) AS raptor_sales_mix FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key JOIN specifications s ON v.specification_key = s.specification_key WHERE LOWER(d.dealer_name) LIKE 'city ford' AND LOWER(s.nameplate) = 'ranger' AND v.retail_processed_date BETWEEN '2024-07-01' AND '2024-09-30';"
  },
  {
    "input" : "provide me the summary of ranger variant/series sold by alpine ford in q3 2024.",
    "query" : "SELECT s.series, COUNT(DISTINCT v.vehicle_key) AS total_sold FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key JOIN specifications s ON v.specification_key = s.specification_key WHERE LOWER(d.dealer_name) LIKE 'alpine ford' AND LOWER(s.nameplate) = 'ranger' AND strftime('%Y', v.retail_processed_date) = '2024' AND strftime('%m', v.retail_processed_date) BETWEEN '07' AND '09' GROUP BY s.series ORDER BY total_sold DESC;"
  },
  {
    "input": "Give me a summary of city Ford by car line MTD achievement.",
    "query": "SELECT s.nameplate AS car_line, COUNT(DISTINCT v.vehicle_key) AS mtd_sales FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key JOIN specifications s ON v.specification_key = s.specification_key WHERE LOWER(d.dealer_name) LIKE 'city Ford' AND v.retail_processed_date BETWEEN '2025-03-01' AND '2025-03-11' GROUP BY s.nameplate ORDER BY mtd_sales DESC;"
  },
  {
    "input": "How many ageing units >90 days have been sold by alpine ford in jan 2024?",
    "query": "SELECT COUNT(DISTINCT v.vehicle_key) AS ageing_units_sold FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key WHERE LOWER(d.dealer_name) LIKE 'alpine ford' AND v.total_days_in_dealer_stock > 90 AND strftime('%Y-%m', v.retail_processed_date) = '2024-01';"
  },
  {
    "input": "What is sales vs objectives of city ford in june 2024?",
    "query": "SELECT st.dealer_name, st.objective_group, st.total_sales, ot.total_objective FROM (SELECT d.dealer_name, s.objective_group, COUNT(DISTINCT v.vehicle_key) AS total_sales FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key JOIN specifications s ON v.specification_key = s.specification_key WHERE LOWER(d.dealer_name) LIKE 'city ford' AND v.retail_processed_date BETWEEN '2024-06-01' AND '2024-06-30' AND s.objective_group IN ('Car', 'SUV', 'Truck') GROUP BY d.dealer_name, s.objective_group) st LEFT JOIN (SELECT d.dealer_name, d.objective_group, SUM(d.objective) AS total_objective FROM dealers d WHERE LOWER(d.dealer_name) LIKE 'city ford' AND d.objective_month = '2024-06-01' AND d.objective_group IN ('Car', 'SUV', 'Truck') GROUP BY d.dealer_name, d.objective_group) ot ON st.dealer_name = ot.dealer_name AND st.objective_group = ot.objective_group;"
  },
  {
    "input": "How is my alpine Ford performing versus L3M run rate (Last 3 months)?",
    "query": "SELECT d.dealer_name, COUNT(DISTINCT v.vehicle_key) AS mtd_sales, (SELECT COUNT(DISTINCT v2.vehicle_key) / 3.0 FROM vehicles v2 JOIN dealers d2 ON v2.dealer_key = d2.dealer_key WHERE LOWER(d2.dealer_name) LIKE 'alpine ford' AND v2.retail_processed_date BETWEEN '2024-10-01' AND '2024-12-31') AS l3m_avg FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key WHERE LOWER(d.dealer_name) LIKE 'alpine ford' AND v.retail_processed_date BETWEEN '2025-01-01' AND '2025-01-31' GROUP BY d.dealer_name;"
  },
  {
    "input": "What is sales vs objectives of alpine ford in q1 2024?",
    "query": "SELECT st.dealer_name, st.objective_group, st.total_sales, ot.total_objective FROM (SELECT d.dealer_name, s.objective_group, COUNT(DISTINCT v.vehicle_key) AS total_sales FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key JOIN specifications s ON v.specification_key = s.specification_key WHERE LOWER(d.dealer_name) LIKE 'alpine ford' AND v.retail_processed_date BETWEEN '2024-01-01' AND '2024-03-31' AND s.objective_group IN ('Car', 'SUV', 'Truck') GROUP BY d.dealer_name, s.objective_group) st LEFT JOIN (SELECT d.dealer_name, d.objective_group, SUM(d.objective) AS total_objective FROM dealers d WHERE LOWER(d.dealer_name) LIKE 'alpine ford' AND d.objective_month BETWEEN '2024-01-01' AND '2024-03-31' AND d.objective_group IN ('Car', 'SUV', 'Truck') GROUP BY d.dealer_name, d.objective_group) ot ON st.dealer_name = ot.dealer_name AND st.objective_group = ot.objective_group;"
  },
  {
    "input": "Give me a summary of city Ford by car line YTD achievement.",
    "query": "SELECT s.nameplate AS car_line, COUNT(DISTINCT v.vehicle_key) AS ytd_sales FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key JOIN specifications s ON v.specification_key = s.specification_key WHERE LOWER(d.dealer_name) LIKE 'city Ford' AND v.retail_processed_date BETWEEN DATE('now', 'start of year') AND DATE('now', 'localtime') AND d.objective_month = DATE('now', 'start of year') GROUP BY s.nameplate order by ytd_sales desc;"
  },
  {
    "input": "Give me a summary of city Ford by car line YTD achievement versus Prior year",
    "query": "SELECT s.nameplate AS car_line, strftime('%Y', v.retail_processed_date) AS year, COUNT(DISTINCT v.vehicle_key) AS ytd_sales FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key JOIN specifications s ON v.specification_key = s.specification_key WHERE LOWER(d.dealer_name) LIKE 'city ford' AND strftime('%m-%d', v.retail_processed_date) <= '01-31' AND strftime('%Y', v.retail_processed_date) IN ('2024', '2025') GROUP BY s.nameplate, year ORDER BY s.nameplate, year DESC;"
  },
  {
    "input": "compare essendon Ford's performance with city ford on October 2024",
    "query": "SELECT st.dealer_name, st.objective_group, st.total_sales, ot.total_objective FROM (SELECT d.dealer_name, s.objective_group, COUNT(DISTINCT v.vehicle_key) AS total_sales FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key JOIN specifications s ON v.specification_key = s.specification_key WHERE LOWER(d.dealer_name) IN ('essendon ford', 'city ford') AND v.retail_processed_date BETWEEN '2024-10-01' AND '2024-10-31' AND s.objective_group IN ('Car', 'SUV', 'Truck') GROUP BY d.dealer_name, s.objective_group) st LEFT JOIN (SELECT d.dealer_name, d.objective_group, SUM(d.objective) AS total_objective FROM dealers d WHERE LOWER(d.dealer_name) IN ('essendon ford', 'city ford') AND d.objective_month = '2024-10-01' AND d.objective_group IN ('Car', 'SUV', 'Truck') GROUP BY d.dealer_name, d.objective_group) ot ON st.dealer_name = ot.dealer_name AND st.objective_group = ot.objective_group;"
  },
  {
    "input": "How many cars did city Ford sell in 2024?",
    "query": "SELECT COUNT(DISTINCT v.vehicle_key) AS total_sales FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key JOIN specifications s ON v.specification_key = s.specification_key WHERE LOWER(d.dealer_name) LIKE 'city ford' AND strftime('%Y', v.retail_processed_date) = '2024' AND LOWER(s.objective_group) = 'car';"
  },
  {
    "input": "Give me a summary of sales achievement by quarter of city ford in 2024?",
    "query": "SELECT dq.quarter, COALESCE(qs.total_sales, 0) AS total_sales, dq.total_objective, ROUND(COALESCE(qs.total_sales, 0) * 100.0 / NULLIF(dq.total_objective, 0), 2) AS achievement_percentage FROM (SELECT d.dealer_key, 'Q' || ((CAST(strftime('%m', d.objective_month) AS INTEGER) - 1) / 3 + 1) AS quarter, SUM(d.objective) AS total_objective FROM dealers d WHERE LOWER(d.dealer_name) LIKE 'city ford' AND strftime('%Y', d.objective_month) = '2024' GROUP BY d.dealer_key, quarter) dq LEFT JOIN (SELECT d.dealer_key, 'Q' || ((CAST(strftime('%m', v.retail_processed_date) AS INTEGER) - 1) / 3 + 1) AS quarter, COUNT(DISTINCT v.vehicle_key) AS total_sales FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key WHERE LOWER(d.dealer_name) LIKE 'city ford' AND strftime('%Y', v.retail_processed_date) = '2024' GROUP BY d.dealer_key, quarter) qs ON dq.dealer_key = qs.dealer_key AND dq.quarter = qs.quarter ORDER BY dq.quarter;"
  },
  {
    "input": "What is the share for city Ford on Ranger in Dec-24?",
    "query": "SELECT COUNT(DISTINCT CASE WHEN LOWER(d.dealer_name) LIKE 'city ford' THEN v.vehicle_key END) * 100.0 / COUNT(DISTINCT v.vehicle_key) AS ranger_share_percentage FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key JOIN specifications s ON v.specification_key = s.specification_key WHERE LOWER(s.nameplate) = 'ranger' AND v.retail_processed_date BETWEEN '2024-12-01' AND '2024-12-31';"
  },
  {
    "input": "How many trucks should city ford sell for q1 2024 to hit 110% of its objective?",
    "query": "SELECT ROUND(SUM(d.objective) * 1.10, 0) AS required_sales_110_percent FROM dealers d WHERE LOWER(d.dealer_name) LIKE 'city ford'AND d.objective_month BETWEEN '2024-01-01' AND '2024-03-31' AND LOWER(d.objective_group) LIKE '%truck%';"
  },
  {
    "input": "how many vehicles does city ford needs to sell to achieve its remaining objectives in feb 2025?",
    "query": "SELECT (COALESCE(obj.total_objective, 0) - COALESCE(sales.retail_sold, 0)) AS vehicles_needed FROM (SELECT SUM(DISTINCT d.objective) AS total_objective FROM dealers d WHERE LOWER(d.dealer_name) = 'city ford' AND d.objective_month = '2025-02-01' AND d.objective_group IN ('Car', 'SUV', 'Truck')) obj, (SELECT COUNT(DISTINCT v.vehicle_key) AS retail_sold FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key WHERE LOWER(d.dealer_name) = 'city ford' AND LOWER(v.vehicle_status) = 'retail sold' AND strftime('%Y-%m', v.retail_processed_date) = '2025-02') sales;"
  },
  {
    "input": "How is city ford performing against its run rate. Will it be able to achieve its objectives in feb 2025?",
    "query": "SELECT ROUND(100.0 * (SELECT COUNT(DISTINCT v.vehicle_key) FROM vehicles v JOIN dealers d2 ON v.dealer_key = d2.dealer_key WHERE LOWER(d2.dealer_name) = 'city ford' AND v.retail_processed_date BETWEEN '2025-02-01' AND '2025-02-04') * 1.0 / (SELECT SUM(d3.objective) FROM dealers d3 WHERE LOWER(d3.dealer_name) = 'city ford' AND d3.objective_month = '2025-02-01' AND d3.objective_group IN ('Car', 'SUV', 'Truck')),2) AS feb_mtd_percentage, ROUND(100.0 * (SELECT AVG(mtd_ratio) FROM (SELECT COUNT(DISTINCT CASE WHEN strftime('%d', v.retail_processed_date) <= '04' THEN v.vehicle_key END) * 1.0 / COUNT(DISTINCT v.vehicle_key) AS mtd_ratio FROM vehicles v JOIN dealers d4 ON v.dealer_key = d4.dealer_key WHERE LOWER(d4.dealer_name) = 'city ford' AND v.retail_processed_date BETWEEN '2024-08-01' AND '2025-01-31' GROUP BY strftime('%Y-%m', v.retail_processed_date))),2) AS avg_6mo_mtd_percentage, CASE WHEN (SELECT COUNT(DISTINCT v.vehicle_key) * 1.0 / (SELECT SUM(d5.objective) FROM dealers d5 WHERE LOWER(d5.dealer_name) = 'city ford' AND d5.objective_month = '2025-02-01' AND d5.objective_group IN ('Car', 'SUV', 'Truck')) FROM vehicles v JOIN dealers d6 ON v.dealer_key = d6.dealer_key WHERE LOWER(d6.dealer_name) = 'city ford' AND v.retail_processed_date BETWEEN '2025-02-01' AND '2025-02-04') > (SELECT AVG(mtd_ratio) FROM (SELECT COUNT(DISTINCT CASE WHEN strftime('%d', v.retail_processed_date) <= '04' THEN v.vehicle_key END) * 1.0 / COUNT(DISTINCT v.vehicle_key) AS mtd_ratio FROM vehicles v JOIN dealers d7 ON v.dealer_key = d7.dealer_key WHERE LOWER(d7.dealer_name) = 'city ford' AND v.retail_processed_date BETWEEN '2024-08-01' AND '2025-01-31' GROUP BY strftime('%Y-%m', v.retail_processed_date))) THEN 'Yes, City Ford is on track to achieve its objectives for February 2025.' ELSE 'No, City Ford is currently behind its 6-month run rate for February 2025.' END AS will_achieve_objective;"
  },
  {
    "input": "How many stocks are available at city ford?",
    "query": "SELECT COUNT(DISTINCT v.vehicle_key) FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key WHERE LOWER(d.dealer_name) = 'city ford' AND LOWER(v.vehicle_status) != 'retail sold' AND v.dealer_invoice_date IS NOT NULL AND v.dealer_invoice_date != '';"
  },
  {
    "input": "How many more days can I manage with the available stocks at city ford?",
    "query": "SELECT ROUND((COUNT(DISTINCT v.vehicle_key) * 25.0) /  NULLIF((SELECT COUNT(DISTINCT v2.vehicle_key) / 3.0 FROM vehicles v2 JOIN dealers d2 ON v2.dealer_key = d2.dealer_key WHERE LOWER(d2.dealer_name) = 'city ford' AND LOWER(v2.vehicle_status) = 'retail sold' AND v2.retail_processed_date BETWEEN '2024-12-01' AND '2025-02-28'), 0),2) AS fdc_days FROM vehicles v JOIN dealers d ON v.dealer_key = d.dealer_key WHERE LOWER(d.dealer_name) = 'city ford' AND LOWER(v.vehicle_status) != 'retail sold' AND v.dealer_invoice_date IS NOT NULL AND v.dealer_invoice_date != '';"
  },
  {
    "input": "Give me Q1 2024 sales of Alpine Ford by nameplate",
    "query": "SELECT s.nameplate, COUNT(DISTINCT v.vehicle_key) AS q1_sales FROM vehicles v JOIN specifications s ON v.specification_key = s.specification_key JOIN dealers d ON v.dealer_key = d.dealer_key WHERE LOWER(d.dealer_name) LIKE 'alpine ford' AND v.retail_processed_date BETWEEN '2024-01-01' AND '2024-03-31' GROUP BY s.nameplate;"
  }
]

# Initialize embeddings
try:
    embeddings_1 = OpenAIEmbeddings(
        openai_api_key=token_fetcher.token,
        openai_api_base="https://api.pivpn.core.ford.com/fordllmapi/api/v1",
        model="text-embedding-ada-002"
    )
except Exception as e:
    st.error(f"⚠️ Failed to initialize embeddings: {str(e)}")
    st.stop()

# Initialize vector store
try:
    # Use a relative path for cloud deployment
    chroma_db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    os.makedirs(chroma_db_path, exist_ok=True)
    
    vectorstore_client = PersistentClient(path=chroma_db_path)
    collection = vectorstore_client.get_or_create_collection(name="langchain")
    vectorstore = Chroma(client=vectorstore_client, collection_name="langchain", embedding_function=embeddings_1)
except Exception as e:
    st.error(f"⚠️ Failed to initialize vector store: {str(e)}")
    st.stop()

# Example Selector
try:
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embeddings_1,
        vectorstore,
        k=2,
        input_keys=["input"]
    )
except Exception as e:
    st.error(f"⚠️ Failed to initialize example selector: {str(e)}")
    st.stop()

# Example Prompt
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}\nSQLQuery:"),
    ("ai", "{query}"),
])

# Few-shot Prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    input_variables=["input", "top_k"]
)

# Final Prompt
final_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a SQLite expert. Given an input question, generate a syntactically correct SQL query compatible with SQLite's SQL dialect.

            ### Instructions:
            - ONLY return the raw SQL query as plain text.
            - DO NOT include markdown formatting, triple backticks, or any prefix like "SQLQuery:".
            - DO NOT provide explanations or commentary.
            - DO NOT use SELECT * — only include necessary columns.
            - Use correct column names and table names from the provided schema.
            - Qualify column names with table names when required.
            - Avoid using unsupported SQLite functions.

            ### Query Guidelines:
            - When querying objective-related data, include: dealer_name, objective_month, objective, and objective_group.
            - Avoid unnecessary aggregation (e.g., don't wrap `objective` in SUM() if it's already per row).
            - When querying vehicle sales, retail sold, or related counts, always use: COUNT(DISTINCT vehicle_key).
            - Filter dealer names using `LOWER(dealer_name) LIKE '%<input>%'`.
            - Use valid SQLite expressions for date filtering:
              - For a specific month: `AND objective_month = '2024-12-01'`
              - For a specific year: `AND strftime('%Y', objective_month) = '2024'`
            - Filter dealer names using `LOWER(dealer_name) LIKE '%<input>%'`.
              - For specific dealer names, use exact matching: `LOWER(dealer_name) LIKE 'city ford'`.
              - when the user clearly asks for a pattern, use matching such as: `LOWER(dealer_name) LIKE '%city%ford%'` (e.g., "all dealers containing city").
            - When summing objectives, always filter by relevant objective groups (e.g., Car, SUV, Truck) using:
              - `AND objective_group IN ('Car', 'SUV', 'Truck')`
            - When a user prompts in by asking how many days that he/she can manage with the available stocks, assume it as FDC (Forward Days Cover).
            - When calculating FDC (Forward Days Cover), use the most recent month for which retail sales data is available. As of now, this is February 2025.
            - When a user prompts in by asking for a Target, assume it as Objectives. Both Target and Objectives mean the same.
            - When a user prompts in by asking questions that requires to filter retail_processed_date, assume the current date is 2025-03-11.

            ### Important:
            **Strictly follow the SQL format and structure used in the few-shot examples below. Generate efficient, clean queries by mimicking these examples.**

            Use only the following table information:
            {table_info}
            
            Below are a few example questions and their corresponding SQL queries:
            {examples}"""
    ),
    few_shot_prompt,
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}"),
])

@st.cache_data
def get_table_details():
    """Load table descriptions from CSV file"""
    try:
        # Use relative path for cloud deployment
        csv_path = os.path.join(os.path.dirname(__file__), "Multiple_dealers_Tables", "Table_descriptions.csv")
        
        # Check if file exists
        if not os.path.exists(csv_path):
            st.error(f"⚠️ Table descriptions file not found at: {csv_path}")
            return "No table information available"
        
        # Read the CSV file
        table_description = pd.read_csv(csv_path)
        
        # Create table details string
        table_details = ""
        for index, row in table_description.iterrows():
            table_details += f"Table Name: {row['table_name']}\n"
            table_details += f"Table Description: {row['column_description']}\n\n"
        
        return table_details
    except Exception as e:
        st.error(f"⚠️ Error loading table details: {str(e)}")
        return "Error loading table information"

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")

class TableList(BaseModel):
    tables: List[Table]

def get_tables(tables: List[Table]) -> List[str]:
    return [table.name for table in tables]

# Get table details
table_details = get_table_details()

table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_details}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

# LangChain prompt for table extraction
table_prompt = ChatPromptTemplate.from_messages([
    ("system", table_details_prompt),
    ("human", "{input}")
])

def is_ford_related(question: str) -> bool:
    # Rename to avoid conflict with chat input 'prompt'
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You're a classification assistant. Respond with 'Yes' or 'No' only. Is the following question related to Ford dealership data like sales, objectives, VINs, or nameplates?"),
        ("human", "{question}")
    ])
    # Add Ford API configuration
    classifier_llm = ChatOpenAI(
        model="gpt-4.1",
        openai_api_base="https://api.pivpn.core.ford.com/fordllmapi/api/v1",
        openai_api_key=token_fetcher.token,
        temperature=0
    )
    result = classifier_llm.invoke(prompt_template.format_messages(question=question)).content.strip().lower()
    return result == "yes"


# Apply `with_structured_output` correctly
try:
    structured_llm = llm.with_structured_output(TableList)
    table_chain = (itemgetter("question") | table_prompt | structured_llm | (lambda output: [table.name for table in output.tables]))
except Exception as e:
    st.error(f"⚠️ Failed to initialize structured LLM: {str(e)}")
    st.stop()

@st.cache_resource
def get_chain():
    """Initialize the main processing chain"""
    try:
        print("Creating chain")
        
        # Initialize SQL Database with relative path
        db_path = os.path.join(os.path.dirname(__file__), "zone_manager_multitable.db")
        
        if not os.path.exists(db_path):
            st.error(f"⚠️ Database file not found at: {db_path}")
            st.stop()
        
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        # Create SQL query chain
        generate_query = create_sql_query_chain(llm, db, final_prompt)
        
        # Execute SQL query
        execute_query = QuerySQLDatabaseTool(db=db)
        
        # Rephrase answer using LLM
        rephrase_answer = answer_prompt | llm | StrOutputParser()
        
        # Define the complete processing chain
        chain = (
            RunnablePassthrough.assign(table_names_to_use=table_chain) |
            RunnablePassthrough.assign(query=generate_query).assign(
                result=itemgetter("query") | execute_query
            ) |
            rephrase_answer
        )
        
        return chain
    except Exception as e:
        st.error(f"⚠️ Failed to create chain: {str(e)}")
        st.stop()

def create_history(messages):
    """Create chat history from messages"""
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_chain(question, messages, examples):
    """Invoke the processing chain with error handling"""
    try:
        chain = get_chain()
        history = create_history(messages)
        
        # Invoke the chain
        response = chain.invoke({
            "question": question,
            "examples": examples,
            "messages": history.messages,
            "table_info": get_table_details()
        })
        
        # Add conversation to memory
        history.add_user_message(question)
        history.add_ai_message(response)
        
        return response
    except Exception as e:
        st.error(f"⚠️ Error processing question: {str(e)}")
        return "Sorry, I encountered an error while processing your question. Please try again."

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            if is_ford_related(prompt):
                response = invoke_chain(prompt, st.session_state.messages, examples)
            else:
                general_llm = ChatOpenAI(
                    model="gpt-4.1", 
                    openai_api_base="https://api.pivpn.core.ford.com/fordllmapi/api/v1",
                    openai_api_key=token_fetcher.token,
                    temperature=0
                )
                response = general_llm.invoke(prompt).content
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


# Sidebar with app information
with st.sidebar:
    st.header("App Information")
    st.info("Zone-Manager Chatbot helps you query vehicle and dealer data using natural language.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.header("Environment Status")
    st.success("✅ Ford LLM Connected")
    if langfuse_handler:
        st.success("✅ Langfuse Connected")
    else:
        st.warning("⚠️ Langfuse Not Connected")

# Handle Streamlit configuration for cloud deployment
if __name__ == "__main__":
    # This will be handled by the Cloud Run startup command
    pass
