import streamlit as st
from openai import OpenAI
import psycopg2
import re
import os
import pandas as pd

# Set up OpenAI API key
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Database connection parameters
db_params = st.secrets["db_credentials"]


def connect_to_db():
    """Establish a connection to the Redshift database."""
    try:
        conn = psycopg2.connect(**db_params)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        return None

def execute_query(conn, query):
    """Execute a SQL query and return the results and cursor."""
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
            return results, cur
    except psycopg2.Error as e:
        print(f"Error executing query: {e}")
        return None, None

def get_gpt4_response(prompt):
    """Get a response from GPT-4 using the OpenAI API."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Changed to "gpt-4" as "gpt-4-0125-preview" is not available
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates SQL queries for a Redshift database. The database contains information about mentees, courses, lessons, companies, and more. Always return only the SQL query, without any explanations, comments, or formatting. Always remember to use redshift syntax. use distinct keyword in case statements."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None

def clean_sql_query(sql_query):
    """Remove any Markdown formatting or unnecessary characters from the SQL query."""
    sql_query = re.sub(r'```\w*\n?', '', sql_query)
    sql_query = sql_query.strip()
    return sql_query

def generate_sql_query(user_input):
    """Generate a SQL query based on the user's input."""
    reference_logic = """
    Logic from reference query:
    1. Use a CTE (Common Table Expression) named 'cte' for complex calculations.
    2. use temp.marketing_mis table to get the data of leads those who did any activity on the website of scaler 
    3. use scaler_ebdb_users table to get the name and email of the leads who are in the temp.marketing_mis table
    4. case when eligible_flag=0 then 'Not Eligible' when eligible_flag=1 then 'Eligible' end as eligible_flag,
    case when assigned_flag=0 then 'Not Assigned' when assigned_flag=1 then 'Assigned' end as assigned_flag,
    case when consumed_flag=0 then 'Not Consumed' when consumed_flag=1 then 'Consumed' end as consumed_flag,
    case when test_launched_flag=0 then 'Not Launched' when test_launched_flag=1 then 'Launched' end as test_launched_flag,
    case when test_passed_flag=0 then 'Test Not Passed' when test_passed_flag=1 then 'Test Passed' end as test_passed_flag,
    CASE 
    WHEN ls_fresh_flag = 0 THEN 'Ineffective Re-engaged'
    WHEN ls_fresh_flag = 1 THEN 'Re-engaged'
    WHEN ls_fresh_flag = 2 THEN 'Fresh'
    WHEN ls_fresh_flag = 3 THEN 'Not in LSQ'
    ELSE 'Unknown'
END AS fresh_flag
,
    case when first_payment_done=0 then 'Payment Not Done' when first_payment_done=1 then 'Payment Done' end as first_payment_done
    Use these enumns for temp.marketing_mis table 
    5. Use event_type from temp.marketing_mis table to get the activity performed type of the leads
    6. use event_rank_registraion column to get the rank of activity performed by the leads  where event_rank_registraion = 1 then it means it is the first activity performed by the lead
    7. to calculate l2p formula is (payments_done/leads_consumed)*100
    use count(distinct case when first_payment_done=1 then email end) as payments
    use count(distinct case when consumed_flag=1 then email end) as consumed 
    8. always use the distinct keyword in the query 
    9. whenever batch is asked use batch column in temp.marketing_mis table to get the batch of the leads
    10. always remember to use distinct keyword in case statements 
    11. always use email column in the query to do any calculation
    12. AVOID USING CASE STATEMENTS IN CTE INSTEAD USE IT IN MAIN QUERY  
    13. REMEMBER THERE CAN BE MULTIPLE ROWS FOR SAME EMAIL AS A USER CAN PERFORM MULTIPLE ACTIVITIES ON THE WEBSITE SO YOU HAVE TO USE DISTINCT IN CTE TOO
    14. count(distinct case when eligible_flag=1 then email end) as eligible
    15. count(distinct email) as gross
    16. used attended column to get whether the lead has attended or not attended the event
    17. The values in event_type are 79_ScalerTopics, others, 29_Career, 11_CallBack, 3_MC, 2_Alum_Session, 99_Non_Mkt 6_Bootcamp 1_Free_Product 4_FRLS 14_Referral
    where 3_MC means masterclass 79_ScalerTopics means Scaler Topics 29_Career means Career Roadmap Tool 11_CallBack means requested callback 2_Alum_Session means Alumni Session 99_Non_Mkt means non marketing 
    6_Bootcamp means Bootcamp 1_Free_Product means Free Live Class 4_FRLS means Free Recorded live Session 14_Referral means Referral. 
    Remember these all event_type are interaction points on website also known as IP 
    18. whenever the event_type is masterclass use event_id of that event_type and connect with id of scaler_ebdb_events table to get the event name using title column in scaler_ebdb_events table and also there is start_time and end_time in events table to get date of masterclass
    for all other event_type you can use event_name in temp.marketing_mis
    19. Use landing_page_url column in temp.marketing_mis to get the source url from where user came 
    20. Use program_type to know in which program user is interested or landed from
    """

    
    prompt = f"""Given the following reference logic 

Reference Logic:
{reference_logic}

Generate a SQL query for the following request: {user_input}

Return only the SQL query, without any explanations, comments, or formatting. Use the DISTINCT keyword where appropriate. Follow the structure and logic of the reference query, adapting it to the specific request."""

    generated_sql = get_gpt4_response(prompt)
    if generated_sql is None:
        return None  # Return None if there was an error generating the SQL query
    return clean_sql_query(generated_sql)

def main():
    # Set page configuration
    st.set_page_config(page_title="Scaler Academy Database Chatbot", page_icon="🤖", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextInput>div>div>input {
        background-color: white;
    }
    .footer {
        text-align: center;
        padding: 10px 0;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

    # Application header
    st.title("Scaler Academy Database Chatbot")
    
    st.markdown("---")
    st.write("Ask questions about growth queries")

    # Initialize connection
    if 'conn' not in st.session_state:
        st.session_state.conn = connect_to_db()

    # Check if connection is successful
    if not st.session_state.conn:
        st.error("Failed to connect to the database. Please check your connection settings.")
        return

    # User input
    user_input = st.text_input("Enter your question:", placeholder="e.g., can you give me payments count in aug-2024")

    if st.button("Submit", key="submit"):
        if user_input:
            with st.spinner("Generating query and fetching results..."):
                # Print user input
                st.subheader("Your question:")
                st.info(user_input)

                generated_sql = generate_sql_query(user_input)

                if generated_sql:
                    st.subheader("Generated SQL query:")
                    st.code(generated_sql, language="sql")

                    results, cur = execute_query(st.session_state.conn, generated_sql)

                    if results and cur:
                        st.subheader("Query results:")
                        df = pd.DataFrame(results)
                        
                        # Get column names from the cursor description
                        column_names = [desc[0] for desc in cur.description]
                        
                        # Assign column names to the dataframe
                        df.columns = column_names
                        
                        # Display results in an expandable section without highlighting
                        with st.expander("View Results Table", expanded=True):
                            st.dataframe(df, use_container_width=True)

                        # Add download button for CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download results as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv",
                        )
                    else:
                        st.warning("No results found or there was an error executing the query.")
                else:
                    st.error("I'm sorry, I couldn't generate a proper query for your request. There might be an issue with the OpenAI API.")
        else:
            st.warning("Please enter a question.")

    # Add a centered footer
    st.markdown("---")
    st.markdown('<p class="footer">Built with ❤️ by the Scaler Product Analytics Team</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
