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
        st.error(f"Error connecting to the database: {e}")
        return None

def execute_query(conn, query):
    """Execute a SQL query and return the results and cursor."""
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
            return results, cur
    except psycopg2.Error as e:
        st.error(f"Error executing query: {e}")
        return None, None
    finally:
        # Always reset the connection after executing a query
        conn.rollback()

def reset_connection():
    """Reset the database connection."""
    if 'conn' in st.session_state:
        try:
            st.session_state.conn.close()
        except:
            pass
    st.session_state.conn = connect_to_db()

def get_gpt4_response(prompt, conversation_history):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates SQL queries for a Redshift database. The database contains information about mentees, courses, lessons, companies, and more. Always return only the SQL query, without any explanations, comments, or formatting. Always remember to use redshift syntax."},
            {"role": "system", "content": "Remember to use the reference logic provided in previous messages when generating queries."},
        ] + conversation_history + [
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the latest GPT-4 model for better context retention
            messages=messages,
            temperature=0.7,  # Slightly reduce temperature for more consistent outputs
            max_tokens=2000,  # Increase max tokens to allow for longer responses
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return None

def clean_sql_query(sql_query):
    """Remove any Markdown formatting or unnecessary characters from the SQL query."""
    sql_query = re.sub(r'```\w*\n?', '', sql_query)
    sql_query = sql_query.strip()
    return sql_query

def generate_sql_query(user_input, conversation_history):
    """Generate a SQL query based on the user's input."""
    reference_logic = """
    Logic from reference query:
    1. Use a CTE (Common Table Expression) named 'cte' for complex calculations.
    2. Use temp.marketing_mis table for marketing leads and temp.non_marketing_mis for non-marketing leads.
    3. Use scaler_ebdb_users table to get the name and email of the leads who are in the temp.marketing_mis table
    4. Use case statements for flag conversions (eligible_flag, assigned_flag, consumed_flag, etc.)
    5. Use event_type from temp.marketing_mis table to get the activity performed type of the leads
    6. Use event_rank_registraion column to get the rank of activity performed by the leads
    7. Calculate l2p formula as (payments_done/leads_consumed)*100
    8. Always use the distinct keyword in the query 
    9. Use batch column in temp.marketing_mis table to get the batch of the leads
    10. Always use email column in the query to do any calculation
    11. Avoid using case statements in CTE, instead use it in main query
    12. Remember there can be multiple rows for same email as a user can perform multiple activities on the website
    13. Use attended column to get whether the lead has attended or not attended the event
    14. Use landing_page_url column in temp.marketing_mis to get the source url from where user came 
    15. Use program_type to know in which program user is interested or landed from
    16. Whenever any condition is applied don't apply that in left join always apply condition in where clause
    17. Calculate final_source within the CTE itself
    18. Use the provided CASE statement for Channels column
    19. ALWAYS USE BATCH FROM CTE2 WHENEVER SOMEONE ASKS INFORMATION RELATED TO A PARTICULAR BATCH.
    20. For temp.marketing_mis use case when first_payment_done=0 then 'Payment Not Done' when first_payment_done=1 then 'Payment Done' end as first_payment_done
    21. case when eligible_flag=0 then 'Not Eligible' when eligible_flag=1 then 'Eligible' end as eligible_flag,
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
    22. For lead level data use Lead level detail query that consists of multiple status like consumed status, lead called status, paid status, payment link sent status, test rolled out status etc.
    """

    reference_queries = """
    -- Query 1: Main marketing query
    WITH cte AS (
        SELECT DISTINCT
            email,
            batch,
            event_type,
            event_rank_registraion,
            eligible_flag,
            assigned_flag,
            consumed_flag,
            test_launched_flag,
            test_passed_flag,
            ls_fresh_flag,
            first_payment_done,
            utm_source,
            utm_medium,
            utm_campaign,
            utm_content,
            event_name,
            landing_page_url,
            referring_url,
            program_type,
            CASE
                WHEN lower(utm_medium) IN ('google','googlesmartdisplay') THEN 'googledisplay'
                WHEN lower(utm_campaign) LIKE '%_ads_googlesearch_brand%' THEN 'brandsearch'
                -- ... (rest of the CASE statement for final_source)
                WHEN lower(utm_source) = 'ads' AND lower(utm_medium) ='google.com' THEN 'googleyoutube'
                ELSE 'other'
            END AS final_source
        FROM temp.marketing_mis
        WHERE event_rank_registraion = 1
    )
    SELECT
        batch,
        COUNT(DISTINCT email) AS gross,
        COUNT(DISTINCT CASE WHEN eligible_flag = 1 THEN email END) AS eligible,
        COUNT(DISTINCT CASE WHEN assigned_flag = 1 THEN email END) AS assigned,
        COUNT(DISTINCT CASE WHEN consumed_flag = 1 THEN email END) AS consumed,
        COUNT(DISTINCT CASE WHEN first_payment_done = 1 THEN email END) AS payments,
        CASE
            WHEN COUNT(DISTINCT CASE WHEN consumed_flag = 1 THEN email END) > 0
            THEN (COUNT(DISTINCT CASE WHEN first_payment_done = 1 THEN email END)::FLOAT / 
                  COUNT(DISTINCT CASE WHEN consumed_flag = 1 THEN email END)) * 100
            ELSE 0
        END AS l2p,
        CASE
            WHEN final_source IN ('organic', 'influencer', 'brandedcontent', 'brandcampaign', 'dv360', 'publisher-ideal') THEN '1.Organic'
            WHEN final_source IN ('facebook', 'googleyoutube', 'googlesearch', 'googlediscovery', 'linkedin', 'googlepmc', 'brandsearch', 'bing', 'googledisplay', 'googleuac', 'columbia', 'quora', 'taboola', 'reddit', 'yahoo', 'affiliate', 'rtbhouse', 'googleuac') THEN '2.Paid'
            WHEN final_source IN ('midfunnel', 'interviewbit', 'community', 'organic_social', 'seo', 'other', 'referral', 'ib-midfunnel', 'topics', 'topics-midfunnel') THEN '3.Non-Paid'
            ELSE '4.Other'
        END AS Channels
    FROM cte
    GROUP BY batch, Channels
    ORDER BY batch, Channels;

    -- Query 2: Lead level detail query
    WITH cte AS (
        SELECT sw."sales batch" AS sales_batch,
               sw."Marketing Batch" AS marketing_batch,
               DATE(a.createdon + interval '330 minutes') activitydate,
               mx_custom_1,
               lower(prospectemailaddress) AS lead_email,
               CASE 
                   WHEN lower(mx_custom_1) LIKE '%dev%' THEN 'devops'
                   WHEN lower(mx_custom_1) LIKE '%data%' OR lower(mx_custom_1) LIKE '%dsml%' THEN 'ds'
                   WHEN lower(mx_custom_1) LIKE '%acad%' THEN 'acad'
                   ELSE 'none'
               END AS course,
               CASE 
                   WHEN a.activityevent IN (487) THEN lower(mx_custom_1)
                   WHEN a.activityevent IN (389) THEN lower(REGEXP_SUBSTR(mx_custom_7, '[A-Za-z0-9._%%+-]+@scaler\\.com'))
                   ELSE coalesce(lower(u.emailaddress))
               END AS bda_email,
               a.activityevent,
               a.prospectid prospectid,
               mx_custom_3,
               mx_custom_11,
               STATUS,
               rank() OVER (PARTITION BY a.prospectid, a.activityevent ORDER BY a.createdon) rnk,
               RANK() OVER (PARTITION BY a.prospectid, a.activityevent, sw."Sales Batch" ORDER BY DATE(a.createdon + interval '330 minutes') ASC) AS rnk2
        FROM interviewbit_mxradon_activities a
        LEFT JOIN interviewbit_mxradon_prospectactivity_extensionbase ab ON ab.prospectactivityextensionid = a.prospectactivityid
        LEFT JOIN (
            SELECT userid,
                   replace(replace(emailaddress, '.54288.obsolete', ''), '.47349.obsolete', '') emailaddress
            FROM interviewbit_mxradon_users
        ) u ON coalesce(ab.OWNER, ab.createdby) = u.userid
        LEFT JOIN scaler_ebdb_sales_week sw ON sw.DATE = DATE(a.createdon + interval '330 minutes')
        WHERE DATE(a.createdon + interval '330 minutes') >= TO_DATE('30-08-2023', 'DD-MM-YYYY')
              AND a.activityevent IN (483, 490, 493, 489, 456, 499, 455, 457, 234, 453, 309, 389, 484, 493, 231, 232, 487, 464, 498, 486, 532, 488, 504, 528, 529)
    ),
    cte2 AS (
        SELECT "Sales Batch" AS batch,
               "Marketing Batch",
               AVG("Sorting") AS sort
        FROM scaler_ebdb_sales_week
        GROUP BY "Sales Batch", "Marketing Batch"
    )
    SELECT cte.*,
           cte2.sort,
           TO_DATE(LEFT(CAST(cte2.sort AS VARCHAR), 4) || '-' || RIGHT(CAST(cte2.sort AS VARCHAR), 2) || '-01', 'YYYY-MM-DD') AS batch_date,
           CASE WHEN activityevent = 487 THEN 'Lead Called' END AS lead_called,
           CASE WHEN activityevent IN (483, 490, 493, 489, 456, 499, 455, 457, 234, 453, 309, 389, 484, 493, 231, 232, 487, 464, 498, 486, 532, 488, 504, 528, 529)
                THEN 'Lead Consumed' END AS consumed_status,
           CASE WHEN activityevent IN (498) AND rnk2 = 1 THEN 'Payment Done' END AS paid_status,
           CASE WHEN activityevent = 499 AND rnk = 1 THEN 'Payment Link Sent' END AS payment_link_status,
           CASE WHEN activityevent = 389 AND STATUS = 'completed' AND mx_custom_11 > 15 THEN 'Session Conducted' END AS session_conducted_status,
           CASE WHEN activityevent = 389 THEN 'Session Scheduled' END AS session_schedule_status,
           CASE WHEN activityevent = 489 THEN 'Test Cleared' END AS test_clear_status,
           CASE WHEN activityevent IN (484, 486, 483) THEN 'Test Rolled Out' END AS test_roll_out_status
    FROM cte
    JOIN cte2 ON cte2.batch = cte.sales_batch;

    -- Query 3: Non-marketing leads query
    SELECT 
        batch,
        email AS lead_email, 
        program_type AS program,
        CASE WHEN effective_flag = 1 THEN 'effective' ELSE 'not effective' END AS effective_flag,
        CASE WHEN assigned_flag = 1 THEN 'assigned' ELSE 'not assigned' END AS assigned_flag,
        CASE WHEN consumed_flag = 1 THEN 'consumed' ELSE 'not consumed' END AS consumed_flag,
        CASE WHEN test_start = 1 THEN 'Test Started' ELSE 'Test Not Started' END AS test_start_flag,
        CASE WHEN test_pass = 1 THEN 'Test Passes' ELSE 'Test Not Passed' END AS test_passed_flag,
        CASE WHEN payment_flag = 1 THEN 'Paid' ELSE 'Not Paid' END AS payment_flag,
        CASE WHEN payment_flag = 1 THEN paying_for_type END AS paid_for_program,
        prospectstage AS current_stage
    FROM temp.non_marketing_mis;
    """

    prompt = f"""Given the following reference logic and reference queries:

Reference Logic:
{reference_logic}

Reference Queries:
{reference_queries}

Conversation History:
{' '.join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]])}

Generate a SQL query for the following request: {user_input}

Return only the SQL query, without any explanations, comments, or formatting. Use the DISTINCT keyword where appropriate. Follow the structure and logic of the reference queries, adapting them to the specific request. Learn from any feedback or corrections in the conversation history.

Important: For non-marketing leads, use temp.non_marketing_mis table. For marketing leads, use temp.marketing_mis table. Choose the appropriate table based on the context of the user's request."""

    generated_sql = get_gpt4_response(prompt, conversation_history)
    return clean_sql_query(generated_sql)

def get_conversation_history():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    return st.session_state.conversation_history

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
    st.write("Ask questions about mentees, courses, lessons, companies, and more.")

    # Initialize connection
    if 'conn' not in st.session_state or st.session_state.conn is None:
        reset_connection()

    # Check if connection is successful
    if not st.session_state.conn:
        st.error("Failed to connect to the database. Please check your connection settings.")
        return

    conversation_history = get_conversation_history()

    # User input
    user_input = st.text_input("Enter your question:", placeholder="e.g., Show me the top 5 mentees by attendance")

    if 'query_results' not in st.session_state:
        st.session_state.query_results = []

    if st.button("Submit", key="submit"):
        if user_input:
            with st.spinner("Generating query and fetching results..."):
                # Print user input
                st.subheader("Your question:")
                st.info(user_input)

                # Add user input to conversation history
                conversation_history.append({"role": "user", "content": user_input})

                generated_sql = generate_sql_query(user_input, conversation_history)

                if generated_sql:
                    st.subheader("Generated SQL query:")
                    st.code(generated_sql, language="sql")

                    # Add generated SQL to conversation history
                    conversation_history.append({"role": "assistant", "content": generated_sql})

                    results, cur = execute_query(st.session_state.conn, generated_sql)

                    if results is None and cur is None:
                        st.error("An error occurred while executing the query. Resetting the connection...")
                        reset_connection()
                        results, cur = execute_query(st.session_state.conn, generated_sql)

                    if results and cur:
                        st.subheader("Query results:")
                        df = pd.DataFrame(results)
                        
                        # Get column names from the cursor description
                        column_names = [desc[0] for desc in cur.description]
                        
                        # Assign column names to the dataframe
                        df.columns = column_names
                        
                        # Add new results to the session state
                        st.session_state.query_results.append({
                            "question": user_input,
                            "query": generated_sql,
                            "dataframe": df
                        })
                    else:
                        st.warning("No results found or there was an error executing the query.")
                else:
                    st.error("I'm sorry, I couldn't generate a proper query for your request.")
        else:
            st.warning("Please enter a question.")

    # Display all results in dropdowns
    if st.session_state.query_results:
        st.subheader("All Query Results")
        for i, result in enumerate(reversed(st.session_state.query_results), 1):
            with st.expander(f"Result {i}: {result['question']}", expanded=(i == 1)):
                st.code(result['query'], language="sql")
                st.dataframe(result['dataframe'], use_container_width=True)
                
                csv = result['dataframe'].to_csv(index=False)
                st.download_button(
                    label="📥 Download results as CSV",
                    data=csv,
                    file_name=f"query_results_{i}.csv",
                    mime="text/csv",
                )

    # Add dropdown for comment or change request
    action_type = st.selectbox("Choose an action:", ["Submit Comment", "Request Changes"])

    if action_type == "Submit Comment":
        user_comment = st.text_area("Enter your comment or suggestion:")
        if st.button("Submit"):
            if user_comment:
                conversation_history.append({"role": "user", "content": f"Comment: {user_comment}"})
                st.success("Thank you for your feedback.")
            else:
                st.warning("Please enter a comment before submitting.")
    else:  # Request Changes
        change_request = st.text_area("Enter your change request:")
        if st.button("Submit Change Request"):
            if change_request:
                conversation_history.append({"role": "user", "content": f"Change Request: {change_request}"})
                with st.spinner("Generating new query and fetching results..."):
                    new_generated_sql = generate_sql_query(change_request, conversation_history)

                    if new_generated_sql:
                        st.subheader("New Generated SQL query:")
                        st.code(new_generated_sql, language="sql")

                        new_results, new_cur = execute_query(st.session_state.conn, new_generated_sql)

                        if new_results and new_cur:
                            st.subheader("New Query Results:")
                            new_df = pd.DataFrame(new_results)
                            new_column_names = [desc[0] for desc in new_cur.description]
                            new_df.columns = new_column_names
                            
                            st.dataframe(new_df, use_container_width=True)

                            # Check if latest_query exists before updating
                            if 'latest_query' in st.session_state and st.session_state.latest_query:
                                # Update the latest query in session state
                                st.session_state.latest_query = {
                                    'id': st.session_state.latest_query['id'],
                                    'question': st.session_state.latest_query['question'],
                                    'sql': new_generated_sql,
                                    'results': new_df
                                }

                                # Update the query in the database
                                update_query(st.session_state.latest_query['id'], new_generated_sql)

                                st.success("Query updated successfully. You can now rate the new results.")
                            else:
                                st.warning("No previous query found. Please submit a new query first.")
                        else:
                            st.warning("No results found or there was an error executing the new query.")
                    else:
                        st.error("I'm sorry, I couldn't generate a proper query for your change request.")
            else:
                st.warning("Please enter a change request before submitting.")

    # Display conversation history
    st.subheader("Conversation History")
    for message in conversation_history:
        role = message["role"]
        content = message["content"]
        st.text(f"{role.capitalize()}: {content}")

    # Limit conversation history to last 10 messages
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]

    # Add a centered footer
    st.markdown("---")
    st.markdown('<p class="footer">Built with ❤️ by the Scaler Product Analytics Team</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
