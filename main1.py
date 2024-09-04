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
        ] + conversation_history + [
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
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
    23. Whenever any condition is applied dont apply that in left join always apply condition in where clause
    24. ,case
            when lower(utm_medium) in ('google','googlesmartdisplay') then 'googledisplay'
            when lower(utm_campaign) like '%_ads_googlesearch_brand%' then 'brandsearch'
            when lower(utm_source) = 'ads' and lower(utm_medium)='whatsapp' then 'facebook'
            when lower(utm_medium) in ('organic_social','osocial','organic_video') then 'organic_social'
            when lower(utm_source) in ('social','osocial','osocial%2f','organic-social-3') then 'organic_social'
            when lower(utm_medium) in ('googlediscovery','googlepmc','googleyoutube','linkedin','googlesearch','facebook','taboola','googledisplay','rtbhouse') then utm_medium
            when lower(utm_medium) like '%reddit%' then 'reddit'
            when lower(utm_source) = 'ads' then utm_medium
            when lower(utm_medium) ='dv360' OR lower(utm_source)='dv360' then 'dv360'
            when lower(utm_source) in ('hotstar.com', 'inshorts.com') then 'brandcampaign'
            when lower(utm_campaign) like '%ads_columbia%' then 'columbia'
            when lower(event_type) = '7_community' then 'community'
            when lower(utm_source) = 'ib' and lower(event_type) = lower('79_ScalerTopics') then 'interviewbit'
            when lower(utm_source) = 'midfunnel' and lower(event_type) = lower('79_ScalerTopics') then 'midfunnel' -- NEW LINE HERE FOR MIDFUNNEL
            when lower(utm_source) = 'topics-midfunnel' then 'topics-midfunnel'
            when lower(utm_source) = 'midfunnel' and (landing_page_url ilike '%topics/%' or referring_url ilike '%topics/%') then 'midfunnel'
            -- when lower(event_type) = lower('79_ScalerTopics') OR lower(utm_source)='topics' OR lower(utm_source) ilike '%%topics%%' then 'topics'
            when lower(utm_source) in ('midfunnel-ib','ib-midfunnel') then 'ib-midfunnel'
            when lower(utm_source) in ('midfunnel','brandedcontent','brandcampaign') then lower(utm_source)
            when lower(event_type) = lower('79_ScalerTopics') OR lower(utm_source) ilike '%topics%' OR lower(utm_medium) ilike '%topics%' OR (landing_page_url ilike '%topics/%' or referring_url ilike '%topics/%') or utm_medium='organic_search_topics' then 'topics'
            when lower(utm_source) in ('1', 'champion','influencer') then 'influencer'
            when lower(utm_source) in ('affiliate','affiliates') then 'affiliate'
            when lower(utm_source) in ('community','discord','community_channel') then 'community'
            when lower(utm_source) in ('ib','interviewbit') and lower(utm_medium) = 'midfunnel' then 'ib-midfunnel'
            when lower(utm_source) in ('ib','interviewbit','interviewbit%2f') OR lower(utm_source) like '%ib moco%' then 'interviewbit'
            when (lower(utm_medium) like '%reminder%' or lower(event_name) = '16_midfunnel') and referring_url ilike '%/interviewbit.%' then 'interviewbit'
            when lower(utm_source) in ('som') then 'seo'
            when lower(utm_source) in ('bing') then 'bing'
            when (lower(event_name) = '13_leadgen' and lower(trim(utm_source))='facebook') then 'facebook'
            when (lower(event_name) = '13_leadgen' and (lower(trim(utm_source))='linkedin' or lower(trim(utm_source))='linkedin-rerun')) then 'linkedin'
            when (lower(event_name) = '11_callback' and (lower(trim(utm_source))='linkedin' or lower(trim(utm_source))='linkedin-rerun')) then 'linkedin'
            when (lower(event_name) = '13_leadgen_li') then 'linkedin'
            when lower(utm_medium) like '%reminder%' then 'midfunnel'
            when lower(utm_source) ilike 'cheatsheet' or lower(utm_source) ilike 'e-guide/books' then 'midfunnel'
            when lower(event_name) = '16_midfunnel' then 'midfunnel'
            when lower(utm_source) like '%delacon%' OR  lower(utm_source) like '%contact us widget%' then 'organic'
            when lower(utm_medium) in ('instructor') then 'influencer'
            when (utm_source is null and code is not null and code <>0) or (utm_source='referral') then 'referral'
            when lower(utm_source) in ('none','na')  or lower(utm_source) ilike 'none%' then 'organic'
            when lower(utm_medium) in ('organic_search','referral') then 'organic'
            when lower(utm_source) like 'midfunnel%' then 'midfunnel'
            when ((landing_page_url like '%gclid=%') OR (landing_page_url like '%fbclid=%')) and (utm_source is null or utm_source = '') then 'other'
            when (utm_source is null or utm_source = '') and (referring_url ilike '%utm_source=ib-midfunnel%' or referring_url ilike '%interviewbit.com%') then 'interviewbit'
            when (utm_source is null or utm_source = '') then 'organic'
            when lower(utm_source) = 'ib-midfunnel' and lower(event_type) = lower('79_ScalerTopics') then 'ib-midfunnel'
            --NEW Addition
            when ((utm_source IS NULL or utm_source = '' or utm_source in ('na','NA','none')) AND (utm_medium IS NULL or utm_medium = '' or utm_medium in ('na','NA','organic_search')) AND (utm_campaign IS NULL or utm_campaign = '' or utm_campaign in ('na','NA')))
            or (lower(utm_source) ilike '%%none%%' and (lower(utm_medium) = 'direct' or lower(utm_medium) = 'top_nudge')) or (utm_medium ilike '%%organic%%') or (utm_source ilike '%%www.scaler.com%%' and utm_medium = 'referral') then 'organic'
            when lower(utm_medium) like '%reminder%' or utm_source='digital-events' then 'midfunnel'
            when event_name = 'Callback_IB_ProblemSolver' then 'interviewbit'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='none' then 'organic'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='e-scaler' then 'brandsearch'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='b-scaler' then 'brandsearch'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='similar' then 'googlediscovery'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='lookalike_payment' then 'googlediscovery'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='lookalike_sales-qualified' then 'googlediscovery'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='affinity' then 'googlediscovery'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='screen-lead-remarketing' then 'googlediscovery'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='screenedleads_weekend' then 'googlediscovery'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='inmarket' then 'googlediscovery'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='screenedleads' then 'googlediscovery'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='in-market-segments-clubbed' then 'googlediscovery'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='e1488_recurring_sde-bootcamp' then 'googlepmc'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='e1465_payment-apps' then 'googlepmc'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='e1499_math-dsa' then 'googlepmc'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' and lower(utm_content) ='e1551_dynamic-programming' then 'googlepmc'
            when lower(utm_source) = 'ads' and lower(utm_medium) ='google.com' then 'googleyoutube'
            else 'other'
        end as final_source 
        using utm_source column in temp.marketing_mis to get the final source and use event_rank_registraion = 1 when need to get the final_source
        Calculate this within the cte itself.
        25. Use CASE
    WHEN final_source IN ('organic', 'influencer', 'brandedcontent', 'brandcampaign', 'dv360', 'publisher-ideal') THEN '1.Organic'
    WHEN final_source IN ('facebook', 'googleyoutube', 'googlesearch', 'googlediscovery', 'linkedin', 'googlepmc', 'brandsearch', 'bing', 'googledisplay', 'googleuac', 'columbia', 'quora', 'taboola', 'reddit', 'yahoo', 'affiliate', 'rtbhouse', 'googleuac') THEN '2.Paid'
    WHEN final_source IN ('midfunnel', 'interviewbit', 'community', 'organic_social', 'seo', 'other', 'referral', 'ib-midfunnel', 'topics', 'topics-midfunnel') THEN '3.Non-Paid'
    ELSE '4.Other'
END AS Channels this for the channels column 
26. Whenever batch is required keep it as a first column. 
    """

    prompt = f"""Given the following reference logic and conversation history:

Reference Logic:
{reference_logic}

Conversation History:
{conversation_history}

Generate a SQL query for the following request: {user_input}

Return only the SQL query, without any explanations, comments, or formatting. Use the DISTINCT keyword where appropriate. Follow the structure and logic of the reference query, adapting it to the specific request. Learn from any feedback or corrections in the conversation history."""

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
        if st.button("Submit"):
            if change_request:
                conversation_history.append({"role": "user", "content": f"Change Request: {change_request}"})
                with st.spinner("Generating new query and fetching results..."):
                    new_generated_sql = generate_sql_query(change_request, conversation_history)

                    if new_generated_sql:
                        st.subheader("New Generated SQL query:")
                        st.code(new_generated_sql, language="sql")

                        conversation_history.append({"role": "assistant", "content": new_generated_sql})

                        new_results, new_cur = execute_query(st.session_state.conn, new_generated_sql)

                        if new_results is None and new_cur is None:
                            st.error("An error occurred while executing the query. Resetting the connection...")
                            reset_connection()
                            new_results, new_cur = execute_query(st.session_state.conn, new_generated_sql)

                        if new_results and new_cur:
                            st.subheader("New Query Results:")
                            new_df = pd.DataFrame(new_results)
                            new_column_names = [desc[0] for desc in new_cur.description]
                            new_df.columns = new_column_names
                            
                            st.session_state.query_results.append({
                                "question": change_request,
                                "query": new_generated_sql,
                                "dataframe": new_df
                            })

                            st.dataframe(new_df, use_container_width=True)

                            # Add download button for new CSV
                            new_csv = new_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download new results as CSV",
                                data=new_csv,
                                file_name="new_query_results.csv",
                                mime="text/csv",
                            )
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

    # Add a centered footer
    st.markdown("---")
    st.markdown('<p class="footer">Built with ❤️ by the Scaler Product Analytics Team</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
