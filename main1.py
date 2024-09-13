import streamlit as st
from openai import OpenAI
import psycopg2
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def get_relevant_feedback(conn, user_input, similarity_threshold=0.7, max_feedback=3):
    """Retrieve most relevant feedback for similar questions."""
    history = get_question_history(conn)
    relevant_feedback = []
    for prev_question, _, _, prev_feedback, prev_feedback_type in history:
        similarity = calculate_similarity(user_input, prev_question)
        if similarity >= similarity_threshold and prev_feedback:
            relevant_feedback.append((similarity, f"{prev_feedback_type}: {prev_feedback[:100]}"))
    
    # Sort by similarity and take top 3
    relevant_feedback.sort(key=lambda x: x[0], reverse=True)
    return [feedback for _, feedback in relevant_feedback[:max_feedback]]

def generate_sql_query(user_input, conversation_history, relevant_feedback):
    """Generate a SQL query based on the user's input and relevant feedback."""
    feedback_prompt = "\n".join(relevant_feedback) if relevant_feedback else "No relevant feedback."
    
    # Truncate conversation history to last 3 messages
    recent_history = conversation_history[-3:]
    history_prompt = "\n".join([f"{msg['role']}: {msg['content'][:50]}..." for msg in recent_history])

    prompt = f"""Generate a SQL query for: {user_input}

Relevant Feedback:
{feedback_prompt}

Recent Conversation:
{history_prompt}

Important: Use temp.non_marketing_mis for non-marketing leads, temp.marketing_mis for marketing leads. Choose based on context. Use DISTINCT where appropriate. Return only the SQL query."""

    generated_sql = get_gpt4_response(prompt, conversation_history)
    return clean_sql_query(generated_sql)

def get_conversation_history():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    return st.session_state.conversation_history

def create_history_table(conn):
    """Create a table to store question history, ratings, and user feedback if it doesn't exist."""
    query = """
    CREATE TABLE IF NOT EXISTS sql_llm_question_history (
        id INT IDENTITY(1,1),
        question VARCHAR(MAX) NOT NULL,
        query VARCHAR(MAX) NOT NULL,
        rating INT,
        user_feedback VARCHAR(MAX),
        feedback_type VARCHAR(20),
        created_at TIMESTAMP DEFAULT SYSDATE
    )
    DISTSTYLE AUTO
    SORTKEY (created_at);
    """
    execute_query(conn, query)

def get_question_history(conn):
    """Retrieve all questions, queries, ratings, and feedback from the database."""
    # First, check if the table exists
    check_table_query = """
    SELECT COUNT(*)
    FROM pg_tables
    WHERE schemaname = 'public' AND tablename = 'sql_llm_question_history';
    """
    result, _ = execute_query(conn, check_table_query)
    
    if result and result[0][0] > 0:
        # Table exists, fetch the data
        select_query = "SELECT question, query, rating, user_feedback, feedback_type FROM sql_llm_question_history"
        results, _ = execute_query(conn, select_query)
        return results if results is not None else []
    else:
        # Table doesn't exist, create it
        create_history_table(conn)
        return []

def save_question_history(conn, question, query, rating, user_feedback=None, feedback_type=None):
    """Save a question, its query, rating, and user feedback to the database."""
    insert_query = """
    INSERT INTO sql_llm_question_history (question, query, rating, user_feedback, feedback_type)
    VALUES (%s, %s, %s, %s, %s)
    """
    with conn.cursor() as cur:
        cur.execute(insert_query, (question, query, rating, user_feedback, feedback_type))
    conn.commit()

def calculate_similarity(question1, question2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([question1, question2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def find_similar_question(conn, new_question, similarity_threshold=0.8):
    """Find a similar question from the database."""
    history = get_question_history(conn)
    for prev_question, prev_query, prev_rating, prev_feedback, prev_feedback_type in history:
        similarity = calculate_similarity(new_question, prev_question)
        if similarity >= similarity_threshold:
            return prev_question, prev_query, prev_rating, prev_feedback, prev_feedback_type
    return None, None, None, None, None

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

    # Create history table if it doesn't exist
    create_history_table(st.session_state.conn)

    # Load question history from database
    if 'previous_questions' not in st.session_state:
        history = get_question_history(st.session_state.conn)
        st.session_state.previous_questions = [(q, r) for q, _, r, _, _ in history]

    conversation_history = get_conversation_history()

    # User input
    user_input = st.text_input("Enter your question:", placeholder="e.g., Show me the top 5 mentees by attendance")

    if 'query_results' not in st.session_state:
        st.session_state.query_results = []

    if st.button("Submit", key="submit"):
        if user_input:
            with st.spinner("Generating query and fetching results..."):
                # Check for similar questions in the database
                similar_question, similar_query, similar_rating, similar_feedback, similar_feedback_type = find_similar_question(st.session_state.conn, user_input)
                
                relevant_feedback = get_relevant_feedback(st.session_state.conn, user_input)
                
                if similar_question:
                    st.info(f"A similar question was found with a rating of {similar_rating}/5.")
                    if similar_feedback:
                        st.info(f"Previous feedback: {similar_feedback[:100]}...")
                    generated_sql = similar_query
                    # Execute the similar query to get results
                    results, cur = execute_query(st.session_state.conn, generated_sql)
                    if results and cur:
                        df = pd.DataFrame(results)
                        df.columns = [desc[0] for desc in cur.description]
                    else:
                        st.warning("No results found for the similar query.")
                        df = None
                else:
                    # Generate new query as before
                    generated_sql = generate_sql_query(user_input, conversation_history, relevant_feedback)
                    
                    if generated_sql:
                        results, cur = execute_query(st.session_state.conn, generated_sql)
                        if results and cur:
                            df = pd.DataFrame(results)
                            df.columns = [desc[0] for desc in cur.description]
                        else:
                            st.warning("No results found for the generated query.")
                            df = None

                # Display results
                if generated_sql:
                    st.subheader("Generated SQL query:")
                    st.code(generated_sql, language="sql")
                    
                    if df is not None:
                        st.subheader("Query results:")
                        st.dataframe(df, use_container_width=True)
                        
                        # Add new results to the session state
                        st.session_state.query_results.append({
                            "question": user_input,
                            "query": generated_sql,
                            "dataframe": df
                        })
                        
                        # Add rating system with a visible scale
                        st.subheader("Rate this response:")
                        rating = st.slider("", min_value=1, max_value=5, value=3, step=1, 
                                           help="1 = Poor, 2 = Fair, 3 = Good, 4 = Very Good, 5 = Excellent")
                        
                        # Add text area for optional feedback
                        feedback = st.text_area("Additional feedback (optional):", 
                                                help="Provide any comments or suggestions for improvement")
                        
                        if st.button("Submit Rating and Feedback"):
                            save_question_history(st.session_state.conn, user_input, generated_sql, rating, feedback, "user_feedback")
                            st.success(f"Thank you! Your rating of {rating}/5 and feedback have been recorded.")
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
