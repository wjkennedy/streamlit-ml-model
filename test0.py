import streamlit as st
from jira import JIRA
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Setup and Configuration
# Load your configuration (API Key, User Email, etc.) here

# Streamlit UI
st.title("Jira Data Export and Clustering")

# Step 2: UI for Configuration
api_key = st.text_input("API Key", "")
user_email = st.text_input("User Email", "")
jira_url = st.text_input("Jira URL", "")
jql_query = st.text_area("JQL Query", "")

if st.button("Retrieve Data"):
    # Step 3: Data Retrieval
    # Placeholder for Jira connection and data retrieval
    options = {"server": jira_url}
    jira = JIRA(options, basic_auth=(user_email, api_key))
    
    # Assuming you're retrieving issues based on JQL query
    issues = jira.search_issues(jql_query)
    
    # Convert issues to DataFrame
    data = [{"summary": issue.fields.summary} for issue in issues]
    df = pd.DataFrame(data)
    
    # Step 4: Data Preprocessing
    # Placeholder for text preprocessing
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['summary'])
    
    # Step 5: Clustering
    num_clusters = st.slider("Number of Clusters", 2, 10, 5)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    df['cluster'] = kmeans.labels_
    
    # Step 6: Visualization and Export
    st.write(df)
    if st.button("Export Data"):
        df.to_csv("jira_data_clusters.csv")
        st.success("Data exported successfully!")
