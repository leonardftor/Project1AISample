from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import streamlit as st
import time


# --- 1. DATA GENERATION ENGINE ---
@st.cache_data
def generate_tpm_data(rows=40):
    np.random.seed(42)
    teams = ['Platform', 'Patient Portal', 'Billing Engine', 'Mobile App']
    data = {
        'ticket_id': [f'PCC-{i}' for i in range(100, 100 + rows)],
        'team': np.random.choice(teams, rows),
        'story_points': np.random.choice([1, 2, 3, 5, 8, 13], rows),
        'days_in_progress': np.random.randint(1, 25, rows),
        'reassignments': np.random.randint(0, 7, rows),
        'comment_sentiment': np.round(np.random.uniform(0, 1, rows), 2)
    }
    df = pd.DataFrame(data)
    df.loc[df['reassignments'] > 3, 'days_in_progress'] += 10
    return df

# --- 2. BACKEND ANALYSIS ENGINE ---
@st.cache_data
def detect_bottlenecks(df):
    model = IsolationForest(contamination=0.15, random_state=42)
    # Analysis based on velocity-killing metrics
    df['is_bottleneck'] = model.fit_predict(df[['days_in_progress', 'reassignments', 'comment_sentiment']])
    return df[df['is_bottleneck'] == -1]


# The following lines are for testing the backend logic independently of the Streamlit UI.
# print(generate_tpm_data()) # Uncomment to see generated data sample
# df = generate_tpm_data() # Generate synthetic data
# bottlenecks = detect_bottlenecks(df)
# print("Identified Bottlenecks:")
# print(bottlenecks[['ticket_id', 'team', 'days_in_progress', 'reassignments', 'comment_sentiment']])


# --- 1. FRONTEND UI ---
st.set_page_config(page_title="TPM AI Insights Engine", layout="wide")

st.title("🚀 An example of AI-Powered Delivery Insights")
st.markdown("""
This tool demonstrates how the use of AI powered delivery insights can move from reactive status reporting to 
**proactive risk mitigation**. It detects anomalies in the SDLC flow that human intuition might miss.
""")

# Sidebar Engagement
useremails = []
st.sidebar.header("📬 Stay Ahead of Risks")
user_email = st.sidebar.text_input("Enter your email for real-time risk alerts:")
if st.sidebar.button("Subscribe to Alerts"):
    if "@" in user_email:
        st.sidebar.success(f"Successfully subscribed {user_email}!")
        useremails.append(user_email)
        useremails = list(set(useremails))  # Ensure uniqueness
        useremails_str = ", ".join(useremails)
        st.sidebar.info(f"Current Subscribers: {useremails_str}")
        writer = st.sidebar.empty()
        writer.write(f"Alert: New subscriber added - {user_email}")
        # In a real app, you'd trigger a Lambda/SMTP function here
    else:
        st.sidebar.error("Please enter a valid email.")


# Main Dashboard Logic
df_health = generate_tpm_data()
bottlenecks = detect_bottlenecks(df_health)

# Metrics Row
col1, col2, col3 = st.columns(3)
col1.metric("Total Active Tickets", len(df_health))
col2.metric("Critical Bottlenecks", len(bottlenecks), delta_color="inverse")
col3.metric("Avg Cycle Time", f"{df_health['days_in_progress'].mean():.1f} Days")

# Visualizations
st.subheader("Team-Level Risk Analysis")

chart_data = df_health.copy()
chart_data['Status'] = chart_data['ticket_id'].apply(lambda x: 'Risk' if x in bottlenecks['ticket_id'].values else 'Healthy')

st.scatter_chart(
    chart_data, 
    x='days_in_progress', 
    y='reassignments', 
    color='Status',
    size='story_points'
)

# Detailed Bottleneck Report
st.subheader("⚠️ Supervisor Intervention Required")
st.write("The AI has flagged the following tickets as statistically significant anomalies:")
st.dataframe(bottlenecks[['ticket_id', 'team', 'days_in_progress', 'reassignments', 'comment_sentiment']], use_container_width=True)

if st.button("Generate Executive Briefing"):
    with st.spinner('AI is generating status report...'):
        time.sleep(1.5)
        st.info(f"Summary: Found {len(bottlenecks)} items with high reassignment counts. Recommendation: Re-align scope for {bottlenecks.iloc[0]['team']} team.")