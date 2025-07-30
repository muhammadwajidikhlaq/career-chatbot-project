# Import necessary libraries
import streamlit as st         # For creating the web app interface
import pandas as pd            # For handling the CSV dataset
import joblib                  # For loading the saved ML model and vectorizer

# Set the web page title and layout to wide screen
st.set_page_config(page_title="Career Guidance Chatbot", layout="wide")

# Load the trained machine learning model and vectorizer
model = joblib.load("intent_model.pkl")                # Load the trained intent classification model
vectorizer = joblib.load("vectorizer.pkl")             # Load the text vectorizer (TF-IDF or CountVectorizer)
df = pd.read_csv("career_guidance_dataset.csv")        # Load the career guidance dataset containing role and answer

# Add custom CSS styling for background, text input, buttons, etc.
st.markdown("""
<style>
/* Set a full-screen fixed background image */
.stApp {
    background: url("https://images.unsplash.com/photo-1503676260728-1c00da094a0b?auto=format&fit=crop&w=1950&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Add a semi-transparent dark overlay to the main content */
.main > div {
    background-color: rgba(0, 0, 0, 0.65);
    padding: 2rem;
    border-radius: 10px;
}

/* Style the text input field */
.stTextInput input {
    color: white !important;                        /* White text inside input box */
    background-color: rgba(0, 0, 0, 0.7);            /* Semi-transparent dark background */
    border-radius: 8px;
    border: 1px solid #ccc;
    padding: 12px;
    font-size: 16px;
}

/* Change placeholder text color */
.stTextInput input::placeholder {
    color: #dddddd;
}

/* Style for the chatbot's response area */
.response-box {
    background-color: rgba(0,0,0,0.75);
    color: white;
    padding: 20px;
    border-radius: 12px;
    margin-top: 25px;
}

/* Style the main heading */
h1 {
    color: white;
    font-size: 42px;
    text-align: center;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# Display the heading with an icon
st.markdown("""
<h1>
    <img src="https://cdn-icons-png.flaticon.com/512/3135/3135789.png" width="50" style="margin-bottom:-8px; margin-right:10px;">
    Career Guidance Chatbot
</h1>
""", unsafe_allow_html=True)

# ---- SIDEBAR: Display Search History ----
st.sidebar.title("📜 Search History")             # Sidebar title
if "history" not in st.session_state:             # Initialize history if not present
    st.session_state.history = []

# Display previous user queries in reverse order
for i, h in enumerate(reversed(st.session_state.history), 1):
    st.sidebar.write(f"{i}. {h}")

# ---- INPUT SECTION: Text field and submit button in a form ----
with st.form("input_form"):                                               # Create a form to handle text input and submission
    user_input = st.text_input(                                           # Create the text input field
        "Enter your interest or question:", 
        placeholder="e.g., AI, business, design"
    )
    submitted = st.form_submit_button("⬆️")                               # Create the submit button (Enter icon)

# ---- MAIN LOGIC: Make prediction and show response ----
if submitted and user_input:                                              # If the form is submitted and user typed something
    cleaned_input = user_input.lower().strip()                            # Clean the input (lowercase and strip spaces)
    input_vector = vectorizer.transform([cleaned_input])                 # Convert the text into a vector for prediction
    predicted_role = model.predict(input_vector)[0]                      # Predict the most suitable career/role

    matched_row = df[df['role'] == predicted_role]                       # Find matching role from the CSV
    st.session_state.history.append(user_input)                          # Add user input to session history

    if not matched_row.empty:                                            # If a matching role was found in the dataset
        answer = matched_row.iloc[0]['answer']                           # Get the explanation/description of that role
        st.markdown(f"""                                                 # Show the result in a styled box
            <div class='response-box'>
                <h3>🎯 Suggested Career: <span style='color:#90ee90'>{predicted_role}</span></h3>
                <p>{answer}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # If no match was found, show an error message
        st.markdown("""
            <div class='response-box'>
                <h4>❌ No career match found.</h4>
                <p>Please try a different keyword or spelling.</p>
            </div>
        """, unsafe_allow_html=True)
