import streamlit as st
import pandas as pd
import sqlite3
import base64
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
import bcrypt

class DatabaseHandler:
    def __init__(self):
        self.conn = self._get_db_connection()
        self.create_tables()
        
    @st.cache_resource
    def _get_db_connection(_self):
        conn = sqlite3.connect('users.db', check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn
    
    def create_tables(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS users 
                               (username TEXT PRIMARY KEY, password TEXT, profile_pic TEXT)''')
            self.conn.execute('''CREATE TABLE IF NOT EXISTS sentiment_analysis_results 
                               (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, text TEXT, 
                               sentiment TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            self.conn.commit()
    
    def add_user(self, username, password, profile_pic):
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        try:
            with self.conn:
                self.conn.execute('INSERT INTO users VALUES (?, ?, ?)', 
                                 (username, hashed_password.decode('utf-8'), profile_pic))
                self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def get_user(self, username):
        with self.conn:
            return self.conn.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
    
    def add_analysis_result(self, username, text, sentiment):
        with self.conn:
            self.conn.execute('''INSERT INTO sentiment_analysis_results 
                               (username, text, sentiment) VALUES (?, ?, ?)''', 
                            (username, text, sentiment))
            self.conn.commit()
    
    def get_analysis_results(self, username):
        with self.conn:
            return self.conn.execute('''SELECT text, sentiment, timestamp 
                                      FROM sentiment_analysis_results 
                                      WHERE username=? 
                                      ORDER BY timestamp DESC''', (username,)).fetchall()
    
    def get_sentiment_counts(self, username):
        with self.conn:
            results = self.conn.execute('''SELECT sentiment, COUNT(*) 
                                          FROM sentiment_analysis_results 
                                          WHERE username=? 
                                          GROUP BY sentiment''', (username,)).fetchall()
        return dict(results) if results else {'positive': 0, 'negative': 0}

class UserAuth:
    def __init__(self, db_handler):
        self.db = db_handler
        self._init_session_state()
        
    def _init_session_state(self):
        if 'username' not in st.session_state:
            st.session_state['username'] = 'Guest'
        if 'profile_pic' not in st.session_state:
            st.session_state['profile_pic'] = "a.webp"
    
    def signup(self, username, password, profile_pic):
        if self.db.add_user(username, password, profile_pic):
            st.session_state.update({
                'username': username,
                'profile_pic': profile_pic
            })
            return True
        return False
    
    def login(self, username, password):
        user = self.db.get_user(username)
        if user and bcrypt.checkpw(password.encode('utf-8'), user[1].encode('utf-8')):
            st.session_state.update({
                'username': user[0],
                'profile_pic': user[2] or "a.webp"
            })
            return True
        return False

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.accuracy = None
        self.report = None
        self._initialize_model()

    @st.cache_data
    def _load_data(_self):
        columns = ['id', 'entity', 'sentiment', 'text']
        train_data = pd.read_csv('twitter_training.csv', names=columns, header=None)
        test_data = pd.read_csv('twitter_validation.csv', names=columns, header=None)
        return train_data, test_data

    @st.cache_resource
    def _train_model(_self, train_data, test_data):
        # Helper to process data
        def process_df(df):
            df = df[df['sentiment'].isin(['Positive', 'Negative'])]
            df['sentiment_numeric'] = df['sentiment'].map({'Negative': 0, 'Positive': 1})
            return df.dropna(subset=['sentiment_numeric', 'text'])

        # Clean and prepare data
        train_data = process_df(train_data)
        test_data = process_df(test_data)
        
        X_train = train_data['text']
        y_train = train_data['sentiment_numeric']
        X_test = test_data['text']
        y_test = test_data['sentiment_numeric']
        
        # Vectorize text
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train classifier
        classifier = MultinomialNB()
        classifier.fit(X_train_vec, y_train)
        
        # Calculate metrics
        y_pred = classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['negative', 'positive'])
        
        return vectorizer, classifier, accuracy, report

    def _initialize_model(self):
        train_data, test_data = self._load_data()
        self.vectorizer, self.classifier, self.accuracy, self.report = self._train_model(train_data, test_data)

    def predict(self, text):
        if not text.strip():
            return None
        vec_text = self.vectorizer.transform([text])
        prediction = self.classifier.predict(vec_text)[0]
        return 'positive' if prediction == 1 else 'negative'
    
class App:
    def __init__(self):
        self.db = DatabaseHandler()
        self.auth = UserAuth(self.db)
        self.analyzer = SentimentAnalyzer()
    
    def run(self):
        self._render_sidebar()
        self._handle_page_navigation()
    
    def _render_sidebar(self):
        st.sidebar.title("Navigation")
        st.sidebar.image(
            st.session_state['profile_pic'], 
            caption=st.session_state['username'],
            use_container_width=True
        )
        self.current_page = st.sidebar.selectbox("Choose a page", [
            "Login", "Sign Up", "Continue as Guest", 
            "Sentiment Analysis", "Metrics"
        ])
    
    def _handle_page_navigation(self):
        if self.current_page == "Login":
            self._render_login_page()
        elif self.current_page == "Sign Up":
            self._render_signup_page()
        elif self.current_page == "Continue as Guest":
            self._render_guest_page()
        elif self.current_page == "Sentiment Analysis":
            self._render_analysis_page()
        elif self.current_page == "Metrics":
            self._render_metrics_page()
    
    def _render_login_page(self):
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if self.auth.login(username, password):
                st.success(f"Welcome back {username}!")
            else:
                st.error("Invalid credentials")
    
    def _render_signup_page(self):
        st.title("Sign Up")
        username = st.text_input("New Username")
        password = st.text_input("New Password", type='password')
        profile_pic = self._handle_profile_pic_upload()
        
        if st.button("Create Account"):
            if self.auth.signup(username, password, profile_pic):
                st.success("Account created successfully!")
            else:
                st.error("Username already exists")
    
    def _handle_profile_pic_upload(self):
        st.subheader("Profile Picture")
        default_pic = "a.webp"
        uploaded = st.file_uploader("Upload image (optional)", type=["jpg", "png", "jpeg"])
        return (
            f"data:image/jpeg;base64,{base64.b64encode(uploaded.read()).decode()}" 
            if uploaded else default_pic
        )
    
    def _render_guest_page(self):
        st.title("Guest Mode")
        st.session_state.update({
            'username': 'Guest',
            'profile_pic': "a.webp"
        })
        st.info("Using guest account with limited features")
    
    def _render_analysis_page(self):
        st.title("Sentiment Analysis")
        text_input = st.text_area("Enter text for analysis")
        
        if st.button("Analyze"):
            sentiment = self.analyzer.predict(text_input)
            if sentiment:
                self._display_result(text_input, sentiment)
                self.db.add_analysis_result(st.session_state['username'], text_input, sentiment)
                self._show_previous_results()
    
    def _display_result(self, text, sentiment):
        st.success("Analysis Complete!")
        st.markdown(f"**Text:** {text}")
        st.markdown(f"**Sentiment:** {sentiment.capitalize()}")
    
    def _show_previous_results(self):
        st.subheader("History")
        results = self.db.get_analysis_results(st.session_state['username'])
        if results:
            st.table(pd.DataFrame(results, columns=["Text", "Sentiment", "Timestamp"]))
    
    def _render_metrics_page(self):
        st.title("Analysis Metrics")
        counts = self.db.get_sentiment_counts(st.session_state['username'])
        
        st.subheader("Sentiment Distribution")
        if counts:
            st.bar_chart(pd.DataFrame.from_dict(counts, orient='index'))
        else:
            st.info("No analysis data available")

if __name__ == "__main__":
    app = App()
    app.run()
