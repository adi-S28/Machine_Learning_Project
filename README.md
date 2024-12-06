## Sentiment Analysis Machine Learning App

The sentiment analysis project leverages advanced data processing methodologies to clean and prepare text data for effective analysis. Using TF-IDF vectorisation, the text is transformed into a numerical format suitable for machine learning models. The core of the project involves training a Naive Bayes classifier, which efficiently categorises input text into positive or negative sentiments. The project incorporates user-friendly technologies such as Streamlit for the interface and SQLite for data management, ensuring seamless interaction and storage. Users can input text, which is processed to generate sentiment predictions that are visually presented for easy interpretation. Along the development process, various challenges were encountered, including data handling and implementation complexities, which were systematically resolved to ensure the smooth functioning of the system.

## Demo App

[![Streamlit App]([https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-starter-kit.streamlit.app/](https://github.com/adi-S28/Machine_Learning_Project/blob/master/streamlit_app.py)


A sentiment analysis project made by me and my classmates

## Some Description:

Introduction Sentiment analysis is a natural language processing (NLP) technique used to determine the emotional tone behind a body of text. It finds widespread use in applications such as customer review analysis, social media monitoring, and opinion mining. The objective of this project is to develop an interactive web-based application using machine learning that can classify user input into positive or negative sentiment. The application is built using Streamlit, a Python-based framework for creating interactive web apps. The underlying machine learning model is a Naive Bayes classifier, and the text data is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency). The app also includes a user authentication system, sentiment analysis result storage in an SQLite database, and a real-time visualization of results. This report provides an in-depth look at the architecture of the application, the algorithmic approach, experimental results, and future directions for improvement.


### Problem Definition and Algorithm 
2.1 Task Definition Sentiment analysis aims to classify text data into predefined categories of sentiment, usually positive, negative, or neutral. In this project, we focus on binary classification (positive/negative). The challenge lies in accurately interpreting the sentiment behind a wide range of text inputs, which may vary in complexity, tone, and vocabulary. The inputs to the model are text strings provided by users through the web app’s interface. The output is a predicted sentiment label, either “positive” or “negative.” 2.2 Algorithm Definition The Naive Bayes classifier is particularly suitable for text classification due to its simplicity, scalability, and competitive performance. It assumes that the features (in this case, words in the text) are conditionally independent, which simplifies computation. TF-IDF is used to transform the input text into a numerical format. The TF-IDF scores represent the importance of each word in the context of the document. Words that appear frequently across all documents receive lower scores, while words that are unique to a specific document are given higher importance. 

### Pseudocode:

Preprocess the text: Tokenization, lowercasing, removal of stopwords
Vectorize the text using TF-IDF
Train the Naive Bayes classifier on the training set
Make predictions on new user input
Store and visualize the results


### Experimental Evaluation:

3.1 Methodology We trained the Naive Bayes model on a sample dataset and evaluated its performance using accuracy, precision, recall, and F1-score. The text data was split into a training set (80%) and a test set (20%) using train_test_split.
3.2 Results The model achieved an accuracy of X% on the test data. A detailed breakdown of precision, recall, and F1-score is provided in the classification report. Visualizations:
Confusion matrix for model predictions, Bar charts comparing the number of positive vs. negative predictions 
3.3 Discussion The results show that the Naive Bayes classifier performs well on simple sentiment classification tasks. However, its performance degrades for more complex inputs, such as sarcastic statements or mixed sentiment.


Related Work Sentiment analysis has been widely studied in the field of NLP. Traditional approaches like Naive Bayes and SVMs have given way to more sophisticated models like LSTMs and BERT, which are better at capturing contextual nuances in text. This project demonstrates the viability of simpler models in real-time applications.
Future Work Future improvements could include integrating advanced models such as BERT for more accurate sentiment analysis, especially on more challenging datasets. Additionally, expanding the app to handle multilingual text or adding more sentiment categories (neutral, mixed) could enhance its usability.
Conclusion This project demonstrates the power of machine learning in creating an interactive sentiment analysis application. The simplicity of the Naive Bayes classifier makes it suitable for real-time sentiment predictions. Future work will focus on addressing the model’s limitations and expanding its capabilities.


P.S. We still have a long way to go but this project did help us get a start in our journey

To use:just run the code and You are good to go
