import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import random

# Download NLTK data if not already available
nltk.download('punkt')

class AIBot:
    def __init__(self):
        # Define training data: User input mapped to intents
        self.training_data = [
            {"intent": "greeting", "text": "hi"},
            {"intent": "greeting", "text": "hello"},
            {"intent": "greeting", "text": "hey"},
            {"intent": "greeting", "text": "good morning"},
            {"intent": "farewell", "text": "bye"},
            {"intent": "farewell", "text": "goodbye"},
            {"intent": "farewell", "text": "see you"},
            {"intent": "info", "text": "what's your name"},
            {"intent": "info", "text": "who are you"},
            {"intent": "thanks", "text": "thank you"},
            {"intent": "thanks", "text": "thanks"},
        ]
        
        # Define possible responses for each intent
        self.responses = {
            "greeting": ["Hello!", "Hi there!", "Greetings!"],
            "farewell": ["Goodbye!", "See you later!", "Take care!"],
            "info": ["I am an AI chatbot created to help you.", "I'm your friendly bot!"],
            "thanks": ["You're welcome!", "Happy to help!", "No problem!"],
            "unknown": ["Sorry, I didn't understand that. Could you rephrase?"]
        }
        
        # Train the bot
        self.train()

    def train(self):
        # Prepare the training data
        texts = [item["text"] for item in self.training_data]
        labels = [item["intent"] for item in self.training_data]
        
        # Vectorize the text data (convert text to numerical format)
        self.vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
        X_train = self.vectorizer.fit_transform(texts)
        
        # Train a Naive Bayes classifier
        self.model = MultinomialNB()
        self.model.fit(X_train, labels)
    
    def get_response(self, user_input):
        # Preprocess the user input and predict the intent
        user_input_vectorized = self.vectorizer.transform([user_input])
        predicted_intent = self.model.predict(user_input_vectorized)[0]
        
        # Return a random response based on the predicted intent
        response_list = self.responses.get(predicted_intent, self.responses["unknown"])
        return random.choice(response_list)

def main():
    # Create an instance of the AI Bot
    bot = AIBot()
    
    print("AI Bot: Hi! I'm here to assist you. Type something or 'bye' to end.")
    
    while True:
        user_input = input("You: ").lower()
        if user_input in ['bye', 'exit', 'quit']:
            print("AI Bot: Goodbye! Have a great day!")
            break
        
        # Get the bot's response
        response = bot.get_response(user_input)
        print("AI Bot:", response)

# Run the bot
if __name__ == "__main__":
    main()
