# Multi-Restaurant Food Assistant

![Multi-Restaurant Assistant](https://img.shields.io/badge/Zomato-Project-FF4B4B)

A smart conversational AI assistant that helps users discover and compare menu items across multiple restaurants like Burger King, Dominos, Behrouz Biryani, and Biryani By Kilo. Built with Python, Streamlit, and RAG technology.

## 📋 Features

- **Multi-Restaurant Support**: Access menu data from 4 popular restaurants in a single interface
- **Smart Search**: Find dishes by name, category, price range, or dietary preferences
- **Restaurant Comparison**: Compare similar dishes across different restaurants
- **Dietary Filtering**: Easily find vegetarian and non-vegetarian options
- **Price Analysis**: Discover budget-friendly options or compare pricing across venues
- **Contextual Answers**: The assistant remembers your conversation to provide more relevant responses
- **Restaurant Focus Mode**: Choose to see information from all restaurants or focus on a specific one

## 🖥️ Screenshots

Multi-Restaurant Assistant Interface (Screenshot description: The application interface showing the main chat window with restaurant selector dropdown and sample conversation about menu items)

## 🛠️ Technology Stack

- **Frontend**: Streamlit for responsive web interface
- **Backend**: Python 3.9+
- **NLP Processing**: Sentence Transformers, NLTK
- **Search Engine**: TF-IDF Vectorization with Cosine Similarity
- **Data Storage**: JSON-based knowledge base
- **Text Generation**: Transformer-based language model

## 📊 Data Sources

The system includes menu data from:

- **Burger King**: 130+ menu items scraped from their official website
- **Dominos**: 160+ menu items including pizzas, sides, and beverages
- **Behrouz Biryani**: 70+ menu items focused on premium biryanis
- **Biryani By Kilo**: 75+ menu items including biryanis and complementary dishes

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository:

git clone https://github.com/IamAadityaMishra/multi-restaurant-assistant.git
cd multi-restaurant-assistant

2. Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

3. Install the required dependencies:

pip install -r requirements.txt

4. Run the application:

streamlit run app.py

5. The application will open in your default web browser at http://localhost:8501

## 🧠 How It Works

1. **Data Collection**: Restaurant menu data is scraped from websites and stored in a structured format
2. **Knowledge Base Creation**: A unified knowledge base is created combining all restaurant data
3. **Query Processing**: User questions are analyzed to identify intent and extract key information
4. **Retrieval**: Relevant menu items and restaurant information is retrieved using vector similarity
5. **Response Generation**: Coherent and helpful responses are generated based on the retrieved information
6. **Context Management**: Conversation history is maintained to understand follow-up questions

## 🤝 Project Structure

multi-restaurant-assistant/
│
├── app.py                   # Main Streamlit application
├── chatbot.py               # Improved RAG chatbot implementation
├── knowledge_base.py        # Knowledge base creation utilities
├── combined_restaurants_kb.json  # Combined knowledge base
│
├── Scraper/                 # Web scraping scripts
│   ├── Burger_King/
│   ├── Dominos/
│   ├── Behrouz/
│   └── BiryaniByKilo/
│
└── requirements.txt         # Project dependencies

## 📝 Sample Queries

- "Show me vegetarian options at Burger King"
- "What is the price range for biryanis at Behrouz?"
- "Compare the cheese pizza from Dominos with other restaurants"
- "Which restaurant has the cheapest burger?"
- "Tell me more about the Whopper"
- "What sides are available at Dominos?"
- "Show me items under ₹200 at all restaurants"

## 🔮 Future Improvements

- Add more restaurants to the knowledge base
- Implement user reviews and ratings
- Add nutritional information and allergen filters
- Enable ordering functionality
- Implement voice interaction
- Create a mobile application version

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors
Aaditya Mishra 
(https://github.com/IamAadityaMishra)

## 🙏 Acknowledgements

- Data sourced from restaurant websites for educational purposes
- Built with Streamlit (https://streamlit.io/)
- NLP capabilities powered by Sentence Transformers (https://www.sbert.net/) and NLTK (https://www.nltk.org/)

---

Made with ❤️ for food lovers everywhere
