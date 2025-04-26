import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
from sentence_transformers import SentenceTransformer

# Download necessary NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class ImprovedRestaurantChatbot:
    def __init__(self, knowledge_base_path):
        """Initialize the Restaurant RAG chatbot with improved response generation"""
        # Load the knowledge base
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            self.kb = json.load(f)
        
        # Use a better language model from Hugging Face
        try:
            # A more powerful conversational model that works well for Q&A
            model_name = "google/flan-t5-large"  # Upgrade from small to large for better understanding
            
            # Initialize the model with lower precision to improve performance
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = None  # Lazy load the model when needed
            
            # Create a generate function that loads model on first use
            self.has_model = True
            self._model_name = model_name
            
            # Try to load sentence transformer for better semantic matching
            try:
                self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                self.use_semantic_search = True
                print("Using semantic search for better understanding")
            except:
                self.use_semantic_search = False
                print("Falling back to TF-IDF search")
                
        except Exception as e:
            print(f"Error initializing model connection: {e}")
            print("Falling back to rule-based responses")
            self.has_model = False
        
        # Create TF-IDF vectorizer for retrieval
        self.vectorizer = TfidfVectorizer(
            stop_words=stopwords.words('english'),
            ngram_range=(1, 2)  # Use both unigrams and bigrams for better matching
        )
        
        # Create document corpus for vectorization, accounting for multiple restaurants
        self.documents = []
        self.doc_to_item_map = {}
        self.restaurants = self.kb.get('restaurants', [])
        self.restaurant_names = [r['name'] for r in self.restaurants]
        
        print(f"Loading {len(self.kb['menu_items'])} menu items from {len(self.restaurants)} restaurants")
        
        # Create embedded representations of menu items
        for i, item in enumerate(self.kb['menu_items']):
            restaurant = item.get('restaurant', 'Unknown')
            if not restaurant or restaurant == "":
                continue
                
            # Create comprehensive document text
            doc = f"{item['name']} {item.get('description', '')} {restaurant} {' '.join(item.get('categories', []))}"
            self.documents.append(doc)
            self.doc_to_item_map[i] = item
        
        # Fit the vectorizer if documents exist
        if self.documents:
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)
            if self.use_semantic_search:
                # Create semantic embeddings for all items (this may take time initially but improves results)
                try:
                    self.semantic_embeddings = self.sentence_model.encode(self.documents, show_progress_bar=True)
                    print(f"Created semantic embeddings for {len(self.documents)} menu items")
                except Exception as e:
                    print(f"Error creating semantic embeddings: {e}")
                    self.use_semantic_search = False
        else:
            self.doc_vectors = None
        
        # Initialize conversation history
        self.conversation_history = []
        self.max_history_length = 5
        
        # Store user's current restaurant focus (if any)
        self.current_restaurant_focus = None
        
        # Create category mappings for faster lookup
        self._create_category_mappings()
        
        # Create restaurant-specific item groupings
        self._create_restaurant_groupings()
        
        # Create price heatmap
        self._analyze_price_ranges()
    
    def _lazy_load_model(self):
        """Lazy load the model on first use to save memory"""
        if not self.model:
            try:
                print("Loading language model (first use)...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self._model_name,
                    torch_dtype=torch.float16,  # Use half precision
                    device_map="auto"  # Auto device placement
                )
                self.generator = pipeline(
                    'text2text-generation', 
                    model=self.model, 
                    tokenizer=self.tokenizer,
                    max_length=256,
                    temperature=0.3,  # Lower temperature for more focused answers
                    do_sample=True
                )
                print("Model loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Falling back to rule-based responses")
                self.has_model = False
                return False
        return True
        
    def _create_category_mappings(self):
        """Create mappings for faster lookup of items by category"""
        self.category_to_items = {}
        
        # Get all unique categories across all restaurants
        for category in self.kb.get('categories', []):
            self.category_to_items[category] = []
            
        # Assign items to categories
        for item in self.kb.get('menu_items', []):
            for category in item.get('categories', []):
                if category in self.category_to_items:
                    self.category_to_items[category].append(item)
    
    def _create_restaurant_groupings(self):
        """Group menu items by restaurant for faster lookups"""
        self.restaurant_to_items = {}
        for restaurant in self.restaurant_names:
            self.restaurant_to_items[restaurant] = []
            
        # Assign items to restaurants
        for item in self.kb.get('menu_items', []):
            restaurant = item.get('restaurant')
            if restaurant and restaurant in self.restaurant_to_items:
                self.restaurant_to_items[restaurant].append(item)
    
    def _analyze_price_ranges(self):
        """Analyze price ranges for different restaurants and categories"""
        # Calculate price ranges by restaurant
        self.price_by_restaurant = {}
        
        for restaurant in self.restaurant_names:
            items = self.restaurant_to_items.get(restaurant, [])
            prices = [item.get('regular_price', 0) for item in items if 'regular_price' in item]
            
            if prices:
                self.price_by_restaurant[restaurant] = {
                    'min': min(prices),
                    'max': max(prices),
                    'avg': sum(prices) / len(prices),
                    'median': sorted(prices)[len(prices) // 2] if prices else 0
                }
            else:
                self.price_by_restaurant[restaurant] = {'min': 0, 'max': 0, 'avg': 0, 'median': 0}
        
        # Calculate price ranges by category
        self.price_by_category = {}
        
        for category, items in self.category_to_items.items():
            prices = [item.get('regular_price', 0) for item in items if 'regular_price' in item]
            
            if prices:
                self.price_by_category[category] = {
                    'min': min(prices),
                    'max': max(prices),
                    'avg': sum(prices) / len(prices),
                    'median': sorted(prices)[len(prices) // 2] if prices else 0
                }
    
    def retrieve(self, query, top_k=5, restaurant_filter=None):
        """
        Retrieve relevant menu items for the query using semantic search or TF-IDF
        
        Args:
            query (str): The user's query
            top_k (int): Number of items to retrieve
            restaurant_filter (str): Optional filter to only return items from a specific restaurant
            
        Returns:
            List of (item, score) tuples
        """
        if not self.documents:
            return []
            
        results = []
        
        # If user has a restaurant focus or explicitly provides one, use it
        restaurant_name = restaurant_filter or self.current_restaurant_focus
        
        # Try semantic search first (if available) for better results
        if self.use_semantic_search:
            try:
                # Encode the query
                query_embedding = self.sentence_model.encode(query)
                
                # Calculate cosine similarities
                similarities = []
                for i, doc_embedding in enumerate(self.semantic_embeddings):
                    # Apply restaurant filter if specified
                    item = self.doc_to_item_map[i]
                    if restaurant_name and item.get('restaurant') != restaurant_name:
                        similarities.append(-1)  # Ensure it's not selected
                    else:
                        # Calculate cosine similarity
                        sim = np.dot(query_embedding, doc_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                        )
                        similarities.append(sim)
                
                # Get top k results
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                # Filter out negative similarities (which means no match)
                results = [(self.doc_to_item_map[i], float(similarities[i])) 
                          for i in top_indices if similarities[i] > 0]
                
                if results:
                    return results
            except Exception as e:
                print(f"Semantic search error: {e}")
                print("Falling back to TF-IDF")
        
        # Fall back to TF-IDF if semantic search fails or isn't available
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        if restaurant_name:
            # Filter for specific restaurant items
            restaurant_indices = [
                i for i, item in self.doc_to_item_map.items() 
                if item.get('restaurant') == restaurant_name
            ]
            
            # Get top items from that restaurant
            restaurant_similarities = [(i, similarities[i]) for i in restaurant_indices]
            restaurant_similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = [
                (self.doc_to_item_map[i], score) 
                for i, score in restaurant_similarities[:top_k] 
                if score > 0
            ]
            
            if results:  # If we found matches in the specified restaurant
                return results
        
        # If no restaurant filter or no results with filter, return global top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = [
            (self.doc_to_item_map[i], float(similarities[i])) 
            for i in top_indices 
            if similarities[i] > 0
        ]
        
        return results
    
    def find_items_by_criteria(self, criteria):
        """Find menu items matching given criteria"""
        results = []
        
        # If restaurant is specified, only search within that restaurant
        restaurant_filter = criteria.get('restaurant')
        items_to_search = (
            self.restaurant_to_items.get(restaurant_filter, []) 
            if restaurant_filter 
            else self.kb.get('menu_items', [])
        )
        
        for item in items_to_search:
            match = True
            
            # Check category
            if 'category' in criteria and criteria['category'] not in item.get('categories', []):
                match = False
                
            # Check veg/non-veg status
            if 'veg_status' in criteria and criteria['veg_status'] != item.get('veg_status'):
                match = False
                
            # Check price range
            if 'price_min' in criteria and item.get('regular_price', float('inf')) < criteria['price_min']:
                match = False
            if 'price_max' in criteria and item.get('regular_price', 0) > criteria['price_max']:
                match = False
                
            # Check name contains
            if 'name_contains' in criteria:
                if criteria['name_contains'].lower() not in item['name'].lower():
                    match = False
                
            # Check allergens
            if 'allergen_free' in criteria:
                for allergen in criteria['allergen_free']:
                    if allergen in item.get('allergens', []):
                        match = False
                        break
            
            if match:
                results.append(item)
        
        return results
    
    def find_specific_item(self, item_name, restaurant=None):
        """
        Find a specific menu item by name, with partial matching
        
        Args:
            item_name (str): The name of the item to find
            restaurant (str, optional): Limit search to a specific restaurant
        
        Returns:
            dict: The menu item if found, None otherwise
        """
        if not item_name:
            return None
        
        # Define which items to search through
        items_to_search = (
            self.restaurant_to_items.get(restaurant, []) 
            if restaurant 
            else self.kb.get('menu_items', [])
        )
            
        # First try exact match
        for item in items_to_search:
            if item['name'].lower() == item_name.lower():
                return item
        
        # Try partial match
        matched_items = []
        for item in items_to_search:
            if item_name.lower() in item['name'].lower():
                # Calculate match quality score - lower is better
                # (word length difference penalizes very long names)
                score = abs(len(item['name']) - len(item_name))
                matched_items.append((item, score))
                
        # Try match by keywords
        if not matched_items:
            keywords = item_name.lower().split()
            for item in items_to_search:
                item_name_lower = item['name'].lower()
                # Count how many keywords match
                matching_keywords = sum(1 for kw in keywords if kw in item_name_lower)
                if matching_keywords >= 2 or (len(keywords) == 1 and matching_keywords == 1):
                    matched_items.append((item, 1000 - matching_keywords))  # Lower score = better match
        
        # Sort by score (lower is better)
        matched_items.sort(key=lambda x: x[1])
        return matched_items[0][0] if matched_items else None
    
    def get_items_by_category(self, category, restaurant=None):
        """
        Get items belonging to a specific category
        
        Args:
            category (str): Category to search for
            restaurant (str, optional): Limit search to a specific restaurant
        """
        # First find the matching category (case-insensitive)
        for cat in self.category_to_items.keys():
            if cat.lower() == category.lower():
                items = self.category_to_items[cat]
                
                # Filter by restaurant if specified
                if restaurant:
                    return [item for item in items if item.get('restaurant') == restaurant]
                return items
        
        # If no exact match, try partial matching
        matched_categories = []
        category_lower = category.lower()
        
        for cat in self.category_to_items.keys():
            if category_lower in cat.lower():
                matched_categories.append(cat)
                
        # If still no matches, see if we can find it using embeddings
        if not matched_categories and self.use_semantic_search:
            try:
                # Get embeddings for the category and all category names
                cat_emb = self.sentence_model.encode(category)
                category_embs = {cat: self.sentence_model.encode(cat) for cat in self.category_to_items.keys()}
                
                # Find the closest matching category
                max_sim = -1
                best_cat = None
                
                for cat, emb in category_embs.items():
                    sim = np.dot(cat_emb, emb) / (np.linalg.norm(cat_emb) * np.linalg.norm(emb))
                    if sim > max_sim and sim > 0.5:  # Threshold to avoid bad matches
                        max_sim = sim
                        best_cat = cat
                
                if best_cat:
                    matched_categories.append(best_cat)
            except Exception as e:
                print(f"Error in semantic category matching: {e}")
        
        # Return items from all matched categories
        result_items = []
        for cat in matched_categories:
            items = self.category_to_items[cat]
            # Filter by restaurant if specified
            if restaurant:
                result_items.extend([item for item in items if item.get('restaurant') == restaurant])
            else:
                result_items.extend(items)
        
        # Remove duplicates
        seen_names = set()
        unique_items = []
        for item in result_items:
            if item['name'] not in seen_names:
                seen_names.add(item['name'])
                unique_items.append(item)
        
        return unique_items
    
    def generate_response(self, query):
        """Generate a response based on the query and retrieved information"""
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Detect if the user is switching restaurant focus
        new_focus = self._check_restaurant_switch(query)
        if new_focus:
            self.current_restaurant_focus = new_focus
            response = f"I'll focus on {new_focus} now. How can I help you with their menu?"
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # Check for comparative queries between restaurants
        if self._is_restaurant_comparison(query):
            response = self._handle_restaurant_comparison(query)
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # Check for repeated queries and respond more directly
        if len(self.conversation_history) >= 3:
            last_user_query = self.conversation_history[-1]["content"]
            previous_user_query = next((msg["content"] for msg in reversed(self.conversation_history[:-1]) 
                                        if msg["role"] == "user"), None)
            
            if previous_user_query and previous_user_query.lower() == last_user_query.lower():
                # This is a repeated question, be more direct
                direct_answer = self.generate_direct_answer(query)
                if direct_answer:
                    self.conversation_history.append({"role": "assistant", "content": direct_answer})
                    return direct_answer
        
        # 1. Identify what kind of question is being asked
        query_type = self.identify_query_type(query)
        
        # 2. Process based on query type
        if query_type == "menu_item_details":
            response = self.handle_item_details_query(query)
        elif query_type == "price_inquiry":
            response = self.handle_price_inquiry(query)
        elif query_type == "dietary_restriction":
            response = self.handle_dietary_restriction(query)
        elif query_type == "comparison":
            response = self.handle_comparison(query)
        elif query_type == "category_inquiry":
            response = self.handle_category_inquiry(query)
        elif query_type == "restaurant_inquiry":
            response = self.handle_restaurant_inquiry(query)
        else:
            # For general queries, try using the LLM
            response = self.handle_general_query(query)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Trim conversation history if it gets too long
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-self.max_history_length * 2:]
        
        return response
    
    def _check_restaurant_switch(self, query):
        """Check if user is trying to switch restaurant focus"""
        query_lower = query.lower()
        
        # Look for phrases indicating restaurant switching
        switch_patterns = [
            r"switch to (.+)",
            r"focus on (.+)",
            r"tell me about (.+) restaurant",
            r"let's talk about (.+)",
            r"change to (.+)",
            r"what about (.+)"
        ]
        
        for pattern in switch_patterns:
            match = re.search(pattern, query_lower)
            if match:
                restaurant_mention = match.group(1).strip()
                
                # Check against restaurant names
                for restaurant in self.restaurant_names:
                    if restaurant.lower() in restaurant_mention:
                        return restaurant
        
        # Direct restaurant mentions
        for restaurant in self.restaurant_names:
            if restaurant.lower() in query_lower and len(query_lower) < len(restaurant) + 15:
                # If the query is mostly just the restaurant name
                return restaurant
        
        return None
    
    def _is_restaurant_comparison(self, query):
        """Check if query is asking for comparison between restaurants"""
        query_lower = query.lower()
        
        # Look for comparison markers
        comparison_markers = ['vs', 'versus', 'compare', 'difference between', 'better']
        if any(marker in query_lower for marker in comparison_markers):
            # Check if at least two restaurant names are mentioned
            mentioned_restaurants = [r for r in self.restaurant_names if r.lower() in query_lower]
            return len(mentioned_restaurants) >= 2
        
        return False
    
    def _handle_restaurant_comparison(self, query):
        """Handle comparison between restaurants"""
        query_lower = query.lower()
        
        # Find mentioned restaurants
        mentioned_restaurants = [r for r in self.restaurant_names if r.lower() in query_lower]
        
        if len(mentioned_restaurants) < 2:
            return "To compare restaurants, please mention at least two restaurant names."
        
        # Limit to two restaurants for focused comparison
        restaurants_to_compare = mentioned_restaurants[:2]
        
        # Build comparison response
        response = f"# Comparing {restaurants_to_compare[0]} vs {restaurants_to_compare[1]}\n\n"
        
        # Compare price ranges
        r1_prices = self.price_by_restaurant.get(restaurants_to_compare[0], {})
        r2_prices = self.price_by_restaurant.get(restaurants_to_compare[1], {})
        
        if r1_prices and r2_prices:
            response += "## 💰 Price Range\n"
            response += f"- **{restaurants_to_compare[0]}**: ₹{r1_prices.get('min', 0)} to ₹{r1_prices.get('max', 0)} (avg: ₹{r1_prices.get('avg', 0):.0f})\n"
            response += f"- **{restaurants_to_compare[1]}**: ₹{r2_prices.get('min', 0)} to ₹{r2_prices.get('max', 0)} (avg: ₹{r2_prices.get('avg', 0):.0f})\n\n"
        
        # Compare menu size
        r1_items = self.restaurant_to_items.get(restaurants_to_compare[0], [])
        r2_items = self.restaurant_to_items.get(restaurants_to_compare[1], [])
        
        response += "## 🍽️ Menu Size\n"
        response += f"- **{restaurants_to_compare[0]}**: {len(r1_items)} items\n"
        response += f"- **{restaurants_to_compare[1]}**: {len(r2_items)} items\n\n"
        
        # Compare veg/non-veg options
        r1_veg = sum(1 for item in r1_items if item.get('veg_status') == 'Veg')
        r1_nonveg = sum(1 for item in r1_items if item.get('veg_status') == 'Non-Veg')
        
        r2_veg = sum(1 for item in r2_items if item.get('veg_status') == 'Veg')
        r2_nonveg = sum(1 for item in r2_items if item.get('veg_status') == 'Non-Veg')
        
        response += "## 🥬 Dietary Options\n"
        response += f"- **{restaurants_to_compare[0]}**: {r1_veg} vegetarian, {r1_nonveg} non-vegetarian\n"
        response += f"- **{restaurants_to_compare[1]}**: {r2_veg} vegetarian, {r2_nonveg} non-vegetarian\n\n"
        
        # Compare unique categories
        r1_categories = set()
        for item in r1_items:
            r1_categories.update(item.get('categories', []))
        
        r2_categories = set()
        for item in r2_items:
            r2_categories.update(item.get('categories', []))
        
        # Find unique categories for each restaurant
        r1_unique = r1_categories - r2_categories
        r2_unique = r2_categories - r1_categories
        
        if r1_unique or r2_unique:
            response += "## 🍕 Unique Categories\n"
            if r1_unique:
                response += f"- **{restaurants_to_compare[0]} unique categories**: {', '.join(sorted(r1_unique)[:5])}"
                if len(r1_unique) > 5:
                    response += f" and {len(r1_unique) - 5} more"
                response += "\n"
            
            if r2_unique:
                response += f"- **{restaurants_to_compare[1]} unique categories**: {', '.join(sorted(r2_unique)[:5])}"
                if len(r2_unique) > 5:
                    response += f" and {len(r2_unique) - 5} more"
                response += "\n"
            
            response += "\n"
        
        # Add sample popular items from each
        response += "## 🌟 Sample Items\n"
        
        # Get 2-3 popular items from each restaurant to highlight
        r1_sample = r1_items[:3] if r1_items else []
        r2_sample = r2_items[:3] if r2_items else []
        
        if r1_sample:
            response += f"**{restaurants_to_compare[0]} sample items**:\n"
            for item in r1_sample:
                price_str = f" (₹{item.get('regular_price')})" if 'regular_price' in item else ""
                response += f"- {item['name']}{price_str}\n"
        
        if r2_sample:
            response += f"\n**{restaurants_to_compare[1]} sample items**:\n"
            for item in r2_sample:
                price_str = f" (₹{item.get('regular_price')})" if 'regular_price' in item else ""
                response += f"- {item['name']}{price_str}\n"
        
        # Add a helpful note at the end
        response += "\nWould you like more specific information about either of these restaurants?"
        
        return response
    
    def generate_direct_answer(self, query):
        """Generate a very direct answer for repeated questions"""
        query_lower = query.lower()
        
        # Extract potential restaurant name from query
        restaurant_name = self._extract_restaurant_name(query_lower)
        
        # Extract potential item name from query
        item_name = self.extract_item_name(query)
        
        if "price" in query_lower or "cost" in query_lower or "how much" in query_lower:
            if item_name:
                item = self.find_specific_item(item_name, restaurant_name)
                if item and 'regular_price' in item:
                    restaurant_prefix = f"At {item['restaurant']}, " if item.get('restaurant') else ""
                    return f"{restaurant_prefix}the price of {item['name']} is ₹{item['regular_price']}."
        
        elif "what's in" in query_lower or "ingredients" in query_lower or "what is in" in query_lower:
            if item_name:
                item = self.find_specific_item(item_name, restaurant_name)
                if item and 'description' in item and item['description']:
                    restaurant_prefix = f"At {item['restaurant']}, " if item.get('restaurant') else ""
                    return f"{restaurant_prefix}{item['name']} contains: {item['description']}"
        
        # For category inquiries like "Tell me about desserts"
        for category in self.category_to_items.keys():
            if category.lower() in query_lower:
                items = self.get_items_by_category(category, restaurant_name)
                if items:
                    restaurant_text = f" at {restaurant_name}" if restaurant_name else ""
                    response = f"Here are some {category} options{restaurant_text}:\n"
                    
                    # Group items by restaurant if no specific restaurant was mentioned
                    if not restaurant_name:
                        items_by_restaurant = {}
                        for item in items[:10]:  # Limit to 10 total items
                            rest = item.get('restaurant', 'Other')
                            if rest not in items_by_restaurant:
                                items_by_restaurant[rest] = []
                            items_by_restaurant[rest].append(item)
                        
                        # Show items organized by restaurant
                        for rest, rest_items in items_by_restaurant.items():
                            response += f"\n**{rest}**:\n"
                            for item in rest_items[:3]:  # Show up to 3 items per restaurant
                                price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                                response += f"- {item['name']}{price_str}\n"
                    else:
                        # Just show items from the specific restaurant
                        for item in items[:7]:
                            price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                            response += f"- {item['name']}{price_str}\n"
                        
                        if len(items) > 7:
                            response += f"...and {len(items) - 7} more {category} items."
                    
                    return response
        
        return None
    
    def handle_item_details_query(self, query):
        """Handle queries about specific menu items"""
        # First, try to extract the restaurant name from the query
        restaurant_name = self._extract_restaurant_name(query)
        
        # Then, try to extract the item name from the query
        item_name = self.extract_item_name(query)
        
        if item_name:
            # If restaurant is explicitly mentioned or we have a focus, use that
            search_restaurant = restaurant_name or self.current_restaurant_focus
            
            # Find the item
            item = self.find_specific_item(item_name, search_restaurant)
            
            if item:
                return self.format_item_details(item)
            else:
                # If no item found in the specific restaurant, try searching all restaurants
                if search_restaurant:
                    item = self.find_specific_item(item_name)
                    if item:
                        return f"I couldn't find '{item_name}' at {search_restaurant}, but I found it at {item['restaurant']}:\n\n{self.format_item_details(item)}"
                
                # Try to find similar items
                query_for_search = f"{item_name}"
                if search_restaurant:
                    query_for_search += f" {search_restaurant}"
                    
                retrieved_info = self.retrieve(query_for_search, restaurant_filter=search_restaurant)
                
                if retrieved_info:
                    restaurant_scope = f" at {search_restaurant}" if search_restaurant else ""
                    response = f"I couldn't find an exact match for '{item_name}'{restaurant_scope}, but here are some similar items:\n\n"
                    
                    # Group by restaurant if no specific restaurant
                    if not search_restaurant:
                        items_by_restaurant = {}
                        for it, _ in retrieved_info:
                            rest = it.get('restaurant', 'Other')
                            if rest not in items_by_restaurant:
                                items_by_restaurant[rest] = []
                            items_by_restaurant[rest].append(it)
                        
                        for rest, items in items_by_restaurant.items():
                            response += f"**{rest}**:\n"
                            for it in items[:2]:
                                price_str = f" (₹{it['regular_price']})" if 'regular_price' in it else ""
                                response += f"- {it['name']}{price_str}\n"
                    else:
                        # Just show items from that restaurant
                        for it, _ in retrieved_info:
                            price_str = f" (₹{it['regular_price']})" if 'regular_price' in it else ""
                            response += f"- {it['name']}{price_str}\n"
                    
                    return response
                else:
                    scope = f" at {search_restaurant}" if search_restaurant else ""
                    return f"I couldn't find any menu item similar to '{item_name}'{scope}. Please try another item name."
        else:
            # General menu inquiry, show popular items
            if self.current_restaurant_focus:
                # Show items from the current restaurant focus
                restaurant_items = self.restaurant_to_items.get(self.current_restaurant_focus, [])
                
                if restaurant_items:
                    # Group items by category
                    items_by_category = {}
                    for item in restaurant_items:
                        for category in item.get('categories', ['other']):
                            if category not in items_by_category:
                                items_by_category[category] = []
                            items_by_category[category].append(item)
                    
                    # Pick a few categories to show
                    selected_categories = list(items_by_category.keys())[:4]
                    
                    response = f"Here are some popular items from {self.current_restaurant_focus}:\n\n"
                    
                    for category in selected_categories:
                        items = items_by_category[category][:3]
                        if items:
                            response += f"**{category.title()}**:\n"
                            for item in items:
                                price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                                response += f"- {item['name']}{price_str}\n"
                            response += "\n"
                    
                    response += "What would you like to know more about?"
                    return response
                else:
                    return f"I don't have any menu items for {self.current_restaurant_focus}. Would you like to explore another restaurant?"
            else:
                # No restaurant focus yet, suggest a few restaurants
                response = "Here are some restaurants I can tell you about:\n\n"
                
                for restaurant in self.restaurant_names[:5]:
                    num_items = len(self.restaurant_to_items.get(restaurant, []))
                    price_info = self.price_by_restaurant.get(restaurant, {})
                    price_range = f"₹{price_info.get('min', 0)}-{price_info.get('max', 0)}" if price_info else "N/A"
                    
                    response += f"- **{restaurant}** ({num_items} items, Price range: {price_range})\n"
                
                response += "\nYou can ask about a specific restaurant, or say something like 'Tell me about Burger King menu'"
                return response
    
    def handle_price_inquiry(self, query):
        """Handle price-related queries"""
        query_lower = query.lower()
        
        # Extract restaurant name if mentioned
        restaurant_name = self._extract_restaurant_name(query)
        search_restaurant = restaurant_name or self.current_restaurant_focus
        
        # Extract item name if mentioned
        item_name = self.extract_item_name(query)
        
        if item_name:
            # Query is about a specific item's price
            item = self.find_specific_item(item_name, search_restaurant)
            
            if item:
                restaurant_prefix = f"At {item['restaurant']}, " if item['restaurant'] and not search_restaurant else ""
                
                if 'regular_price' in item:
                    response = f"{restaurant_prefix}the price of {item['name']} is ₹{item['regular_price']}."
                    if 'offer_price' in item and item['offer_price']:
                        response += f" It's currently available at a special offer price of ₹{item['offer_price']}."
                    return response
                else:
                    return f"I couldn't find the price information for {item['name']}."
            else:
                # Try to find the item in any restaurant if not found in specified restaurant
                if search_restaurant:
                    item = self.find_specific_item(item_name)
                    if item:
                        return f"I couldn't find {item_name} at {search_restaurant}, but at {item['restaurant']}, it costs ₹{item.get('regular_price', 'N/A')}."
                
                scope = f" at {search_restaurant}" if search_restaurant else ""
                return f"I couldn't find a menu item called '{item_name}'{scope}."
        else:
            # Generic price range inquiry
            price_range = self.extract_price_range(query)
            if price_range:
                min_price, max_price = price_range
                matching_items = self.find_items_by_criteria({
                    'price_min': min_price,
                    'price_max': max_price,
                    'restaurant': search_restaurant
                })
                
                if matching_items:
                    restaurant_scope = f" at {search_restaurant}" if search_restaurant else ""
                    response = f"Here are items within your price range (₹{min_price}-₹{max_price}){restaurant_scope}:\n\n"
                    
                    # Group by restaurant if no specific restaurant
                    if not search_restaurant:
                        items_by_restaurant = {}
                        for item in sorted(matching_items, key=lambda x: x.get('regular_price', 0))[:20]:
                            rest = item.get('restaurant', 'Other')
                            if rest not in items_by_restaurant:
                                items_by_restaurant[rest] = []
                            items_by_restaurant[rest].append(item)
                        
                        for rest, items in items_by_restaurant.items():
                            response += f"**{rest}**:\n"
                            for item in items[:5]:
                                response += f"- {item['name']} (₹{item.get('regular_price', 'N/A')})\n"
                            
                            if len(items) > 5:
                                response += f"...and {len(items) - 5} more items.\n"
                            response += "\n"
                    else:
                        # Just show items from that restaurant
                        for item in sorted(matching_items, key=lambda x: x.get('regular_price', 0))[:12]:
                            response += f"- {item['name']} (₹{item.get('regular_price', 'N/A')})\n"
                        
                        if len(matching_items) > 12:
                            response += f"\n...and {len(matching_items) - 12} more items."
                    
                    return response
                else:
                    scope = f" at {search_restaurant}" if search_restaurant else ""
                    return f"I couldn't find any items in the price range ₹{min_price}-₹{max_price}{scope}."
            else:
                # General price information
                if search_restaurant:
                    price_info = self.price_by_restaurant.get(search_restaurant, {})
                    
                    if price_info:
                        response = f"At {search_restaurant}, menu items range in price from ₹{price_info.get('min', 0)} to ₹{price_info.get('max', 0)}.\n\n"
                        
                        # Add budget options
                        budget_items = self.find_items_by_criteria({
                            'price_max': 100,
                            'restaurant': search_restaurant
                        })
                        
                        if budget_items:
                            response += "**Budget options (under ₹100):**\n"
                            for item in sorted(budget_items, key=lambda x: x.get('regular_price', 0))[:5]:
                                response += f"- {item['name']} (₹{item.get('regular_price', 'N/A')})\n"
                        
                        # Add mid-range options
                        mid_range_items = self.find_items_by_criteria({
                            'price_min': 100,
                            'price_max': 200,
                            'restaurant': search_restaurant
                        })
                        
                        if mid_range_items:
                            response += "\n**Mid-range options (₹100-₹200):**\n"
                            for item in sorted(mid_range_items, key=lambda x: x.get('regular_price', 0))[:5]:
                                response += f"- {item['name']} (₹{item.get('regular_price', 'N/A')})\n"
                        
                        return response
                    else:
                        return f"I don't have price information for {search_restaurant}."
                else:
                    # Compare price ranges across restaurants
                    response = "**Price ranges across restaurants:**\n\n"
                    
                    for restaurant, price_info in sorted(self.price_by_restaurant.items(), 
                                                        key=lambda x: x[1].get('min', 0)):
                        response += f"- **{restaurant}**: ₹{price_info.get('min', 0)} to ₹{price_info.get('max', 0)}"
                        response += f" (avg: ₹{price_info.get('avg', 0):.0f})\n"
                    
                    response += "\nYou can ask about specific price ranges like 'Show items under ₹200 at Burger King'"
                    return response
    
    def handle_dietary_restriction(self, query):
        """Handle queries related to dietary restrictions"""
        query_lower = query.lower()
        
        # Extract restaurant name if mentioned
        restaurant_name = self._extract_restaurant_name(query)
        search_restaurant = restaurant_name or self.current_restaurant_focus
        
        # Handle vegetarian queries
        if ("vegetarian" in query_lower or "veg" in query_lower) and "non" not in query_lower:
            # Find vegetarian items, filtered by restaurant if specified
            if search_restaurant:
                veg_items = [item for item in self.restaurant_to_items.get(search_restaurant, [])
                           if item.get('veg_status') == "Veg"]
            else:
                # Across all restaurants
                veg_items = [item for item in self.kb['menu_items'] 
                           if item.get('veg_status') == "Veg"]
            
            if veg_items:
                restaurant_scope = f" at {search_restaurant}" if search_restaurant else ""
                response = f"We have {len(veg_items)} vegetarian items{restaurant_scope}. Here are some popular ones:\n\n"
                
                # Group by restaurant if no specific restaurant
                if not search_restaurant:
                    items_by_restaurant = {}
                    for item in veg_items[:20]:
                        rest = item.get('restaurant', 'Other')
                        if rest not in items_by_restaurant:
                            items_by_restaurant[rest] = []
                        items_by_restaurant[rest].append(item)
                    
                    for rest, items in items_by_restaurant.items():
                        response += f"**{rest}**:\n"
                        for item in items[:3]:
                            price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                            response += f"- {item['name']}{price_str}\n"
                        response += "\n"
                else:
                    # Group by category for a single restaurant
                    veg_by_category = {}
                    for item in veg_items:
                        for category in item.get('categories', ['other']):
                            if category not in veg_by_category:
                                veg_by_category[category] = []
                            veg_by_category[category].append(item)
                    
                    # Show up to 3 categories
                    for i, (category, items) in enumerate(veg_by_category.items()):
                        if i >= 3:
                            break
                            
                        response += f"**{category.title()}:**\n"
                        for item in items[:3]:
                            price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                            response += f"- {item['name']}{price_str}\n"
                        response += "\n"
                
                return response
            else:
                scope = f" at {search_restaurant}" if search_restaurant else ""
                return f"I couldn't find any vegetarian items{scope}."
        
        # Handle non-vegetarian queries
        elif "non-vegetarian" in query_lower or "non veg" in query_lower or ("chicken" in query_lower and "menu" in query_lower):
            # Find non-vegetarian items, filtered by restaurant if specified
            if search_restaurant:
                non_veg_items = [item for item in self.restaurant_to_items.get(search_restaurant, [])
                               if item.get('veg_status') == "Non-Veg"]
            else:
                # Across all restaurants
                non_veg_items = [item for item in self.kb['menu_items'] 
                               if item.get('veg_status') == "Non-Veg"]
            
            if non_veg_items:
                restaurant_scope = f" at {search_restaurant}" if search_restaurant else ""
                response = f"We have {len(non_veg_items)} non-vegetarian items{restaurant_scope}. Here are some popular ones:\n\n"
                
                # Group by restaurant if no specific restaurant
                if not search_restaurant:
                    items_by_restaurant = {}
                    for item in non_veg_items[:20]:
                        rest = item.get('restaurant', 'Other')
                        if rest not in items_by_restaurant:
                            items_by_restaurant[rest] = []
                        items_by_restaurant[rest].append(item)
                    
                    for rest, items in items_by_restaurant.items():
                        response += f"**{rest}**:\n"
                        for item in items[:3]:
                            price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                            response += f"- {item['name']}{price_str}\n"
                        response += "\n"
                else:
                    # Group by category for a single restaurant
                    non_veg_by_category = {}
                    for item in non_veg_items:
                        for category in item.get('categories', ['other']):
                            if category not in non_veg_by_category:
                                non_veg_by_category[category] = []
                            non_veg_by_category[category].append(item)
                    
                    # Show up to 3 categories
                    for i, (category, items) in enumerate(non_veg_by_category.items()):
                        if i >= 3:
                            break
                            
                        response += f"**{category.title()}:**\n"
                        for item in items[:3]:
                            price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                            response += f"- {item['name']}{price_str}\n"
                        response += "\n"
                
                return response
            else:
                scope = f" at {search_restaurant}" if search_restaurant else ""
                return f"I couldn't find any non-vegetarian items{scope}."
        
        # Allergen information
        elif any(allergen in query_lower for allergen in ["gluten", "dairy", "milk", "soy", "nut", "sesame"]):
            allergen_maps = {
                "dairy": "milk",
                "gluten": "gluten",
                "soy": "soybean",
                "nut": "nuts",
                "sesame": "sesame"
            }
            
            allergens_mentioned = []
            for key, value in allergen_maps.items():
                if key in query_lower:
                    allergens_mentioned.append(value)
            
            if allergens_mentioned:
                restaurant_scope = f" at {search_restaurant}" if search_restaurant else ""
                response = f"For customers concerned about {', '.join(allergens_mentioned)}{restaurant_scope}, please note:\n\n"
                
                # Find items that contain the mentioned allergens
                allergen_items = []
                
                if search_restaurant:
                    items_to_check = self.restaurant_to_items.get(search_restaurant, [])
                else:
                    items_to_check = self.kb['menu_items']
                
                for item in items_to_check:
                    if any(allergen.lower() in ' '.join(item.get('allergens', [])).lower() 
                          for allergen in allergens_mentioned):
                        allergen_items.append(item)
                
                if allergen_items:
                    response += "These items contain the allergens you mentioned:\n\n"
                    
                    # Group by restaurant if no specific restaurant
                    if not search_restaurant:
                        items_by_restaurant = {}
                        for item in allergen_items[:15]:
                            rest = item.get('restaurant', 'Other')
                            if rest not in items_by_restaurant:
                                items_by_restaurant[rest] = []
                            items_by_restaurant[rest].append(item)
                        
                        for rest, items in items_by_restaurant.items():
                            response += f"**{rest}**:\n"
                            for item in items[:3]:
                                allergens_str = f" (contains {', '.join(item.get('allergens', []))})"
                                response += f"- **{item['name']}**{allergens_str}\n"
                            response += "\n"
                    else:
                        for item in allergen_items[:7]:
                            allergens_str = f" (contains {', '.join(item.get('allergens', []))})"
                            response += f"- **{item['name']}**{allergens_str}\n"
                    
                    response += "\nI recommend asking the restaurant staff for specific allergen information when ordering."
                    return response
                else:
                    return f"I couldn't find any items containing {', '.join(allergens_mentioned)}{restaurant_scope}."
        
        scope = f" at {search_restaurant}" if search_restaurant else ""
        return f"We offer a variety of dietary options{scope} including vegetarian and non-vegetarian choices. Do you have a specific dietary restriction or preference you'd like to know about?"
    
    def handle_comparison(self, query):
        """Handle comparison queries between menu items"""
        # Extract restaurant name if mentioned
        restaurant_name = self._extract_restaurant_name(query)
        search_restaurant = restaurant_name or self.current_restaurant_focus
        
        items_to_compare = self.extract_comparison_items(query)
        
        if len(items_to_compare) >= 2:
            item1_name, item2_name = items_to_compare[:2]
            
            # Find the items in the menu, prioritizing the mentioned/focused restaurant
            item1 = self.find_specific_item(item1_name, search_restaurant)
            item2 = self.find_specific_item(item2_name, search_restaurant)
            
            # If items not found in the specific restaurant, try finding them in any restaurant
            if not item1 and search_restaurant:
                item1 = self.find_specific_item(item1_name)
            if not item2 and search_restaurant:
                item2 = self.find_specific_item(item2_name)
            
            if item1 and item2:
                response = f"# Comparing {item1['name']} and {item2['name']}\n\n"
                
                # Add restaurant info if they're from different restaurants
                if item1.get('restaurant') != item2.get('restaurant'):
                    response += f"**Restaurant**: {item1['name']} is from {item1.get('restaurant', 'Unknown')}, "
                    response += f"{item2['name']} is from {item2.get('restaurant', 'Unknown')}\n\n"
                
                # Compare prices
                price1 = item1.get('regular_price', 'N/A')
                price2 = item2.get('regular_price', 'N/A')
                response += f"**Price**: {item1['name']} (₹{price1}) vs {item2['name']} (₹{price2})\n"
                
                # Compare veg/non-veg status
                response += f"**Type**: {item1['name']} is {item1['veg_status']} | {item2['name']} is {item2['veg_status']}\n"
                
                # Compare categories
                cats1 = item1.get('categories', [])
                cats2 = item2.get('categories', [])
                if cats1 and cats2:
                    response += f"**Categories**: {', '.join(cats1[:3])} vs {', '.join(cats2[:3])}\n"
                
                # Compare calories if available
                cal1 = item1.get('calories', item1.get('nutrition', {}).get('calories', 'N/A'))
                cal2 = item2.get('calories', item2.get('nutrition', {}).get('calories', 'N/A'))
                if cal1 != 'N/A' or cal2 != 'N/A':
                    response += f"**Calories**: {item1['name']} ({cal1} kcal) vs {item2['name']} ({cal2} kcal)\n"
                
                # Compare descriptions if available
                if item1.get('description') and item2.get('description'):
                    response += f"\n**{item1['name']} contains**: {item1['description'][:100]}...\n"
                    response += f"**{item2['name']} contains**: {item2['description'][:100]}...\n"
                
                # Recommendation based on comparison
                response += "\n**Which might you prefer?** "
                if item1['veg_status'] != item2['veg_status']:
                    response += f"If you're looking for a vegetarian option, {item1['name'] if item1['veg_status'] == 'Veg' else item2['name']} is the way to go."
                elif price1 != price2 and price1 != 'N/A' and price2 != 'N/A':
                    cheaper = item1['name'] if float(price1) < float(price2) else item2['name']
                    response += f"{cheaper} is more budget-friendly."
                else:
                    response += "Both are great choices with their own unique flavors."
                
                return response
            else:
                missing_items = []
                if not item1:
                    missing_items.append(item1_name)
                if not item2:
                    missing_items.append(item2_name)
                
                scope = f" at {search_restaurant}" if search_restaurant else ""
                return f"I couldn't find the following item(s){scope}: {', '.join(missing_items)}"
        else:
            return "To compare menu items, please mention two specific items you'd like to compare."
    
    def handle_category_inquiry(self, query):
        """Handle category-specific inquiries"""
        query_lower = query.lower()
        
        # Extract restaurant name if mentioned
        restaurant_name = self._extract_restaurant_name(query)
        search_restaurant = restaurant_name or self.current_restaurant_focus
        
        # Extract category from the query
        category_names = list(self.category_to_items.keys())
        mentioned_categories = []
        
        # Look for exact matches first
        for category in category_names:
            if category.lower() in query_lower:
                mentioned_categories.append(category)
                
        # If no exact matches, try partial matches
        if not mentioned_categories:
            for category in category_names:
                for word in query_lower.split():
                    # Check if category contains the word or vice versa
                    if word in category.lower() or category.lower() in word:
                        if len(word) >= 4:  # To avoid matching short words like "the", "and", etc.
                            mentioned_categories.append(category)
                            break
        
        # De-duplicate and take the first category
        if mentioned_categories:
            category = mentioned_categories[0]
            items = self.get_items_by_category(category, search_restaurant)
            
            if items:
                restaurant_scope = f" at {search_restaurant}" if search_restaurant else ""
                response = f"# {category.title()} Menu{restaurant_scope}\n\n"
                
                # If no restaurant filter, group by restaurant
                if not search_restaurant:
                    items_by_restaurant = {}
                    for item in items:
                        rest = item.get('restaurant', 'Other')
                        if rest not in items_by_restaurant:
                            items_by_restaurant[rest] = []
                        items_by_restaurant[rest].append(item)
                    
                    for rest, rest_items in items_by_restaurant.items():
                        response += f"## {rest}\n"
                        
                        # Group by veg/non-veg
                        veg_items = [item for item in rest_items if item.get('veg_status') == "Veg"]
                        non_veg_items = [item for item in rest_items if item.get('veg_status') == "Non-Veg"]
                        
                        if veg_items:
                            response += "**Vegetarian Options:**\n"
                            for item in sorted(veg_items, key=lambda x: x.get('regular_price', 0))[:3]:
                                price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                                response += f"- {item['name']}{price_str}\n"
                            if len(veg_items) > 3:
                                response += f"...and {len(veg_items) - 3} more veg items.\n"
                        
                        if non_veg_items:
                            response += "\n**Non-Vegetarian Options:**\n"
                            for item in sorted(non_veg_items, key=lambda x: x.get('regular_price', 0))[:3]:
                                price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                                response += f"- {item['name']}{price_str}\n"
                            if len(non_veg_items) > 3:
                                response += f"...and {len(non_veg_items) - 3} more non-veg items.\n"
                        
                        response += "\n"
                else:
                    # Group by veg/non-veg
                    veg_items = [item for item in items if item.get('veg_status') == "Veg"]
                    non_veg_items = [item for item in items if item.get('veg_status') == "Non-Veg"]
                    
                    if veg_items:
                        response += "## 🌱 Vegetarian Options\n"
                        for item in sorted(veg_items, key=lambda x: x.get('regular_price', 0))[:5]:
                            price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                            response += f"- {item['name']}{price_str}\n"
                        if len(veg_items) > 5:
                            response += f"...and {len(veg_items) - 5} more vegetarian {category} items.\n"
                        response += "\n"
                    
                    if non_veg_items:
                        response += "## 🍗 Non-Vegetarian Options\n"
                        for item in sorted(non_veg_items, key=lambda x: x.get('regular_price', 0))[:5]:
                            price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                            response += f"- {item['name']}{price_str}\n"
                        if len(non_veg_items) > 5:
                            response += f"...and {len(non_veg_items) - 5} more non-vegetarian {category} items.\n"
                        response += "\n"
                
                # Add a brief description of the category if available
                category_descriptions = {
                    "whopper": "Signature flame-grilled burgers with premium ingredients.",
                    "burger": "Classic burgers with various patty options and toppings.",
                    "wrap": "Soft tortilla wraps filled with flavorful ingredients.",
                    "pizza": "Italian-style flatbreads topped with sauce, cheese, and various toppings.",
                    "sides": "Perfect accompaniments to complement your main meal.",
                    "beverages": "Refreshing drinks to complement your meal.",
                    "dessert": "Sweet treats to finish your meal.",
                    "meal": "Complete meal combinations at great value."
                }
                
                if category.lower() in category_descriptions:
                    response = f"**{category.title()}**: {category_descriptions[category.lower()]}\n\n" + response
                
                # Add price range for this category
                if items and any(item.get('regular_price') for item in items):
                    prices = [item['regular_price'] for item in items if 'regular_price' in item]
                    if prices:
                        min_price = min(prices)
                        max_price = max(prices)
                        response += f"\n**Price range**: ₹{min_price} - ₹{max_price}"
                
                return response
            else:
                scope = f" at {search_restaurant}" if search_restaurant else ""
                return f"I couldn't find any items in the {category} category{scope}."
        
        # If no specific category was found
        scope = f" at {search_restaurant}" if search_restaurant else ""
        
        

        # Get categories for the specific restaurant or all restaurants
        if search_restaurant:
            # Get categories for the specific restaurant
            restaurant_info = next((r for r in self.restaurants if r['name'] == search_restaurant), None)
            if restaurant_info:
                available_categories = restaurant_info['categories']
                response = f"Here are the menu categories at {search_restaurant}:\n\n"
                for cat in sorted(available_categories):
                    # Get item count for this category at this restaurant
                    cat_items = [item for item in self.restaurant_to_items.get(search_restaurant, [])
                              if cat in item.get('categories', [])]
                    response += f"- **{cat}** ({len(cat_items)} items)\n"
                
                response += "\nYou can ask about a specific category like 'Show me burgers at Burger King'"
                return response
            else:
                return f"I couldn't find information about {search_restaurant}."
        else:
            # Show popular categories across all restaurants
            response = "Here are popular food categories across our restaurants:\n\n"
            
            # Get most common categories
            category_counts = {}
            for cat in self.category_to_items:
                category_counts[cat] = len(self.category_to_items[cat])
            
            # Sort by popularity
            popular_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            
            for cat, count in popular_categories:
                # Get a list of restaurants that have this category
                restaurants_with_cat = set()
                for item in self.category_to_items[cat]:
                    if item.get('restaurant'):
                        restaurants_with_cat.add(item['restaurant'])
                
                rest_text = f" (available at {', '.join(sorted(restaurants_with_cat))})" if restaurants_with_cat else ""
                response += f"- **{cat}** ({count} items){rest_text}\n"
            
            response += "\nYou can ask about a specific category or specify a restaurant, like 'Show me burgers at Burger King'"
            return response
    
    def handle_restaurant_inquiry(self, query):
        """Handle restaurant-specific inquiries"""
        query_lower = query.lower()
        restaurant_name = self._extract_restaurant_name(query)
        
        if restaurant_name:
            # User is asking about a specific restaurant
            restaurant_info = next((r for r in self.restaurants if r['name'] == restaurant_name), None)
            items = self.restaurant_to_items.get(restaurant_name, [])
            
            if restaurant_info and items:
                response = f"# {restaurant_name}\n\n"
                
                # Add basic restaurant info
                price_info = self.price_by_restaurant.get(restaurant_name, {})
                if price_info:
                    response += f"**Price Range**: ₹{price_info.get('min', 0)} to ₹{price_info.get('max', 0)}\n"
                
                # Count veg/non-veg items
                veg_count = sum(1 for item in items if item.get('veg_status') == 'Veg')
                non_veg_count = sum(1 for item in items if item.get('veg_status') == 'Non-Veg')
                response += f"**Menu Size**: {len(items)} items ({veg_count} veg, {non_veg_count} non-veg)\n\n"
                
                # Get categories
                categories = restaurant_info.get('categories', [])
                if categories:
                    response += "**Categories**:\n"
                    for cat in sorted(categories)[:6]:
                        cat_items = [item for item in items if cat in item.get('categories', [])]
                        response += f"- {cat} ({len(cat_items)} items)\n"
                    
                    if len(categories) > 6:
                        response += f"...and {len(categories) - 6} more categories.\n"
                    
                    response += "\n"
                
                # Show popular items
                response += "**Popular Items**:\n"
                # Group by category to get different types
                items_by_category = {}
                for item in items:
                    for cat in item.get('categories', ['other']):
                        if cat not in items_by_category:
                            items_by_category[cat] = []
                        items_by_category[cat].append(item)
                
                # Show one item from each of top 3 categories
                shown_items = []
                for cat in list(items_by_category.keys())[:3]:
                    if items_by_category[cat]:
                        item = items_by_category[cat][0]
                        price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                        response += f"- {item['name']}{price_str} - {cat}\n"
                        shown_items.append(item['name'])
                
                # Add sample question suggestions at the end
                response += "\nYou can ask:\n"
                response += f"- 'Show me the {next(iter(categories), 'menu')} at {restaurant_name}'\n"
                if veg_count > 0:
                    response += f"- 'What vegetarian options does {restaurant_name} have?'\n"
                response += f"- 'What are the most popular items at {restaurant_name}?'\n"
                
                return response
            else:
                return f"I don't have information about {restaurant_name} or its menu items."
        else:
            # General restaurant inquiry, show all restaurants
            response = "# Available Restaurants\n\n"
            
            for restaurant in self.restaurants:
                name = restaurant['name']
                items = self.restaurant_to_items.get(name, [])
                price_info = self.price_by_restaurant.get(name, {})
                
                response += f"**{name}**\n"
                if price_info:
                    response += f"- Price range: ₹{price_info.get('min', 0)} to ₹{price_info.get('max', 0)}\n"
                
                veg_count = sum(1 for item in items if item.get('veg_status') == 'Veg')
                non_veg_count = sum(1 for item in items if item.get('veg_status') == 'Non-Veg')
                response += f"- Menu: {len(items)} items ({veg_count} veg, {non_veg_count} non-veg)\n"
                
                # Show a few categories
                categories = restaurant.get('categories', [])
                if categories:
                    cat_text = ", ".join(sorted(categories)[:3])
                    if len(categories) > 3:
                        cat_text += f" and {len(categories) - 3} more"
                    response += f"- Categories: {cat_text}\n"
                
                # Add a sample item if available
                if items:
                    sample_item = next((i for i in items if i.get('regular_price') is not None), items[0])
                    price_str = f" (₹{sample_item['regular_price']})" if 'regular_price' in sample_item else ""
                    response += f"- Sample item: {sample_item['name']}{price_str}\n"
                
                response += "\n"
            
            response += "You can ask about a specific restaurant, like 'Tell me about Burger King' or 'Show Dominos menu'."
            return response
    
    def handle_general_query(self, query):
        """Handle general queries or fall back to LLM"""
        retrieved_info = self.retrieve(query, top_k=4)
        
        if retrieved_info:
            # Format context for the LLM
            context = ""
            for item, score in retrieved_info:
                # Include restaurant name in the context
                restaurant = f" from {item['restaurant']}" if item.get('restaurant') else ""
                price_str = f"₹{item['regular_price']}" if 'regular_price' in item else "Not available"
                veg_status = f"{item.get('veg_status', 'Unknown')}"
                
                context += f"Item: {item['name']}{restaurant}\n"
                context += f"Price: {price_str}\n"
                context += f"Type: {veg_status}\n"
                
                if 'description' in item and item['description']:
                    desc = item['description']
                    if len(desc) > 150:
                        desc = desc[:150] + "..."
                    context += f"Description: {desc}\n"
                
                if 'categories' in item and item['categories']:
                    context += f"Categories: {', '.join(item['categories'])}\n"
                
                context += "\n"
            
            # Try to generate a response using the LLM
            if self.has_model:
                if self._lazy_load_model():
                    try:
                        prompt = f"""You are an AI food assistant for a restaurant chatbot.
User query: {query}
Relevant menu information:
{context}

Provide a helpful answer ONLY based on the menu information above. Be concise and focused.
If menu info doesn't answer the query, say you don't have that information.
Don't make up information not in the context. If you can't answer, say so.
""".strip()

                        response = self.generator(prompt)[0]['generated_text']
                        
                        # Clean up common LLM artifacts
                        response = re.sub(r'^(AI:|Assistant:|Chatbot:|Based on the menu information provided,)', '', response).strip()
                        
                        # Verify the response is substantive and relevant
                        if len(response) > 10 and not any(x in response.lower() for x in ["i don't have that information", "i can't answer"]):
                            return response
                    except Exception as e:
                        print(f"LLM error: {e}")
            
            # If the LLM fails or is unavailable, fall back to a rule-based approach
            if query.lower().startswith("what") or query.lower().startswith("which") or query.lower().startswith("how"):
                response = "Based on our menu, I found these items that might answer your query:\n\n"
            else:
                response = "Here are some relevant menu items:\n\n"
            
            # Group by restaurant
            items_by_restaurant = {}
            for item, _ in retrieved_info:
                rest = item.get('restaurant', 'Other')
                if rest not in items_by_restaurant:
                    items_by_restaurant[rest] = []
                items_by_restaurant[rest].append(item)
            
            # Format the response by restaurant
            for restaurant, items in items_by_restaurant.items():
                response += f"**{restaurant}**:\n"
                for item in items:
                    price_str = f" (₹{item['regular_price']})" if 'regular_price' in item else ""
                    veg_icon = "🌱" if item.get('veg_status') == "Veg" else "🍗" if item.get('veg_status') == "Non-Veg" else ""
                    response += f"- {veg_icon} {item['name']}{price_str}\n"
                    
                    # Add a brief description if available
                    if 'description' in item and item['description'] and len(response) < 800:  # Limit total length
                        desc = item['description']
                        if len(desc) > 100:
                            desc = desc[:100] + "..."
                        response += f"  {desc}\n"
                response += "\n"
        else:
            # No relevant items found, general response
            response = "I don't have specific information about that. You can ask me about:\n\n"
            response += "- Menu items, prices, and descriptions\n"
            response += "- Comparing different dishes or restaurants\n"
            response += "- Dietary options (vegetarian/non-vegetarian)\n"
            response += "- Restaurant-specific queries\n\n"
            response += "Would you like to see some popular menu items from our restaurants?"
        
        return response
    
    def _extract_restaurant_name(self, query):
        """Extract restaurant name from query"""
        query_lower = query.lower()
        
        # Check all restaurant names in order of length (to avoid partial matches)
        restaurants_by_length = sorted(self.restaurant_names, key=len, reverse=True)
        
        for restaurant in restaurants_by_length:
            if restaurant.lower() in query_lower:
                return restaurant
        
        return None
    
    def extract_item_name(self, query):
        """Extract a potential menu item name from user query"""
        query_lower = query.lower()
        
        # First try to extract the restaurant name to remove it from consideration
        restaurant_name = self._extract_restaurant_name(query)
        if restaurant_name:
            query_lower = query_lower.replace(restaurant_name.lower(), "")
        
        # Common words to ignore in food item extraction
        ignore_words = [
            "price", "cost", "how", "much", "what", "is", "are", "the", "of", "in",
            "about", "tell", "me", "from", "do", "you", "have", "show", "list", "all",
            "compare", "vs", "versus", "between", "and", "or", "with", "without", "any",
            "for", "menu", "restaurant", "food", "order", "get", "want", "please", "options",
            "vegetarian", "veg", "non-veg", "non", "there", "their", "difference"
        ]
        
        tokens = query_lower.split()
        filtered_tokens = [token for token in tokens if token not in ignore_words]
        
        # Join the remaining tokens to form a potential item name
        potential_item_name = " ".join(filtered_tokens).strip()
        
        # If it's too short or just common words, return None
        if len(potential_item_name) <= 2:
            return None
            
        return potential_item_name
    
    def extract_price_range(self, query):
        """Extract price range from user query"""
        query_lower = query.lower()
        
        # Look for "under X" pattern
        under_match = re.search(r'under (?:rs\.?|₹)?(\d+)', query_lower)
        if under_match:
            max_price = int(under_match.group(1))
            return (0, max_price)
        
        # Look for "below X" pattern
        below_match = re.search(r'below (?:rs\.?|₹)?(\d+)', query_lower)
        if below_match:
            max_price = int(below_match.group(1))
            return (0, max_price)
        
        # Look for "above X" pattern
        above_match = re.search(r'above (?:rs\.?|₹)?(\d+)', query_lower)
        if above_match:
            min_price = int(above_match.group(1))
            return (min_price, 99999)
        
        # Look for "between X and Y" pattern
        between_match = re.search(r'between (?:rs\.?|₹)?(\d+) and (?:rs\.?|₹)?(\d+)', query_lower)
        if between_match:
            price1 = int(between_match.group(1))
            price2 = int(between_match.group(2))
            return (min(price1, price2), max(price1, price2))
        
        # Look for X-Y pattern
        range_match = re.search(r'(?:rs\.?|₹)?(\d+)\s*-\s*(?:rs\.?|₹)?(\d+)', query_lower)
        if range_match:
            price1 = int(range_match.group(1))
            price2 = int(range_match.group(2))
            return (min(price1, price2), max(price1, price2))
        
        return None
    
    def extract_comparison_items(self, query):
        """Extract items for comparison from user query"""
        query_lower = query.lower()
        
        # Look for "compare X and Y" pattern
        compare_match = re.search(r'compare (.*?) and (.*?)($|\?|\.)', query_lower)
        if compare_match:
            item1 = compare_match.group(1).strip()
            item2 = compare_match.group(2).strip()
            return [item1, item2]
        
        # Look for "X vs Y" pattern
        vs_match = re.search(r'(.*?) (?:vs|versus) (.*?)($|\?|\.)', query_lower)
        if vs_match:
            item1 = vs_match.group(1).strip()
            item2 = vs_match.group(2).strip()
            return [item1, item2]
        
        # Look for "difference between X and Y" pattern
        diff_match = re.search(r'difference between (.*?) and (.*?)($|\?|\.)', query_lower)
        if diff_match:
            item1 = diff_match.group(1).strip()
            item2 = diff_match.group(2).strip()
            return [item1, item2]
        
        # Fall back to check for "and" - less reliable
        parts = query_lower.split(' and ')
        if len(parts) == 2:
            item1 = parts[0].split()[-1]  # last word before "and"
            item2 = parts[1].split()[0]   # first word after "and"
            if item1 and item2:
                return [item1, item2]
        
        return []
    
    def format_item_details(self, item):
        """Format details about a menu item"""
        name = item.get('name', 'Unknown')
        price = item.get('regular_price', 'N/A')
        description = item.get('description', 'No description available.')
        veg_status = item.get('veg_status', 'Unknown')
        restaurant = item.get('restaurant', 'Unknown')
        
        response = f"# {name}\n\n"
        response += f"**Restaurant**: {restaurant}\n"
        response += f"**Price**: ₹{price}\n"
        response += f"**Type**: {veg_status}\n"
        
        if 'categories' in item and item['categories']:
            response += f"**Categories**: {', '.join(item['categories'])}\n"
        
        if description:
            response += f"\n**Description**:\n{description}\n"
        
        # Additional info if available
        if 'allergens' in item and item['allergens']:
            response += f"\n**Allergens**: {', '.join(item['allergens'])}\n"
        
        if 'calories' in item and item['calories']:
            response += f"\n**Calories**: {item['calories']}\n"
        
        if 'nutrition' in item and item['nutrition']:
            response += "\n**Nutrition**:\n"
            for nutrient, value in item['nutrition'].items():
                response += f"- {nutrient.capitalize()}: {value}\n"
        
        return response
    
    def identify_query_type(self, query):
        """Identify the type of query being asked"""
        query_lower = query.lower()
        
        # Restaurant specific query
        if any(x in query_lower for x in ['restaurant', 'restaurants', 'places', 'outlets']) or any(restaurant.lower() in query_lower for restaurant in self.restaurant_names):
            return "restaurant_inquiry"
        
        # Price inquiry
        if any(x in query_lower for x in ['price', 'cost', 'how much', 'cheap', 'expensive', 'budget', 'affordable']):
            return "price_inquiry"
        
        # Menu item details
        if any(x in query_lower for x in ['what is', 'what are', 'tell me about', 'details', 'ingredients', 'what\'s in', 'what is in']):
            return "menu_item_details"
        
        # Dietary restriction
        if any(x in query_lower for x in ['vegetarian', 'vegan', 'veg', 'non-veg', 'non veg', 'allergy', 'allergen']):
            return "dietary_restriction"
        
        # Comparison
        if any(x in query_lower for x in ['compare', 'vs', 'versus', 'difference between', 'better']):
            return "comparison"
        
        # Category inquiry - check if any known category is mentioned
        for category in self.category_to_items.keys():
            if category.lower() in query_lower:
                return "category_inquiry"
        
        return "general"