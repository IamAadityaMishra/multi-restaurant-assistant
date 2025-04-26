import streamlit as st
from chatbot import ImprovedRestaurantChatbot
import os
import time

# Page configuration
st.set_page_config(
    page_title="Multi-Restaurant Assistant",
    page_icon="🍽️",
    layout="wide"
)

# Apply custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #FF4B4B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .restaurant-selector {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .restaurant-logo {
        width: 40px;
        height: 40px;
        margin-right: 10px;
        border-radius: 50%;
        object-fit: cover;
    }
    .stTextInput>div>div>input {
        caret-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# Initialize or get chatbot from session state
if 'chatbot' not in st.session_state:
    # Check if knowledge base exists, otherwise create it
    kb_path = "combined_restaurants_kb.json"
    if not os.path.exists(kb_path):
        st.error("Combined knowledge base not found. Please run knowledge_base.py first.")
        st.stop()
    
    with st.spinner("Initializing chatbot..."):
        st.session_state.chatbot = ImprovedRestaurantChatbot(kb_path)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your multi-restaurant assistant. I can help you with menu items, prices, and dietary information from Burger King, Dominos, BiryaniByKilo, and Behrouz. What would you like to know?"}
    ]

# Initialize current restaurant focus
if 'current_focus' not in st.session_state:
    st.session_state.current_focus = None

# Create two-column layout for header area
header_col1, header_col2 = st.columns([3, 1])

# Display header in first column
with header_col1:
    st.markdown('<p class="main-header">🍽️ Multi-Restaurant Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find and compare menu items across popular restaurants</p>', unsafe_allow_html=True)

# Restaurant selector in second column
with header_col2:
    restaurant_options = ["All Restaurants"] + st.session_state.chatbot.restaurant_names
    
    # Determine the default index based on current focus
    default_index = 0
    if st.session_state.current_focus in restaurant_options:
        default_index = restaurant_options.index(st.session_state.current_focus)
    
    selected_restaurant = st.selectbox(
        "Focus on restaurant:",
        options=restaurant_options,
        index=default_index,
        key="restaurant_selector"
    )

    # Update restaurant focus when selector changes
    if selected_restaurant != st.session_state.current_focus:
        if selected_restaurant == "All Restaurants":
            st.session_state.current_focus = None
            st.session_state.chatbot.current_restaurant_focus = None
        else:
            st.session_state.current_focus = selected_restaurant
            st.session_state.chatbot.current_restaurant_focus = selected_restaurant
        
        # Add a system message about the focus change
        if len(st.session_state.messages) > 1:  # Only add if there are previous messages
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"I'll now focus on {selected_restaurant}. How can I help you?"
                if selected_restaurant != "All Restaurants" 
                else "I'll now show results from all restaurants. How can I help you?"
            })

# Display info about available restaurants
restaurant_info_col1, restaurant_info_col2 = st.columns([2, 1])

with restaurant_info_col1:
    # Create tabs if we have multiple restaurants
    if hasattr(st.session_state.chatbot, 'restaurants'):
        # Only show restaurant info if not focused on a specific restaurant
        if not st.session_state.current_focus:
            st.markdown("### Available Restaurants")
            restaurants = st.session_state.chatbot.restaurants
            
            # Create two columns to display restaurant info
            for i in range(0, len(restaurants), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(restaurants):
                        rest = restaurants[i]
                        price_range = st.session_state.chatbot.price_by_restaurant.get(rest['name'], {})
                        items = st.session_state.chatbot.restaurant_to_items.get(rest['name'], [])
                        veg_count = sum(1 for item in items if item.get('veg_status') == 'Veg')
                        non_veg_count = sum(1 for item in items if item.get('veg_status') == 'Non-Veg')
                        
                        st.markdown(f"**{rest['name']}**")
                        st.markdown(f"- 🍽️ {len(items)} menu items ({veg_count} veg, {non_veg_count} non-veg)")
                        st.markdown(f"- 💰 Price range: ₹{price_range.get('min', 0)} to ₹{price_range.get('max', 0)}")
                
                with col2:
                    if i+1 < len(restaurants):
                        rest = restaurants[i+1]
                        price_range = st.session_state.chatbot.price_by_restaurant.get(rest['name'], {})
                        items = st.session_state.chatbot.restaurant_to_items.get(rest['name'], [])
                        veg_count = sum(1 for item in items if item.get('veg_status') == 'Veg')
                        non_veg_count = sum(1 for item in items if item.get('veg_status') == 'Non-Veg')
                        
                        st.markdown(f"**{rest['name']}**")
                        st.markdown(f"- 🍽️ {len(items)} menu items ({veg_count} veg, {non_veg_count} non-veg)")
                        st.markdown(f"- 💰 Price range: ₹{price_range.get('min', 0)} to ₹{price_range.get('max', 0)}")

# Sidebar with sample questions
with restaurant_info_col2:
    st.markdown("### Try asking:")
    
    # If there's a specific restaurant focus, adapt suggestions
    if st.session_state.current_focus:
        restaurant = st.session_state.current_focus
        st.markdown(f"- What's on the menu at {restaurant}?")
        st.markdown(f"- What vegetarian options does {restaurant} have?")
        st.markdown(f"- What are the best sellers at {restaurant}?")
        st.markdown(f"- What's the price range at {restaurant}?")
    else:
        st.markdown("- Compare prices across restaurants")
        st.markdown("- Which restaurant has the best pizza?")
        st.markdown("- Show me vegetarian options at all restaurants")
        st.markdown("- Which restaurant has the cheapest burgers?")

    st.markdown("### Common queries:")
    st.markdown("- What dishes do you recommend?")
    st.markdown("- Compare Burger King and Dominos")
    st.markdown("- Show me items under ₹200")
    st.markdown("- What's in the Whopper?")

# Create a horizontal line to separate header from chat
st.markdown("<hr>", unsafe_allow_html=True)

# Create the chat container
chat_container = st.container()
with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Function for processing user input
def process_user_input(user_input):
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get response from chatbot with timing
        with st.spinner("Thinking..."):
            start_time = time.time()
            response = st.session_state.chatbot.generate_response(user_input)
            end_time = time.time()
            print(f"Response generated in {end_time - start_time:.2f} seconds")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

# Chat input
if prompt := st.chat_input("Ask me about any restaurant or menu item..."):
    process_user_input(prompt)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Built with Streamlit • Uses data from multiple restaurant menus • Last updated April 2025"
    "</div>", 
    unsafe_allow_html=True
)