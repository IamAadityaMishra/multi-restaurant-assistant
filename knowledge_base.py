import pandas as pd
import re
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os
from datetime import datetime

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def clean_text(text):
    """Clean and normalize text data"""
    if pd.isna(text) or text == "N/A":
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()

def extract_nutritional_info(description):
    """Extract nutritional information from description"""
    nutrition = {}
    
    # Look for calories
    cal_match = re.search(r'(?:Kcal|calories):\s*([\d.]+)', description, re.IGNORECASE)
    if cal_match:
        nutrition['calories'] = float(cal_match.group(1))
    
    # Extract other nutritional info
    patterns = {
        'carbs': r'Carbs\s*([\d.]+)',
        'protein': r'Protein:\s*([\d.]+)',
        'fat': r'Fat:\s*([\d.]+)',
        'sugar': r'Sugar:\s*([\d.]+)',
        'sodium': r'Sodium:\s*([\d.]+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            nutrition[key] = float(match.group(1))
    
    return nutrition

def extract_allergens(description):
    """Extract allergen information from description"""
    allergens = []
    if "Contains:" in description:
        allergen_text = description.split("Contains:")[1].strip()
        allergen_list = re.split(r',|\s+and\s+', allergen_text)
        allergens = [a.strip().lower() for a in allergen_list if a.strip()]
    
    return allergens

def create_knowledge_base_from_csv(csv_file, restaurant_name):
    """Create structured knowledge base for a single restaurant from CSV data"""
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize restaurant-specific knowledge base
    knowledge_base = {
        "restaurant_name": restaurant_name,
        "menu_items": [],
        "categories": set(),
        "price_ranges": {"min": float('inf'), "max": 0},
        "dietary_options": {"veg": [], "non_veg": []}
    }
    
    # Process each menu item
    for _, row in df.iterrows():
        # Skip duplicate entries based on name and price
        if any(item['name'] == row['Name'] and 
               item.get('regular_price') == (row['Regular Price'] if pd.notna(row['Regular Price']) else None) 
               for item in knowledge_base['menu_items']):
            continue
            
        # Basic item info
        item = {
            "name": row['Name'],
            "description": clean_text(row['Description']) if 'Description' in row else "",
            "veg_status": row.get('Veg/Non-Veg', "Unknown"),
            "restaurant": restaurant_name  # Add restaurant name to each item
        }
        
        # Handle prices
        if 'Regular Price' in row and pd.notna(row['Regular Price']) and row['Regular Price']:
            try:
                price = float(row['Regular Price'])
                item['regular_price'] = price
                # Update price ranges
                knowledge_base['price_ranges']['min'] = min(knowledge_base['price_ranges']['min'], price)
                knowledge_base['price_ranges']['max'] = max(knowledge_base['price_ranges']['max'], price)
            except (ValueError, TypeError):
                item['regular_price'] = None
        
        if 'Offer Price' in row and pd.notna(row['Offer Price']) and row['Offer Price']:
            try:
                item['offer_price'] = float(row['Offer Price'])
            except (ValueError, TypeError):
                item['offer_price'] = None
        
        # Extract calories if available
        if 'Calories' in row and pd.notna(row['Calories']) and row['Calories'] != "N/A":
            try:
                item['calories'] = float(row['Calories'])
            except (ValueError, TypeError):
                item['calories'] = None
        
        # Extract nutrition and allergens from description
        if 'Description' in row and pd.notna(row['Description']):
            item['nutrition'] = extract_nutritional_info(row['Description'])
            item['allergens'] = extract_allergens(row['Description'])
        else:
            item['nutrition'] = {}
            item['allergens'] = []
        
        # Determine item category based on name and description
        item_text = f"{row['Name']} {row['Description'] if 'Description' in row and pd.notna(row['Description']) else ''}"
        
        # Get category from CSV if available
        if 'Category' in row and pd.notna(row['Category']):
            categories = [cat.strip() for cat in str(row['Category']).split(',')]
            if categories:
                item['categories'] = categories
                knowledge_base['categories'].update(categories)
        else:
            # Auto-categorize if no category is provided
            categories = []
            # Generic categories that might apply across multiple restaurants
            if "burger" in item_text.lower():
                categories.append("burger")
            if "pizza" in item_text.lower():
                categories.append("pizza")
            if "wrap" in item_text.lower():
                categories.append("wrap")
            if "sandwich" in item_text.lower():
                categories.append("sandwich")
            if "salad" in item_text.lower():
                categories.append("salad")
            if "fries" in item_text.lower() or "nugget" in item_text.lower():
                categories.append("sides")
            if "dessert" in item_text.lower() or "sundae" in item_text.lower() or "ice cream" in item_text.lower():
                categories.append("dessert")
            if "coffee" in item_text.lower() or "tea" in item_text.lower() or "beverage" in item_text.lower():
                categories.append("beverages")
            
            # If no category determined, mark as "other"
            if not categories:
                categories.append("other")
            
            item['categories'] = categories
            knowledge_base['categories'].update(categories)
        
        # Add to veg or non-veg lists
        if item['veg_status'] == "Veg":
            knowledge_base['dietary_options']['veg'].append(row['Name'])
        elif item['veg_status'] == "Non-Veg":
            knowledge_base['dietary_options']['non_veg'].append(row['Name'])
        
        # Create search tokens for efficient retrieval
        tokens = word_tokenize(f"{row['Name']} {item['description']}")
        stop_words = set(stopwords.words('english'))
        item['search_tokens'] = [token.lower() for token in tokens if token.lower() not in stop_words]
        
        # Add to menu items
        knowledge_base['menu_items'].append(item)
    
    # Convert categories set to list for JSON serialization
    knowledge_base['categories'] = list(knowledge_base['categories'])
    
    return knowledge_base

def create_knowledge_base_from_json(json_file, restaurant_name=None):
    """Create structured knowledge base for a single restaurant from JSON data"""
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine restaurant name
    if restaurant_name is None:
        # Try to extract from filename if not provided
        restaurant_name = os.path.basename(json_file).split('_')[0].capitalize()
    
    # Initialize restaurant-specific knowledge base
    knowledge_base = {
        "restaurant_name": restaurant_name,
        "menu_items": [],
        "categories": set(),
        "price_ranges": {"min": float('inf'), "max": 0},
        "dietary_options": {"veg": [], "non_veg": []}
    }
    
    # Process menu items
    for item in data.get('menu_items', []):
        if not item:  # Skip empty items
            continue
            
        # Add restaurant name to each item
        item['restaurant'] = restaurant_name
        
        # Update price ranges
        if 'regular_price' in item and item['regular_price']:
            price = float(item['regular_price'])
            knowledge_base['price_ranges']['min'] = min(knowledge_base['price_ranges']['min'], price)
            knowledge_base['price_ranges']['max'] = max(knowledge_base['price_ranges']['max'], price)
        
        # Update categories
        if 'categories' in item:
            knowledge_base['categories'].update(item['categories'])
        
        # Update dietary options
        if 'veg_status' in item:
            if item['veg_status'] == "Veg":
                knowledge_base['dietary_options']['veg'].append(item['name'])
            elif item['veg_status'] == "Non-Veg":
                knowledge_base['dietary_options']['non_veg'].append(item['name'])
        
        # Add to menu items
        knowledge_base['menu_items'].append(item)
    
    # Convert categories set to list for JSON serialization
    knowledge_base['categories'] = list(knowledge_base['categories'])
    
    return knowledge_base

def merge_knowledge_bases(knowledge_bases):
    """Merge multiple restaurant knowledge bases into one combined knowledge base"""
    combined_kb = {
        "restaurants": [],
        "menu_items": [],
        "categories": set(),
        "price_ranges": {"min": float('inf'), "max": 0},
        "dietary_options": {"veg": [], "non_veg": []},
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Process each restaurant's knowledge base
    for kb in knowledge_bases:
        # Add restaurant info
        combined_kb['restaurants'].append({
            "name": kb['restaurant_name'],
            "categories": kb['categories'],
            "price_range": kb['price_ranges'],
            "item_count": len(kb['menu_items'])
        })
        
        # Add menu items
        combined_kb['menu_items'].extend(kb['menu_items'])
        
        # Update categories
        combined_kb['categories'].update(kb['categories'])
        
        # Update price ranges
        if kb['price_ranges']['min'] < combined_kb['price_ranges']['min']:
            combined_kb['price_ranges']['min'] = kb['price_ranges']['min']
        if kb['price_ranges']['max'] > combined_kb['price_ranges']['max']:
            combined_kb['price_ranges']['max'] = kb['price_ranges']['max']
        
        # Update dietary options
        combined_kb['dietary_options']['veg'].extend(kb['dietary_options']['veg'])
        combined_kb['dietary_options']['non_veg'].extend(kb['dietary_options']['non_veg'])
    
    # Convert categories set to list for JSON serialization
    combined_kb['categories'] = list(combined_kb['categories'])
    
    # Add summary statistics
    combined_kb['stats'] = {
        "total_restaurants": len(combined_kb['restaurants']),
        "total_menu_items": len(combined_kb['menu_items']),
        "veg_items_count": len(combined_kb['dietary_options']['veg']),
        "non_veg_items_count": len(combined_kb['dietary_options']['non_veg'])
    }
    
    return combined_kb

def save_knowledge_base(knowledge_base, output_file):
    """Save knowledge base to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    print(f"Knowledge base saved to {output_file}")
    print(f"Contains {len(knowledge_base['menu_items'])} menu items from {len(knowledge_base.get('restaurants', []))} restaurants")

def process_restaurant_data(data_sources):
    """Process multiple restaurant data sources and combine them"""
    knowledge_bases = []
    
    for source in data_sources:
        file_path = source['file']
        restaurant_name = source['name']
        file_type = source['type']
        
        print(f"Processing {restaurant_name} data...")
        
        if file_type == 'csv':
            kb = create_knowledge_base_from_csv(file_path, restaurant_name)
        elif file_type == 'json':
            kb = create_knowledge_base_from_json(file_path, restaurant_name)
        else:
            print(f"Unknown file type: {file_type}")
            continue
        
        print(f"Extracted {len(kb['menu_items'])} menu items for {restaurant_name}")
        knowledge_bases.append(kb)
    
    # Merge all knowledge bases
    combined_kb = merge_knowledge_bases(knowledge_bases)
    
    return combined_kb

def main():
    """Main function to create and save the combined knowledge base"""
    # Define restaurant data sources
    data_sources = [
        {
            'name': 'Burger King',
            'file': 'C:\\Users\\divya\\Downloads\\Zomato\\V2\\Scraper\\Burger_King\\burger_king_menu.csv',
            'type': 'csv'
        },
        {
            'name': 'Dominos',
            'file': 'C:\\Users\\divya\\Downloads\\Zomato\\V2\\Scraper\\Dominos\\dominos_menu.json',
            'type': 'json'
        },
        {
            'name': 'BiryaniByKilo',
            'file': 'C:\\Users\\divya\\Downloads\\Zomato\\V2\\Scraper\\BiryaniByKilo\\BiryaniByKilo_data.json',
            'type': 'json'
        },
        {
            'name': 'Behrouz',
            'file': 'C:\\Users\\divya\\Downloads\\Zomato\\V2\\Scraper\\Behrouz\\behrouz_menu.json',
            'type': 'json'
        }
        # Add more restaurants as needed
        # {
        #     'name': 'Pizza Hut',
        #     'file': 'pizza_hut_menu.csv',
        #     'type': 'csv'
        # },
    ]
    
    # Process all restaurant data
    combined_kb = process_restaurant_data(data_sources)
    
    # Save the combined knowledge base
    save_knowledge_base(combined_kb, "combined_restaurants_kb.json")
    
    print("Success! The combined knowledge base is ready for your RAG chatbot.")
    print(f"Total restaurants: {combined_kb['stats']['total_restaurants']}")
    print(f"Total menu items: {combined_kb['stats']['total_menu_items']}")

if __name__ == "__main__":
    main()