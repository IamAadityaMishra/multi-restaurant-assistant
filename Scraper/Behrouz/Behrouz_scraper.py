from bs4 import BeautifulSoup
import json
import re
import os
from datetime import datetime

def scrape_behrouz_menu(html_file):
    """
    Scrapes Behrouz Biryani menu data from HTML file
    
    Args:
        html_file: Path to the HTML file
        
    Returns:
        Dictionary with restaurant info and menu items
    """
    print(f"Reading HTML file: {html_file}")
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Get restaurant info
    restaurant_name = "Behrouz Biryani"
    location = soup.select_one("p.style__LocalityName-sc-12fdzoi-25")
    restaurant_location = location.text.strip() if location else ""
    
    # Find all menu item cards - using the specific class structure
    menu_items = []
    
    # Process all product cards with the specific class
    product_cards = soup.select("div.style__CardWrapper-ont81y-0")
    
    print(f"Found {len(product_cards)} potential menu items")
    
    for card in product_cards:
        # Initialize item dictionary
        item = {
            'name': '',
            'description': '',
            'regular_price': None,
            'veg_status': None,
            'categories': [],
            
            'restaurant': restaurant_name,
            'location': restaurant_location
        }
        
        # Extract name
        name_element = card.select_one("div[data-qa='productName'] p")
        if name_element:
            item['name'] = name_element.text.strip()
        
        # Extract price
        price_element = card.select_one("span[data-qa='totalPrice']")
        if price_element:
            # Get the price value without the ₹ symbol
            price_text = price_element.text.replace("₹", "").strip()
            try:
                item['regular_price'] = float(price_text)
            except (ValueError, TypeError):
                pass
        
        # Extract veg/non-veg status
        veg_icon = card.select_one("div[data-qa='isVeg']")
        non_veg_icon = card.select_one("div[data-qa='isNonVeg']")
        
        if veg_icon:
            item['veg_status'] = "Veg"
        elif non_veg_icon:
            item['veg_status'] = "Non-Veg"
        
        # Extract image URL
      
        
        # Try to determine category from headers
        parent_section = card.find_parent("div", class_="style__CardContent-sc-8kb98l-4")
        if parent_section:
            category_header = parent_section.find_previous("h1", class_="style__CardHeading-sc-8kb98l-2")
            if category_header:
                item['categories'].append(category_header.text.strip())
        
        # Only add items with valid names
        if item['name']:
            menu_items.append(item)
    
    # Extract all category names for reference
    categories = []
    category_headers = soup.find_all("h1", class_="style__CardHeading-sc-8kb98l-2")
    for header in category_headers:
        category = header.text.strip()
        if category:
            categories.append(category)
    
    restaurant_data = {
        "restaurant_name": restaurant_name,
        "restaurant_location": restaurant_location,
        "categories": categories,
        "menu_count": len(menu_items),
        "scraped_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "menu_items": menu_items
    }
    
    return restaurant_data

def save_to_json(data, output_file):
    """
    Saves the scraped data to a JSON file
    
    Args:
        data: Dictionary with restaurant and menu info
        output_file: Path to save the JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(data['menu_items'])} menu items to {output_file}")
    
    # Print summary
    if data['menu_items']:
        print("\nSample items:")
        for item in data['menu_items'][:3]:
            print(f"- {item['name']} ({item['veg_status']}): ₹{item['regular_price']}")
            if 'categories' in item and item['categories']:
                print(f"  Category: {', '.join(item['categories'])}")

def main():
    input_file = "c:\\Users\\divya\\Downloads\\Zomato\\V2\\Scraper\\dineout_menu1.html"
    output_file = "behrouz_menu.json"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return
    
    # Scrape the data
    restaurant_data = scrape_behrouz_menu(input_file)
    
    # Save to JSON
    save_to_json(restaurant_data, output_file)
    
    # Summary
    veg_count = sum(1 for item in restaurant_data['menu_items'] if item['veg_status'] == "Veg")
    non_veg_count = sum(1 for item in restaurant_data['menu_items'] if item['veg_status'] == "Non-Veg")
    
    print(f"\nTotal menu items: {len(restaurant_data['menu_items'])}")
    print(f"Vegetarian items: {veg_count}")
    print(f"Non-vegetarian items: {non_veg_count}")
    
    # Analysis - price ranges
    if restaurant_data['menu_items']:
        prices = [item['regular_price'] for item in restaurant_data['menu_items'] 
                 if item['regular_price'] is not None]
        if prices:
            print(f"Price range: ₹{min(prices)} to ₹{max(prices)}")
            print(f"Average price: ₹{sum(prices)/len(prices):.2f}")
    
    # Generate a CSV version too


if __name__ == "__main__":
    main()