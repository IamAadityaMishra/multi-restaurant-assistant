from bs4 import BeautifulSoup
import json
import os
from datetime import datetime

def scrape_menu(html_file):
    """
    Scrapes restaurant menu data from HTML file
    
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
    restaurant_name = "Biryani By Kilo"
    restaurant_location = "Multiple Locations"
    
    # Find all menu items - using the specific classes in this HTML
    menu_items = []
    
    # Find product cards
    product_cards = soup.find_all("div", class_="recommendedCard_item_card__AdajQ")
    print(f"Found {len(product_cards)} menu items")
    
    for card in product_cards:
        item = {
            'name': '',
            'description': '',
            'price': None,
            'original_price': None,
            'discount': None,
            'veg_status': None,
            'category': None,
            'is_bestseller': False
            
        }
        
        # Extract name
        name_element = card.select_one("p.recommendedCard_itemName__mf3Bj")
        if name_element:
            item['name'] = name_element.text.strip()
        
        # Extract description
        description_element = card.select_one("p.recommendedCard_item_descriptionData_descriptionLabel__V_H8s")
        if description_element:
            item['description'] = description_element.text.strip()
        
        # Extract price
        price_element = card.select_one("div.recommendedCard_item_descriptionContainer_price__sAS9v span:last-child")
        if price_element:
            price_text = price_element.text.replace("₹", "").strip()
            try:
                item['price'] = float(price_text)
            except (ValueError, TypeError):
                pass
        
        # Extract original price (if exists)
        original_price_element = card.select_one("span.recommendedCard_item_descriptionContainer_strike_price__ayqOk")
        if original_price_element:
            original_price_text = original_price_element.text.replace("₹", "").strip()
            try:
                item['original_price'] = float(original_price_text)
                if item['price'] and item['original_price']:
                    item['discount'] = item['original_price'] - item['price']
            except (ValueError, TypeError):
                pass
        
        # Check if bestseller
        bestseller_badge = card.select_one("p.recommendedCard_item_bestseller__c1FyV")
        if bestseller_badge:
            item['is_bestseller'] = True
        
        # Determine if veg or non-veg
        veg_element = card.select_one("div.recommendedCard_item_descriptionContainer_type__TI7EU")
        if veg_element:
            if "recommendedCard_nonVeg__4wAK0" in veg_element.get("class", []):
                item['veg_status'] = "Non-Veg"
            else:
                item['veg_status'] = "Veg"
        
       
        
        # Get category from parent section
        # Look for the nearest category title which is outside the card
        # We'll add this later when processing all items
        
        # Extract popularity/order count
        order_text_element = card.select_one("p.recommendedCard_item_descriptionData_orderRate_label__GLZLq")
        if order_text_element:
            order_text = order_text_element.text
            if "People order in last week" in order_text:
                try:
                    item['popularity'] = int(order_text.split("People")[0].strip())
                except (ValueError, TypeError):
                    pass
        
        # Only add items with valid names
        if item['name']:
            menu_items.append(item)
    
    # Get categories from section headers
    category_headers = soup.find_all("p", class_="homePage_productBody_category_nameLabel__gjjCu homePage_productBody_category_category_title__Uke9q")
    categories = [header.text.strip() for header in category_headers]
    
    # Assign categories to items based on their position in the HTML
    current_category = None
    for header in soup.find_all(["p", "div"], class_=lambda c: c and "category_title" in c):
        current_category = header.text.strip()
        next_items = header.find_next("div", class_="productCardList_productCardListing__mA6MR")
        if next_items:
            for card in next_items.find_all("div", class_="recommendedCard_item_card__AdajQ"):
                name_element = card.select_one("p.recommendedCard_itemName__mf3Bj")
                if name_element:
                    item_name = name_element.text.strip()
                    # Find this item in our menu_items list and assign category
                    for item in menu_items:
                        if item['name'] == item_name:
                            item['category'] = current_category
                            break
    
    # Create restaurant data structure
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
    
    # Print sample items
    if data['menu_items']:
        print("\nSample items:")
        for item in data['menu_items'][:3]:
            print(f"- {item['name']}")
            if item['description']:
                print(f"  Description: {item['description']}")
            print(f"  Price: ₹{item['price'] if item['price'] else 'N/A'}")
            print(f"  Type: {item['veg_status'] if item['veg_status'] else 'Unknown'}")
            if item['category']:
                print(f"  Category: {item['category']}")
            print("")

def main():
    # Use specific HTML file
    html_file = "dineout_menu1.html"
    
    # Default output filename
    output_file = "menu_data.json"
    
    # Check if file exists
    if not os.path.exists(html_file):
        print(f"Error: File {html_file} not found")
        return
    
    # Scrape the data
    restaurant_data = scrape_menu(html_file)
    
    # Save to JSON
    save_to_json(restaurant_data, output_file)
    
    # Summary
    veg_count = sum(1 for item in restaurant_data['menu_items'] if item['veg_status'] == "Veg")
    non_veg_count = sum(1 for item in restaurant_data['menu_items'] if item['veg_status'] == "Non-Veg")
    bestseller_count = sum(1 for item in restaurant_data['menu_items'] if item['is_bestseller'])
    
    print(f"\nTotal menu items: {len(restaurant_data['menu_items'])}")
    print(f"Vegetarian items: {veg_count}")
    print(f"Non-vegetarian items: {non_veg_count}")
    print(f"Bestseller items: {bestseller_count}")
    
    # Analysis - price ranges
    if restaurant_data['menu_items']:
        prices = [item['price'] for item in restaurant_data['menu_items'] 
                 if item['price'] is not None]
        if prices:
            print(f"Price range: ₹{min(prices)} to ₹{max(prices)}")
            print(f"Average price: ₹{sum(prices)/len(prices):.2f}")

if __name__ == "__main__":
    main()