from bs4 import BeautifulSoup
import json
import re
import os

def scrape_dominos_menu(html_file):
    """
    Scrapes menu data from a Domino's Pizza HTML file
    
    Args:
        html_file: Path to the Domino's menu HTML file
        
    Returns:
        A list of dictionaries containing menu items with their details
    """
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all menu item cards
    menu_items = []
    
    # Get all card items
    card_items = soup.find_all('div', class_='card-item')
    
    for card in card_items:
        # Initialize item dictionary
        item = {
            'name': '',
            'description': '',
            'regular_price': None,
            'veg_status': None,
            'crust': None,
            'size': None,
            'striked_price': None,  # For discounted items
            'categories': []
        }
        
        # Get category header (parent of card items)
        category_header = card.find_previous('h3', class_='card-catg-name')
        if category_header:
            item['categories'].append(category_header.text.strip())
        
        # Extract name
        title_element = card.select_one('.pizza-title')
        if title_element:
            item['name'] = title_element.text.strip()
        
        # Extract description
        desc_element = card.select_one('.pizza-desc')
        if desc_element:
            item['description'] = desc_element.text.strip()
        
        # Extract price
        price_element = card.select_one('.pizza-price span span')
        if price_element:
            try:
                item['regular_price'] = float(price_element.text.strip())
            except ValueError:
                pass
        
        # Check if there's a striked-through original price (for discounted items)
        striked_price = card.select_one('.striked-price span')
        if striked_price:
            try:
                item['striked_price'] = float(striked_price.text.strip())
            except ValueError:
                pass
        
        # Extract vegetarian/non-vegetarian status
        veg_icon = card.select_one('.tag-veg')
        non_veg_icon = card.select_one('.tag-non-veg')
        if veg_icon:
            item['veg_status'] = "Veg"
        elif non_veg_icon:
            item['veg_status'] = "Non-Veg"
        
        # Extract crust and size
        size_element = card.select_one('.p-size')
        if size_element:
            item['size'] = size_element.text.strip()
        
        crust_element = card.select_one('.p-crust')
        if crust_element:
            item['crust'] = crust_element.text.strip()
        
        # Only add items with valid names
        if item['name']:
            menu_items.append(item)
    
    return menu_items

def save_to_json(menu_items, output_file):
    """
    Saves the menu items to a JSON file
    
    Args:
        menu_items: List of menu item dictionaries
        output_file: Path to save the JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"menu_items": menu_items}, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(menu_items)} menu items to {output_file}")
    
    # Print sample of what was scraped
    if menu_items:
        print("\nSample items:")
        for item in menu_items[:3]:
            print(f"- {item['name']} ({item['veg_status']}): ₹{item['regular_price']} - {item['description'][:50]}...")

def main():
    input_file = "dineout_menu1.html"
    output_file = "dominos_menu.json"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return
    
    # Scrape the data
    menu_items = scrape_dominos_menu(input_file)
    
    # Save to JSON
    save_to_json(menu_items, output_file)
    
    # Summary
    categories = set()
    veg_count = 0
    non_veg_count = 0
    
    for item in menu_items:
        categories.update(item['categories'])
        if item['veg_status'] == "Veg":
            veg_count += 1
        elif item['veg_status'] == "Non-Veg":
            non_veg_count += 1
    
    print(f"\nTotal menu items: {len(menu_items)}")
    print(f"Vegetarian items: {veg_count}")
    print(f"Non-vegetarian items: {non_veg_count}")
    print(f"Categories: {', '.join(categories)}")

if __name__ == "__main__":
    main()