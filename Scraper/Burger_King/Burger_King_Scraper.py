from bs4 import BeautifulSoup
import csv
import re

def parse_burger_king_menu(file_path):
    """
    Parse Burger King menu HTML file and extract product information.
    """
    with open(file_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Find all product items
    items = soup.find_all("div", class_="card-gen_wrapper card-gen_wrapper__block")

    # Create CSV to store results
    with open("burger_king_menu.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Regular Price", "Offer Price", "Description", "Calories", "Veg/Non-Veg"])

        for item in items:
            # Extract product name
            name_el = item.find("div", class_="card-gen__name-text")
            name = name_el.text.strip() if name_el else "N/A"
            
            # Extract prices
            price_el = item.find("span", class_="card-gen__currency")
            reg_price = ""
            if price_el:
                # Extract just the number from format like "₹ 99/-"
                reg_price = re.search(r'₹\s*(\d+)', price_el.text)
                reg_price = reg_price.group(1) if reg_price else ""
            
            offer_price_el = item.find("span", class_="card-gen__offer-currency")
            offer_price = ""
            if offer_price_el:
                # Extract just the number from format like "₹ 226/-"
                offer_price = re.search(r'₹\s*(\d+)', offer_price_el.text)
                offer_price = offer_price.group(1) if offer_price else ""
            
            # Extract description
            desc_el = item.find("div", class_="card-gen__description")
            description = desc_el.text.strip() if desc_el else ""
            
            # Extract calories
            calories = "N/A"
            nutrition_el = item.find("div", style="color: rgb(97, 97, 97); font-size: 14px;")
            if nutrition_el:
                calories_match = re.search(r'([\d.]+)\s*Kcal', nutrition_el.text)
                if calories_match:
                    calories = calories_match.group(1)
            
            # Check if item is vegetarian or non-vegetarian
            veg_status = "Veg"
            veg_img = item.find("img", src="/static/media/veg.2d5a7ccc.svg")
            non_veg_img = item.find("img", src=re.compile(r'data:image/png;base64.*'))
            if non_veg_img and not veg_img:
                veg_status = "Non-Veg"
            
            # Write to CSV
            writer.writerow([name, reg_price, offer_price, description, calories, veg_status])
            print(f"{name} - ₹{reg_price} (₹{offer_price}) - {veg_status}")

if __name__ == "__main__":
    parse_burger_king_menu("burger_king.html")