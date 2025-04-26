from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# Configure Selenium WebDriver
options = Options()
options.add_argument("--headless")  # Run in headless mode
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# Initialize WebDriver
driver = webdriver.Chrome(options=options)

# URL to scrape
url = "https://biryanibykilo.com/"
driver.get(url)

# Wait for the page to load completely
time.sleep(10)  # Adjust sleep time as needed

# Save the page source to an HTML file
with open("dineout_menu1.html", "w", encoding="utf-8") as file:
    file.write(driver.page_source)

print("✅ Page HTML saved as 'dineout_menu.html'.")

# Close the WebDriver
driver.quit()
