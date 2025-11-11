import requests
import re
from bs4 import BeautifulSoup

def fetch_and_print_grid(url):
    # Fetch the HTML document
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table with coordinate data
    table = soup.find('table', class_='c5')
    if not table:
        print("No table found!")
        return
    
    # Extract data from table rows (skip header row)
    data = []
    rows = table.find_all('tr')[1:]  # Skip header row
    
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 3:
            x_text = cells[0].get_text().strip()
            char_text = cells[1].get_text().strip()
            y_text = cells[2].get_text().strip()
            
            # Parse coordinates
            try:
                x = int(x_text)
                y = int(y_text)
                char = char_text
                data.append((x, char, y))
            except ValueError:
                continue
    
    if not data:
        print("No coordinate data found!")
        return
    
    print(f"Found {len(data)} coordinate points")
    
    # Get bounds
    max_x = max(x for x, _, _ in data)
    max_y = max(y for _, _, y in data)
    
    print(f"Grid size: {max_x + 1} x {max_y + 1}")
    
    # Build grid (filled with spaces)
    grid = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    
    # Place characters at their coordinates
    for x, char, y in data:
        grid[y][x] = char
    
    # Print the grid
    print("\nGrid:")
    for row in grid:
        print("".join(row))

if __name__ == "__main__":
    url = "https://docs.google.com/document/d/e/2PACX-1vRPzbNQcx5UriHSbZ-9vmsTow_R6RRe7eyAU60xIF9Dlz-vaHiHNO2TKgDi7jy4ZpTpNqM7EvEcfr_p/pub"
    fetch_and_print_grid(url)