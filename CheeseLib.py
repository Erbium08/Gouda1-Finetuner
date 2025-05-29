import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import time
import string

BASE_URL = "https://www.cheese.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_cheese_links_for_letter(letter):
    page = 1
    all_links = []

    while True:
        url = f"{BASE_URL}/alphabetical/{letter}/?page={page}"
        print(f"\n🔍 Fetching: {url}")

        try:
            res = requests.get(url, headers=HEADERS)
            print(f"🔁 Status code: {res.status_code}")
            if res.status_code != 200:
                print(f"❌ Failed to fetch page {page} for letter '{letter}'")
                break

            soup = BeautifulSoup(res.text, "html.parser")
            cheese_links = soup.select("div.product-item h3 a[href]")

            if not cheese_links:
                print(f"⚠️ No more cheeses found on page {page} for '{letter}' — ending.")
                break

            print(f"✅ Found {len(cheese_links)} cheeses on page {page}")
            for link in cheese_links[:3]:
                print(f"   ➤ {link.text.strip()} → {BASE_URL + link['href']}")

            all_links.extend(BASE_URL + a['href'] for a in cheese_links)
            page += 1
            time.sleep(0.5)

        except Exception as e:
            print(f"❗ Error on page {page} for letter '{letter}': {e}")
            break

    print(f"🔢 Finished letter '{letter}' — total links: {len(all_links)}")
    return all_links



def get_all_cheese_links():
    all_links = []
    for letter in tqdm(string.ascii_lowercase, desc="Collecting cheese URLs"):
        links = get_cheese_links_for_letter(letter)
        print(f"🔢 Total collected so far: {len(all_links)} + {len(links)} = {len(all_links) + len(links)}")
        all_links.extend(links)
        time.sleep(0.5)
    deduped = list(set(all_links))
    print(f"\n🧀 Total unique cheese pages collected: {len(deduped)}")
    return deduped

def parse_cheese_page(url):
    res = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")

    name = soup.find("h1").text.strip()
    description_block = soup.find("div", class_="description")
    description = description_block.text.strip() if description_block else ""

    info_section = soup.find("section", id="cheese-info")
    rows = info_section.find_all("tr") if info_section else []
    info_dict = {}
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 2:
            key = cols[0].text.strip().lower().rstrip(":")
            val = cols[1].text.strip()
            info_dict[key] = val

    def safe(key):
        return info_dict.get(key, "unknown")

    summary = (
        f"From {safe('country')}, this cheese uses {safe('milk')} milk in the region {safe('region')}. "
        f"It is a {safe('type')} cheese with a {safe('texture')} texture, it has a {safe('rind')} rind and a {safe('colour')} colour. "
        f"It has flavours of {safe('flavour')} with an aroma of {safe('aroma')}, it "
        f"{'is' if 'yes' in safe('vegetarian').lower() else 'is not'} vegetarian and "
        f"{'is' if 'yes' in safe('vegan').lower() else 'is not'} vegan."
    )

    input_text = f"{description}\n\n{summary}"
    output_text = name
    if "synonyms" in info_dict:
        output_text += f" | Aliases: {info_dict['synonyms']}"

    return {
        "input": input_text,
        "output": output_text
    }

def main():
    links = get_all_cheese_links()
    print(f"✅ Found {len(links)} cheese pages.")

    results = []
    for url in tqdm(links, desc="Scraping cheese pages"):
        try:
            result = parse_cheese_page(url)
            results.append(result)
            time.sleep(0.3)
        except Exception as e:
            print(f"Error parsing {url}: {e}")

    with open("cheese_dataset.jsonl", "w", encoding="utf-8") as f:
        for entry in results:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Done! Saved {len(results)} cheese entries.")

if __name__ == "__main__":
    main()
