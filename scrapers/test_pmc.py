import urllib.request
import json
import re
import xml.etree.ElementTree as ET

def test_pmc():
    query = "dysgraphia+handwriting"
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term={query}&retmode=json&retmax=10"
    req = urllib.request.Request(url, headers={'User-Agent': 'DysgraphiaResearch/1.0'})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    pmc_ids = data.get('esearchresult', {}).get('idlist', [])
    print(f"Found {len(pmc_ids)} PMC articles for query '{query}': {pmc_ids}")
    
    # Try fetching full text XML for the first article
    for pmcid in pmc_ids[:3]:
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&retmode=xml"
        req = urllib.request.Request(fetch_url, headers={'User-Agent': 'DysgraphiaResearch/1.0'})
        try:
            with urllib.request.urlopen(req) as resp:
                xml_content = resp.read().decode('utf-8', errors='ignore')
            # Look for graphic xlink:href
            graphics = re.findall(r'<graphic[^>]+xlink:href="([^"]+)"', xml_content)
            print(f"PMC{pmcid}: Found {len(graphics)} graphic figures: {graphics[:3]}")
        except Exception as e:
            print(f"PMC{pmcid} fetch error: {e}")

if __name__ == '__main__':
    test_pmc()
