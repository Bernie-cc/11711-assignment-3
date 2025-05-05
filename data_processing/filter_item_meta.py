import gzip
import json

# Load Beauty_5.json and extract unique ASINs
with open('Beauty_5.json', 'r') as beauty_file:
    beauty_data = [json.loads(line) for line in beauty_file]
    beauty_asins = set(entry['asin'] for entry in beauty_data)

print(f"Extracted {len(beauty_asins)} unique ASINs from Beauty_5.json.")

# Filter meta_Beauty.json.gz based on ASINs
filtered_meta = []
with gzip.open('meta_Beauty.json.gz', 'r') as meta_file:  # Open gzipped file in text mode
    for line in meta_file:
        try:
            meta_entry = eval(line)  
            if meta_entry['asin'] in beauty_asins:
                filtered_meta.append(meta_entry)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: - Error: {e}")

# Write the filtered data to meta_Beauty_filter.json
with open('meta_Beauty_filter.json', 'w') as output_file:
    for entry in filtered_meta:
        json.dump(entry, output_file)
        output_file.write('\n')
