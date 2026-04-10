import os
from datasets import load_dataset

def download_and_save_wiki():
    # Define the subsets you want to download
    wiki_subsets = {
        "or": "20231101.or",
        "hi": "20231101.hi",
        "kn": "20231101.kn",
        "tcy": "20231101.tcy"
    }

    # Create a directory to store the downloaded datasets
    save_dir = "./wikipedia_dumps"
    os.makedirs(save_dir, exist_ok=True)

    for lang_code, subset in wiki_subsets.items():
        print(f"[{lang_code.upper()}] Downloading Wikipedia subset: {subset}...")
        
        try:
            # 1. Download/Load the dataset
            # Note: Wikipedia dataset usually has a 'train' split by default
            ds = load_dataset("wikimedia/wikipedia", subset, split="train")
            
            print(f"[{lang_code.upper()}] Downloaded {len(ds)} articles. Saving to disk...")
            
            # 2. Save it locally
            output_path = os.path.join(save_dir, f"wiki_{lang_code}")
            ds.save_to_disk(output_path)
            
            print(f"[{lang_code.upper()}] Successfully saved to {output_path}\n")
            
        except Exception as e:
            print(f"[{lang_code.upper()}] Failed to download or save. Error: {e}\n")

    print("All downloads finished!")

if __name__ == "__main__":
    download_and_save_wiki()