# -*- coding: utf-8 -*-
import os
import concurrent.futures
from GoogleImageScraper import GoogleImageScraper
from patch import webdriver_executable
import pandas as pd
import sys 
from datetime import datetime

def worker_thread(search_key, search_url):
    image_scraper = GoogleImageScraper(webdriver_path, image_path, search_key,search_url, number_of_images, headless, min_resolution, max_resolution)
    image_urls = image_scraper.find_image_urls()
    image_scraper.save_images(image_urls, keep_filenames)
    del image_scraper

if __name__ == "__main__":
    start_time = datetime.now()
    
    if len(sys.argv) != 3:
        print("[ERROR] Wrong format")
        print(">>> python3 main.py data.csv 50")
        exit() 
    df = pd.read_csv(sys.argv[1])
    webdriver_path = os.path.normpath(os.path.join(os.getcwd(), 'webdriver', webdriver_executable()))
    image_path = os.path.normpath(os.path.join(os.getcwd(), 'photos'))

    search_keys = df['title'].tolist()
    search_url = df['url'].tolist()

    number_of_images = int(sys.argv[2])
    headless = True 
    min_resolution = (0, 0)              
    max_resolution = (9999, 9999)        
    keep_filenames = False              

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(search_keys)) as executor:
        executor.map(worker_thread, search_keys, search_url)

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))