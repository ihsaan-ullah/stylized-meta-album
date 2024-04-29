# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

import time
from urllib.parse import urlparse
import os
import requests
import io
from PIL import Image
import patch

def split_size(data):
    x,_,y = data.split(" ") 
    return [int(x.replace(' ', '')),int(y.replace(' ', '')) ]

def check_copyright_exists(xpath):
    try:
        webdriver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return True
    return False

class GoogleImageScraper():
    def __init__(self, webdriver_path, image_path, search_key, search_url, number_of_images=1, headless=True, min_resolution=(0, 0), max_resolution=(1920, 1080)):
        image_path = os.path.join(image_path, search_key) 
        if not os.path.exists(image_path):
            print("[INFO] Image path not found. Creating a new folder.")
            os.makedirs(image_path)
        while(True):
            try:
                options = Options()
                if(headless):
                    options.add_argument('--headless')
                driver = webdriver.Chrome(webdriver_path, chrome_options=options)
                driver.set_window_size(1400,1050)
                driver.get("https://www.google.com")
                if driver.find_elements_by_id("L2AGLb"):
                    driver.find_element_by_id("L2AGLb").click()
                break
            except: 
                try:
                    driver
                except NameError:
                    is_patched = patch.download_lastest_chromedriver()
                else:
                    is_patched = patch.download_lastest_chromedriver(driver.capabilities['version'])
                if (not is_patched):
                    exit("[ERR] error chromedriver https://chromedriver.chromium.org/downloads")

        self.driver = driver
        self.search_key = search_key
        self.number_of_images = number_of_images
        self.webdriver_path = webdriver_path
        self.image_path = image_path
        #self.url = "https://www.google.com/search?q=%s&source=lnms&tbm=isch&sa=X&ved=2ahUKEwie44_AnqLpAhUhBWMBHUFGD90Q_AUoAXoECBUQAw&biw=1920&bih=947"%(search_key)
        self.url = search_url
        self.headless=headless
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.alt_link = []

    def find_image_urls(self): 
        print("[INFO] Gathering image links")
        image_urls=[]
        count = 0
        missed_count = 0
        self.driver.get(self.url)
        time.sleep(3)
        indx = 1
        while self.number_of_images > count:
            try:
                copyright_label = self.driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div[%s]/a[1]/div[3]'%(str(indx))).get_attribute('outerHTML')
                if("w8utCe gFhjPe h312td" in copyright_label):
                    self.number_of_images+=1
                    indx += 1
                    count += 1
                    continue
            except Exception:
                try:
                    imgurl = self.driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div[%s]/a[1]/div[1]/img'%(str(indx)))
                    imgurl.click()
                    missed_count = 0 

                except Exception:
                    missed_count = missed_count + 1
                    if (missed_count > 10):
                        print("[INFO] Maximum missed photos reached, exiting...")
                        break

                try:
                    time.sleep(1)
                    class_names = ["n3VNCb"]
                    span_names = ["VSIspc"]
                    spans = [self.driver.find_elements_by_class_name(span_name) for span_name in span_names if len(self.driver.find_elements_by_class_name(span_name)) != 0 ][0]
                    images = [self.driver.find_elements_by_class_name(class_name) for class_name in class_names if len(self.driver.find_elements_by_class_name(class_name)) != 0 ][0]
                    for image, span in zip(images,spans):
                        # span_split = span.get_attribute('innerHTML')
                        # compare = split_size(span_split)
                        # print(compare)
                        # if compare[0]<512 and compare[1]<512:
                        #     self.number_of_images+=1
                        #     indx += 1
                        #     count += 1
                        #     continue

                        src_link = image.get_attribute("src")
                        alt_link = image.get_attribute("alt")
                        if(("http" in  src_link) and (not "encrypted" in src_link)):
                            print(
                                f"[INFO] {self.search_key} \t #{count} \t {src_link}")
                            image_urls.append(src_link)
                            self.alt_link.append(alt_link)
                            count +=1
                            break
                except Exception:
                    print("[INFO] Unable to get link")

                try:
                    if(count%3==0):
                        self.driver.execute_script("window.scrollTo(0, "+str(indx*60)+");")
                    element = self.driver.find_element_by_class_name("mye4qd")
                    element.click()
                    print("[INFO] Loading next page")
                    time.sleep(3)
                except Exception:
                    time.sleep(1)
                indx += 1
        self.driver.quit()
        return image_urls

    def save_images(self,image_urls, keep_filenames):
        print(keep_filenames) 
        print("[INFO] Saving image, please wait...")
        count_image = 1
        for indx,image_url in enumerate(image_urls):
            try:
                print("[INFO] Image url:%s"%(image_url))
                search_string = ''.join(e for e in self.search_key if e.isalnum())
                image = requests.get(image_url,timeout=5)
                if image.status_code == 200:
                    with Image.open(io.BytesIO(image.content)) as image_from_web:
                        if(image_from_web.height<512 and image_from_web.width<512):
                            continue
                        try:
                            if (keep_filenames): 
                                o = urlparse(image_url)
                                image_url = o.scheme + "://" + o.netloc + o.path
                                name = os.path.splitext(os.path.basename(image_url))[0] 
                                filename = "%s.%s"%(name,image_from_web.format.lower())
                            else:
                                filename = "%s%s.%s"%(search_string,str(count_image),image_from_web.format.lower())
                                count_image+=1

                            image_path = os.path.join(self.image_path, filename)

                            img_exif = image_from_web.getexif()
                            #img_exif[33432] = image_url
                            ctime = time.time()
                            img_exif[306] = time.ctime(ctime)
                            img_exif[315] = image_url
                            img_exif[270] = self.alt_link[indx]

                            image_from_web.save(image_path, exif = img_exif)
                            
                        except OSError:
                            rgb_im = image_from_web.convert('RGB')
                            rgb_im.save(image_path)
                        image_resolution = image_from_web.size
                        if image_resolution != None:
                            if image_resolution[0]<self.min_resolution[0] or image_resolution[1]<self.min_resolution[1] or image_resolution[0]>self.max_resolution[0] or image_resolution[1]>self.max_resolution[1]:
                                image_from_web.close()
                                os.remove(image_path)

                        image_from_web.close()
            except Exception as e:
                print("[ERROR] Download failed: ",e)
                pass
        print("✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅")
