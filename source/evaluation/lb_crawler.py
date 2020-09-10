# Import Modules
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import requests
import time

# Input Link
link = 'https://lookbook.nu/explore/'
tag = 'formal'
link = link + tag

# Load webdriver
driver = webdriver.Chrome('../../module/webdriver/chromedriver.exe')
driver.get(link)
time.sleep(2)

# Auto pageDown
body = driver.find_element_by_tag_name("a")
pageDown = 1000
time.sleep(2)
while pageDown:
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.3)
    pageDown -= 1
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

# Get real url
url = []
items = driver.find_elements_by_class_name('look-square')
for item in range(len(items)):
    parse_url = driver.find_elements_by_class_name('look-image-link')
    url.append(parse_url[item].get_attribute('href'))
driver.close()

# Parse and Restore images
for idx in range(len(url)):
    req = requests.get(url[idx])
    bs = BeautifulSoup(req.text, 'html.parser')
    results = bs.find_all('img', {'id': 'main_photo'})
    if len(results) is 0:
        print('({}/{}) Passed: Private Image'.format(idx, len(url) - 1))
        pass
    else:
        img_link = 'https:' + results[0]['src']
        print('({}/{}) {}'.format(idx, len(url)-1, img_link))
        with open('../../data/image/evaluation/'+tag+'/'+str(idx)+'.jpg', 'wb') as f:
            f.write(requests.get(img_link).content)
            f.close()
