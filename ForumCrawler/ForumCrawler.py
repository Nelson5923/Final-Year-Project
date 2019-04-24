import requests
from lxml import etree
from selenium import webdriver
import json
import re

import time
from browsermobproxy import Server
from FetchSingleThread import fetch
from pandas import ExcelWriter
import openpyxl

# Configure the Proxy Server for Monitoring HAR
server = Server("C:/browsermob-proxy-2.1.4/bin/browsermob-proxy.bat")
server.start()
proxy = server.create_proxy()
proxy.new_har("Data")

# Option
Option1 = webdriver.ChromeOptions()
Option1.add_argument("--proxy-server={0}".format(proxy.proxy))

Option2 = webdriver.ChromeOptions()
Option2.add_argument('headless')
prefs = {"profile.managed_default_content_settings.images": 2}
Option2.add_experimental_option("prefs", prefs)

# Initialize
driver = webdriver.Chrome(executable_path='C:\Chrome\chromedriver.exe', chrome_options=Option1)
driver2 = webdriver.Chrome(executable_path='C:\Chrome\chromedriver.exe', chrome_options=Option2)
forum = 'hkg'
url = "https://hkug.arukascloud.io/topics/5?type=" + forum
driver.get(url)

# Skip the Page

for skip in range(1):
    driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    time.sleep(10)
    button = driver.find_elements_by_xpath('//button[@class="ant-btn ant-btn-primary"]')[0]
    button.click()

# Swap the Page

for i in range(1,10000):

    # Click the Page Swap Button

    driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    time.sleep(6)
    button = driver.find_elements_by_xpath('//button[@class="ant-btn ant-btn-primary"]')[0]
    button.click()

    # Process the HAR File

    if forum == 'hkg':

        PAGE = '&page=' + str(i)
        print('--- Getting Page' + str(i) + ' ---')
        for ent in proxy.har['log']['entries']:
            if PAGE in ent['request']['url']:
                driver2.get(ent['request']['url'])
                j = json.loads(re.sub('<.*?>', '', driver2.page_source))
                for Topic in j['topic_list']:
                    if int(Topic['Total_Replies']) > 10:
                        url = 'https://hkug.arukascloud.io/topics/5/' + str(Topic['Message_ID']) + '?forum=HKG'
                        fetch(url,int(Topic['Total_Replies'] / 25) + 1, driver2, str(Topic['Message_ID']))

        print('--- Complete Page' + str(i) + ' ---')

    if forum == 'lihkg':

        PAGE = '&page=' + str(i)
        print('--- Getting Page' + str(i) + ' ---')
        for ent in proxy.har['log']['entries']:
            if PAGE in ent['request']['url']:
                driver2.get(ent['request']['url'])
                j = json.loads(re.sub('<.*?>', '', driver2.page_source))
                for Topic in j['response']['items']:
                    if int(Topic['no_of_reply']) > 10:
                        url = 'https://hkug.arukascloud.io/topics/5/' + str(Topic['thread_id']) + '?forum=LIHKG'
                        fetch(url,int(int(Topic['no_of_reply']) / 25) + 1, driver2, str(Topic['thread_id']))

        print('--- Complete Page' + str(i) + ' ---')

server.stop()
driver.close()
driver2.close()



