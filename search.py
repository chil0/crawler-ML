#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os,shutil
import re
import requests
from selenium import webdriver  
import time  
import urllib 
import img_resize

#输出目录
OUTPUT_DIR = 'img\\'
#关键字数组：将在输出目录内创建以以下关键字们命名的txt文件

SEARCH_KEY_WORDS = ['pine tree','maple leaves','sakura blossom']
#页数
PAGE_NUM = 20

repeateNum = 0
preLen = 0

keyword2int = {}

def getSearchUrl(keyWord):
	if(isEn(keyWord)):
		return 'https://www.google.com.hk/search?q=' + keyWord + '&safe=strict&source=lnms&tbm=isch'
	else:
		return 'https://www.google.com.hk/search?q=' + keyWord + '&safe=strict&hl=zh-CN&source=lnms&tbm=isch'

def isEn(keyWord):  
	return all(ord(c) < 128 for c in keyWord)

# 启动Firefox浏览器  


if os.path.exists(OUTPUT_DIR) == False:
	os.makedirs(OUTPUT_DIR)


def down_pic(pic_urls, keyword):
    """给出图片链接列表, 下载所有图片"""
    path = OUTPUT_DIR + str(keyword) + '\\'
    mkdir(path)
    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=5)
            string = path + str(i + 1) + '.jpg'
            with open(string, 'wb') as f:
                f.write(pic.content)
                print('Successfully downloaded the %sth picture: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            print('Failed while downloading %sth image: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path+' Created successfully')
        return True
    else:
        print(path+' Directory already exists')
        return False

def output(SEARCH_KEY_WORD):
	global repeateNum
	global preLen
	driver = webdriver.Firefox()
	
	print('Searching for the picture ' + SEARCH_KEY_WORD + ' , please wait...')

	# 如果此处为搜搜，搜索郁金香，此处可配置为：http://pic.sogou.com/pics?query=%D3%F4%BD%F0%CF%E3&di=2&_asf=pic.sogou.com&w=05009900&sut=9420&sst0=1523883106480
	# 爬取页面地址，该处为google图片搜索url  
	url = getSearchUrl(SEARCH_KEY_WORD);

	# 如果是搜搜，此处配置为：'//div[@id="imgid"]/ul/li/a/img'
	# 目标元素的xpath，该处为google图片搜索结果内img标签所在路径
	xpath = '//div[@id="rg"]/div/div/a/img'

	# 浏览器打开爬取页面  
	driver.get(url)  

	outputFile = OUTPUT_DIR + '/' + SEARCH_KEY_WORD + '.txt'
	outputSet = set()

	# 模拟滚动窗口以浏览下载更多图片  
	pos = 0  
	m = 0 # 图片编号  
	for i in range(PAGE_NUM):  
		pos += i*600 # 每次下滚600  
		js = "document.documentElement.scrollTop=%d" % pos  
		driver.execute_script(js)  
		time.sleep(1)
		for element in driver.find_elements_by_xpath(xpath):
			
			img_url = element.get_attribute('src')
			
			if img_url is not None and img_url.startswith('http'):
				
				outputSet.add(img_url)
		if preLen == len(outputSet):
			if repeateNum == 2:
				repeateNum = 0
				preLen = 0
				break
			else:
				repeateNum = repeateNum + 1
		else:
			repeateNum = 0
			preLen = len(outputSet)
	
	down_pic(outputSet, keyword2int[SEARCH_KEY_WORD])


	print(SEARCH_KEY_WORD+'Image search completed')
	
	driver.close()

def main():
	i = 0
	for val in SEARCH_KEY_WORDS:
		keyword2int[val] = i
		output(val)
		i = i + 1
	print(keyword2int)
	img_resize.entrance()


main()

import classify
classify.classify(keyword2int)