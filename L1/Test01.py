import requests
from bs4 import BeautifulSoup
import pandas as pd

# 请求URL
def get_page_content(request_url):
    url = 'http://car.bitauto.com/xuanchegongju/?mid=8'
    # 得到页面的内容
    headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    html=requests.get(url,headers=headers,timeout=10)
    content = html.text
    # 通过content创建BeautifulSoup对象
    soup = BeautifulSoup(content, 'html.parser', from_encoding='utf-8')
    return soup

# 找到完整的投诉信息框
def analysis(soup):
    temp = soup.find('div',class_='search-result-list')
    # 创建DataFrame
    df = pd.DataFrame(columns=['cx_name', 'cx_price'])
    a_list = temp.find_all('a')

    for a in a_list:
        # ToDo：提取汽车投诉信息
        temp = {}
        p_list = a.find_all('p')
        # 放到DataFrame中
        temp['cx_name'], temp['cx_price'] = p_list[0].text, p_list[1].text
        df = df.append(temp, ignore_index=True)
    return df


page_num = 3
base_url = 'http://www.http://car.bitauto.com/xuanchegongju/?mid=8&page='

# 创建DataFrame
result = pd.DataFrame(columns=['cx_name', 'cx_price'])

for i in range(page_num):
    request_url = base_url + str(i+1) + ".shtml"
    soup = get_page_content(request_url)
    df = analysis(soup)
    print(df)
    result = result.append(df)

result.to_csv('result1.csv', index=False)
