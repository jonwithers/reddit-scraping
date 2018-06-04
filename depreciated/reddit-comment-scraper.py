import pandas as pd
import requests
import json
import time

url_base = "http://www.reddit.com/"

slug_hot = "hot.json"

slug_bpt = "r/BlackPeopleTwitter/" # an optional intermediate slug to throw in to view a specific subreddit's 'hot' page
slug_wpt = "r/WhitePeopleTwitter/"

user = {'User-agent': 'JonBot 0.1'} # I need a User-agent to get in

list_of_dictionaries = []
aft = ''
features = ['created_utc', 'body', 'ups']

for i in range(1000):
    j = i % 25
    this_dict = {}
    url_wpt = url_base + slug_wpt + slug_hot + aft
    res = requests.get(url_wpt, headers = user)
    data = res.json()

    slug_wpt_id = data['data']['children'][j]['data']['id']
    comments_this_post = data['data']['children'][j]['data']['num_comments']
    post_title = data['data']['children'][j]['data']['title']
    
    if (i+1)%25==0:
        print("{} posts scraped!".format(i+1))
        aft = '?after='+data['data']['after']
    
    this_dict['comments_this_post'] = comments_this_post
    this_dict['post_title'] = post_title
#     print(post_title)
    url_wpt_comments = url_base + slug_wpt + 'comments/' + slug_wpt_id + '.json'
    res = requests.get(url_wpt_comments, headers = user)
    data = res.json()
    try:
        comment_data = data[1]['data']['children'][0]['data']
    except:
        pass
    for feature in features:
        this_dict[feature] = comment_data[feature]
#     print(comment_data['ups'])
    time.sleep(3)
    list_of_dictionaries.append(this_dict)

pd.DataFrame(list_of_dictionaries).to_csv('Wpt_1000.csv')
