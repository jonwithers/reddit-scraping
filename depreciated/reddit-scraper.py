import pandas as pd
import requests
import json

url_base = "http://www.reddit.com/"

slug_hot = "hot.json"

slug_subreddit = "r/boston/" # an optional intermediate slug to throw in to view a specific subreddit's 'hot' page

user = {'User-agent': 'Jon Withers Bot 0.1'} # I need a User-agent to get in

res = requests.get(url_base + slug_hot, headers = user) # put slug_subreddit in if you want

data = res.json()

all_keys = data['data']['children'][0]['data'].keys()
all_keys -= ['preview', 'post_hint']

# things_to_get = ['title','subreddit','created_utc','num_comments', 'score','spoiler', 'distinguished']
things_to_get = list(all_keys)
list_of_dicts = []
for i in range(25):
    this_dict = {}
    for j in range(len(things_to_get)):

        this_key = things_to_get[j]
        this_dict[this_key] = data['data']['children'][i]['data'][this_key]
    list_of_dicts.append(this_dict)

print(pd.DataFrame(list_of_dicts))
