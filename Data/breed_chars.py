#!python3

import pandas as pd
import os
from os import makedirs
from os.path import join, exists, expanduser
import json

data_dir = '/home/sclaypool/gitProjects/DogApp/Data/'

url_file = join(data_dir, 'urls.json')
urls = json.load(open(url_file, 'r'))

trait_file = join(data_dir, 'traits.json')
traits = json.load(open(trait_file, 'r'))
 
# def init_traits():
    # _traits = {'weight': [],
               # 'height': [],
               # 'group': ['hound'],
               # 'coat': ['short', 'medium', 'long', 'curly', 'hairless'],
               # 'ear': ['pointed', 'round', 'long'],
               # 'build': ['skinny', 'muscular', 'squat', 'normal'],
               # }
    # return _traits

def load_labs():
    labs = pd.read_csv(join(data_dir, 'labels.csv'))
    return labs

def get_resp(trait, vals):
    resp = input().lower().strip()
    if len(resp) == 0:
        return get_resp(traits, vals)
    if len(vals) > 0:
        # change categorical to that item if given a number
        if resp.isnumeric():
            resp = vals[int(resp) - 1]
        elif not resp in traits[trait]:
            print(f"do you want to add {resp}? (y/n)")
            resp2 = input()
            if len(resp) > 0 and resp2[0] == 'y':
                traits[trait].append(resp)
                with open(trait_file, 'w') as tf:
                    json.dump(traits, tf)
            else:
                resp = get_resp(trait, vals)

    return resp


def open_image(breed):
    url = f"https://www.akc.org/?s={breed.replace(' ', '-')}"
    update = True
    if breed in urls.keys():
        url = urls[breed]
        update = False
    # os.system(f"google-chrome --new-window https://www.bing.com/images/search?q={breed.replace(' ', '+')}&FORM=HDRSC2")
    os.system(f"google-chrome {url}")


    if (update):
        new_url = input("real url: ").strip().lower()
        urls[breed] = new_url
        with open(url_file, 'w') as ufile:
            json.dump(urls, ufile)


def document(breed):
    print(f"BREED: {breed}")
    open_image(breed)
    breed_traits = {}

    for trait, vals in traits.items():
        print(f'trait: {trait}')
        for i, val in enumerate(vals):
            print(f'{i + 1}. {val}')

        breed_traits[trait] = get_resp(trait, vals)

    return breed_traits


def document_breeds(breeds):
    dog_dict = {}
    json_file = join(data_dir, 'breeds.json')
    with open(json_file, 'r') as prev_data:
        dog_dict = json.load(prev_data)

    cont = False
    for i, breed in enumerate(breeds):
        if not cont:
            edit = True
        if breed in dog_dict.keys():
            print(f"{i} / {len(breeds)}. current for {breed}: {dog_dict[breed]}")
            if not cont:
                resp = input("do you want to edit? (y/n/c)").strip().lower()
                if len(resp) > 0 and resp[0] == 'n':
                    edit = False
                elif len(resp) > 0 and resp[0] == 'c':
                    edit = False
                    cont = True
        else:
            cont = False
            edit = True
        if edit:
            print(f"{i} / {len(breeds)}")
            vals = document(breed)
            dog_dict[breed] = vals

        with open(json_file, "w") as data_file:
            json.dump(dog_dict, data_file)



if __name__ == "__main__":
    print("load_characteristics")
    l = load_labs()
    b = l.breed.unique()
    print(b)
    document_breeds(b)
