from pymongo import MongoClient, ASCENDING, DESCENDING
import os
from urllib import quote_plus
import random

from credentials import MONGOLAB_URL

# Loop through sorted redactions
# Add data to mongdb
# Include series, control symbol

REDACTIONS_DIR = '/Users/tim/mycode/asio-timeline/src/processing/redactions'


def upload(start=None):
    '''
    Upload basic details of redactions to db
    '''
    dbclient = MongoClient(MONGOLAB_URL)
    db = dbclient.get_default_database()
    for dir_path, sub_dirs, files in os.walk(REDACTIONS_DIR, topdown=True):
        for file in files:
            if file[-4:] == '.jpg':
                print file
                barcode, page, index, width, height = file.split('-')
                if start and barcode >= start:
                    redaction = {
                        'image': file,
                        'barcode': barcode,
                        'page': int(page[1:]),
                        'index': int(index),
                        'width': int(width),
                        'height': int(height[:-4]),
                        'random_id': random.random(),
                        'random_sample': [random.random(), 0]
                    }
                    item = db.items.find_one({'identifier': barcode})
                    redaction['series'] = item['series']
                    redaction['control_symbol'] = item['control_symbol']
                    try:
                        redaction['year'] = int(item['contents_dates']['start_date'][:4])
                    except (ValueError, IndexError, TypeError):
                        pass
                    page_path = '{}/{}-[{}]'.format(quote_plus(item['series'].replace('/', '-')), quote_plus(item['control_symbol'].replace('/', '-')), barcode)
                    page_image_url = '/{}/{}-{}.jpg'.format(page_path, barcode, page)
                    redaction['page_image_url'] = page_image_url
                    db.redactions.replace_one({'image': file}, redaction, upsert=True)

