import os
import cv2
# import cv2.cv as cv
import tesserocr
from PIL import Image
import numpy as np
import glob
from math import floor

from pymongo import MongoClient, ASCENDING, DESCENDING
from urllib import quote_plus
from credentials import MONGOLAB_URL


def get_db():
    dbclient = MongoClient(MONGOLAB_URL)
    db = dbclient.get_default_database()
    return db


rootdir = '/Users/tim/mycode/asiofiles/share/A6119/images'
FACES_DIR = '/Users/tim/mycode/asio-timeline/src/processing/faces'
CROP_DIR = '/Users/tim/mycode/asio-timeline/src/processing/crops'
FACE_CLASSIFIER = '/usr/local/Cellar/opencv/2.4.12/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
TEXT_DIR = '/Users/tim/mycode/asio-timeline/src/processing/text'
IMAGE_DIR = '/Users/tim/mycode/asiofiles/share/{}/images'
REDACTIONS_DIR = '/Users/tim/mycode/asio-timeline/src/processing/redactions'


# EXPERIMENTS WITH PHOTOS

def extract_faces():
    face_cl = cv2.CascadeClassifier(FACE_CLASSIFIER)
    crop_file = '{}/{}-{}.jpg'
    for root, dirs, files in os.walk(rootdir, topdown=True):
        for dir in dirs:
            print dir
            for dir_path, sub_dirs, files in os.walk(os.path.join(root, dir), topdown=True):
                for file in files:
                    if file[-3:] == 'jpg':
                        f = 1
                        print 'Processing {}'.format(file)
                        try:
                            image = cv2.imread(os.path.join(dir_path, file), 0)
                            faces = face_cl.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))
                            print faces
                        except cv2.error:
                            pass
                        else:
                            for (x, y, w, h) in faces:
                                face = image[y:y+h, x:x+w]
                                fn = crop_file.format(FACES_DIR, os.path.splitext(os.path.basename(file))[0], f)
                                cv2.imwrite(fn, face)
                                f += 1


def find_photos():
    crop_file = '{}/{}-{}.jpg'
    for root, dirs, files in os.walk(rootdir, topdown=True):
        for dir in dirs:
            print dir
            for dir_path, sub_dirs, files in os.walk(os.path.join(root, dir), topdown=True):
                for file in files:
                    if file[-3:] == 'jpg':
                        image = cv2.imread(os.path.join(dir_path, file))
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        #gray = cv2.GaussianBlur(gray, (3, 3), 0)
                        edged = cv2.Canny(gray, 10, 250)
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
                        (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        rects = []
                        for c in cnts:
                            # approximate the contour
                            peri = cv2.arcLength(c, True)
                            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                            # if the approximated contour has four points, then assume that the
                            # contour is a book -- a book is a rectangle and thus has four vertices
                            if len(approx) == 4:
                                # cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
                                rects.append(cv2.boundingRect(c))
                            for index, rect in enumerate(rects):
                                x, y, width, height = rect
                                if width > 200 and height > 200:
                                    crop = image[y:y+height, x:x+width]
                                    fn = crop_file.format(CROP_DIR, os.path.splitext(os.path.basename(file))[0], index)
                                    cv2.imwrite(fn, crop)

    # cv2.imshow("Output", image)


# EXPERIMENTS WITH OCR


def find_forms():
    test_words = ['folio', 'archives', 'exemption']
    for root, dirs, files in os.walk(rootdir, topdown=True):
        for dir in dirs:
            count = 0
            for dir_path, sub_dirs, files in os.walk(os.path.join(root, dir), topdown=True):
                for file in files:
                    if file[-3:] == 'jpg':
                        image = Image.open(os.path.join(dir_path, file))
                        ocr = tesserocr.image_to_text(image)
                        text = ocr.lower()
                        for test in test_words:
                            if test in text:
                                image.save(os.path.join('forms', file))
                                count += 1
                                break
            print '{}: {}'.format(dir, count)


def extract_text(series):
    text_dir = os.path.join(TEXT_DIR, series)
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)
    image_dir = IMAGE_DIR.format(series)
    for root, dirs, files in os.walk(image_dir, topdown=True):
        for dir in dirs:
            print dir
            text_path = os.path.join(text_dir, dir)
            if not os.path.exists(text_path):
                os.makedirs(text_path)
            for dir_path, sub_dirs, files in os.walk(os.path.join(root, dir), topdown=True):
                for file in files:
                    if file[-3:] == 'jpg':
                        text_file = os.path.join(text_path, '{}.txt'.format(file[:-4]))
                        if not os.path.exists(text_file):
                            image = Image.open(os.path.join(dir_path, file))
                            ocr = tesserocr.image_to_text(image)
                            with open(text_file, 'wb') as ocr_file:
                                ocr_file.write(ocr.encode('utf-8'))


# EXPERIMENTS WITH REDACTIONS 


def find_redacted(start='0', series=None, crop=False, oddities=False, composite=False, details=False):
    '''
    I ran this in two stages. First to save copies of redactions.
    I then manually removed false positives, and ran again, using the (now sorted)
    redactions as a check.

    It was developed through trial and error, hence all the commented out bits.
    I've left them in as reminders of what I tried.

    crop -- saves cropped copies of the redactions as separate images
    oddities -- identifies and saves heavily redacted pages for inspection
    composite -- builds up heatmap type image of redaction positions
    details -- saves size and position of redactions to db
    '''

    if series:
        image_dir = IMAGE_DIR.format(series)
    else:
        image_dir = 'tests'
    if composite:
        # comp_image = Image.new('RGB', (2400, 3200), 'white')
        comp_image = np.zeros((3200, 2400, 3), np.uint16)
        comp_image[:] = (65535, 65535, 65535)
        comp_count = 0
    for dir_path, sub_dirs, files in os.walk(image_dir, topdown=True):
        for file in files:
            # load the image
            if file[-4:] == '.jpg' and 'test' not in file:
                file_parts = file[:-4].split('-')
                barcode = file_parts[0]
                page = int(file_parts[1][1:])
                if barcode >= start:
                    orig_image = cv2.imread(os.path.join(dir_path, file))
                    oh, ow = orig_image.shape[:2]
                    if ow > 1200:
                        ih = int((1200.00 / ow) * oh)
                        iw = 1200
                        ratio = ow / 1200.0
                        image = cv2.resize(orig_image, (iw, ih), interpolation=cv2.INTER_AREA)
                    else:
                        image = orig_image.copy()
                        ratio = 1
                        iw = ow
                        ih = oh
                    image_area = iw * ih
                    # ret, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
                    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    # import the necessary packages
                    # lower = np.array([0, 0, 0])
                    # upper = np.array([30, 30, 30])
                    # shapeMask = cv2.inRange(image, 0, 60)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray = cv2.bilateralFilter(gray, 15, 20, 20)
                    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    # gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    ret, gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                    edged = cv2.inRange(gray, 0, 30)
                    # edged = cv2.Canny(gray, 0, 30)
                    # edged = auto_canny(gray)
                    # find the contours in the mask
                    _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # cv2.imshow("Mask", shapeMask)
                    # image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                    # loop over the contours
                    redacted = 0
                    count = 0
                    redactions = []
                    for cnt in contours:
                        # draw the contour and show it
                        area = cv2.contourArea(cnt)
                        if area > 1000 and area < 1000000:
                            # peri = cv2.arcLength(cnt, True)
                            # approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                            # print len(approx)
                            # if len(approx) < 12:
                            # Don't really need to rectabgles just the centres, use moments instead
                            # moments = cv2.moments(approx)
                            # x = int(moments["m10"] / moments["m00"])
                            # y = int(moments["m01"] / moments["m00"])
                            rect = cv2.minAreaRect(cnt)
                            x = int(rect[0][0])
                            y = int(rect[0][1])
                            # box = cv2.boxPoints(rect)
                            # box = np.int0(box)
                            # cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)
                            sample = gray[y - 5:y + 5, x - 5:x + 5]
                            no_rows = sample.shape[0]
                            no_cols = sample.shape[1]
                            black = True
                            for row in range(no_rows):
                                for col in range(no_cols):
                                    if sample[row, col] != 0:
                                        black = False

                            if black:
                                count += 1
                                rx, ry, rw, rh = cv2.boundingRect(cnt)
                                image_file = '{}-{}-{}-{}.jpg'.format(file[:-4], count, rw, rh)
                                redaction = os.path.join(REDACTIONS_DIR, image_file)
                                if os.path.exists(redaction):
                                    if not (rx < 200 and ry < 200):  # Try to exclude holes in the corners
                                        if not (rx < 20 or ry < 20 or rx > (iw - 20) or ry > (ih - 20) or rw > 1100 or rh > 900):
                                            if crop:
                                                r_image = image[ry - 10:ry + rh + 10, rx - 10:rx + rw + 10]
                                                cv2.imwrite(redaction, r_image)
                                            if composite:
                                                c_ratio = 2400.0 / iw
                                                bx = int(floor(rx * c_ratio))
                                                by = int(floor(ry * c_ratio))
                                                bw = int(floor(rw * c_ratio))
                                                bh = int(floor(rh * c_ratio))
                                                overlay = np.zeros((3600, 2400, 3), np.uint16)
                                                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (10, 10, 10), -1)
                                                comp_image = cv2.subtract(comp_image, overlay)
                                                # cv2.addWeighted(overlay, 0.01, comp_image, 0.99, 0, comp_image)
                                                # mask = Image.new('L', box_size, 1)
                                                # rect = Image.new('RGBA', box_size, 0)
                                                # rect.putalpha(mask)
                                                # comp_image.paste(rect, (int(floor(rx * ratio)), int(floor(ry * ratio))), rect)
                                                comp_count += 1
                                                print comp_count
                                                if comp_count in [c for c in range(1000, 250000, 1000)]:
                                                    # comp_image.save(os.path.join('composites', 'composite-{}.jpg'.format(comp_count)))
                                                    cv2.imwrite(os.path.join('composites2', 'composite-{}.png'.format(comp_count)), comp_image)
                                            if details:
                                                db = get_db()
                                                # Save info to redactions and page entries
                                                # Because I stupidly resized the image I need to convert coords back for original size.
                                                ox = int(floor(rx * ratio))
                                                oy = int(floor(ry * ratio))
                                                ow = int(floor(rw * ratio))
                                                oh = int(floor(rh * ratio))
                                                # cv2.rectangle(orig_image, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 3)
                                                # cv2.imwrite(os.path.join('details', file), orig_image)
                                                position = [ox, oy, ow, oh]
                                                db.redactions.update_one({'image': image_file}, {'$set': {'position': position, 'area': area}})
                                                redactions.append(image_file)
                                            redacted += area
                                            # redactions.append(cnt)
                                            # cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
                    # new_image = os.path.join('testoddities', file)
                    # cv2.imwrite(new_image, image)
                    percentage = (float(redacted) / image_area) * 100
                    if oddities and percentage > 10:
                        new_image = os.path.join('oddities', file)
                        cv2.imwrite(new_image, image)
                    if details and redactions:
                        total_redacted = {'total': count, 'redactions': redactions, 'area': redacted, 'percentage': percentage}
                        db.images.update_one({'identifier': barcode, 'page': page}, {'$set': {'redacted': total_redacted}})
                    print '{}: {} of {}, {}'.format(file, redacted, image_area, percentage)


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def remove_redactions():
    db = get_db()
    db.images.update_many({}, {'$unset': {'redacted': 1}})


def update_redacted():
    '''
    This is basically the same as find_redacted() with details=True.
    The whole thing takes quite a while, so I needed a way of restarting if something failed.
    This looks to see what's already in the db rather than starting from scratch.
    '''

    db = get_db()
    pipeline = [
        {
            '$match': {'position': {'$exists': False}}
        },
        {
            '$group': {
                '_id': '$page_image_url'
            }
        }
    ]
    redactions = db.redactions.aggregate(pipeline).batch_size(20)
    for path in redactions:
        page_image = rootdir + path['_id'].replace('/A6119', '').replace('+', ' ').replace('%5B', '[').replace('%5D', ']')
        file = page_image.split('/')[-1]
        details = file.split('-')
        barcode = details[0]
        page = details[1][1:-4]
        orig_image = cv2.imread(page_image)
        oh, ow = orig_image.shape[:2]
        if ow > 1200:
            ih = int((1200.00 / ow) * oh)
            iw = 1200
            ratio = ow / 1200.0
            image = cv2.resize(orig_image, (iw, ih), interpolation=cv2.INTER_AREA)
        else:
            image = orig_image.copy()
            ratio = 1
            iw = ow
            ih = oh
        image_area = iw * ih
        # ret, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
        # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # import the necessary packages
        # lower = np.array([0, 0, 0])
        # upper = np.array([30, 30, 30])
        # shapeMask = cv2.inRange(image, 0, 60)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 15, 20, 20)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        ret, gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        edged = cv2.inRange(gray, 0, 30)
        # edged = cv2.Canny(gray, 0, 30)
        # edged = auto_canny(gray)
        # find the contours in the mask
        _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("Mask", shapeMask)
        # image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # loop over the contours
        redacted = 0
        count = 0
        redactions = []
        for cnt in contours:
            # draw the contour and show it
            area = cv2.contourArea(cnt)
            if area > 1000 and area < 1000000:
                # peri = cv2.arcLength(cnt, True)
                # approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                # print len(approx)
                # if len(approx) < 12:
                # Don't really need to rectabgles just the centres, use moments instead
                # moments = cv2.moments(approx)
                # x = int(moments["m10"] / moments["m00"])
                # y = int(moments["m01"] / moments["m00"])
                rect = cv2.minAreaRect(cnt)
                x = int(rect[0][0])
                y = int(rect[0][1])
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                # cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)
                sample = gray[y - 5:y + 5, x - 5:x + 5]
                no_rows = sample.shape[0]
                no_cols = sample.shape[1]
                black = True
                for row in range(no_rows):
                    for col in range(no_cols):
                        if sample[row, col] != 0:
                            black = False
                if black:
                    count += 1
                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                    image_file = '{}-{}-{}-{}.jpg'.format(file[:-4], count, rw, rh)
                    redaction = os.path.join(REDACTIONS_DIR, image_file)
                    if os.path.exists(redaction):
                        if not (rx < 200 and ry < 200):  # Try to exclude holes in the corners
                            if not (rx < 20 or ry < 20 or rx > (iw - 20) or ry > (ih - 20) or rw > 1100 or rh > 900):
                                db = get_db()
                                # Save info to redactions and page entries
                                # Because I stupidly resized the image I need to convert coords back for original size.
                                ox = int(floor(rx * ratio))
                                oy = int(floor(ry * ratio))
                                ow = int(floor(rw * ratio))
                                oh = int(floor(rh * ratio))
                                # cv2.rectangle(orig_image, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 3)
                                # cv2.imwrite(os.path.join('details', file), orig_image)
                                position = [ox, oy, ow, oh]
                                # print position
                                db.redactions.update_one({'image': image_file}, {'$set': {'position': position, 'area': area}})
                                redactions.append(image_file)
                                redacted += area
                                # redactions.append(cnt)
                                # cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
        # new_image = os.path.join('testoddities', file)
        # cv2.imwrite(new_image, image)
        percentage = (float(redacted) / image_area) * 100
        if redactions:
            total_redacted = {'total': count, 'redactions': redactions, 'area': redacted, 'percentage': percentage}
            # print total_redacted
            db.images.update_one({'identifier': barcode, 'page': page}, {'$set': {'redacted': total_redacted}})
        print '{}: {} of {}, {}'.format(file, redacted, image_area, percentage)
