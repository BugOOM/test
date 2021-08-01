import os
import cv2
import json
import xml.dom.minidom
import xml.etree.ElementTree as ET
import shutil

coco_json_dir = './cocosplit/datasplit'
image_sets_dir = './VOC2007/ImageSets/Main'
coco_train_image_dir = './coco/trainval2014'
coco_test_image_dir = './coco/test2014'
if not os.path.exists(coco_json_dir):
    os.makedirs(coco_json_dir)

MODE = 'train'
if MODE == 'train':
    imgs = open(os.path.join(image_sets_dir, 'trainval.txt'), 'r').read().split('\n')
    img_save_dir = coco_train_image_dir
    f = open(os.path.join(coco_json_dir, 'trainvalno5k.json'), 'w')
else:
    imgs = open(os.path.join(image_sets_dir, 'test.txt'), 'r').read().split('\n')
    img_save_dir = coco_test_image_dir
    f = open(os.path.join(coco_json_dir, '5k.json'), 'w')
xml_file_dir = './VOC2007/Annotations'
imgs_dir = './VOC2007/JPEGImages'
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir)
annotations_info = {'images': [], 'annotations': [], 'categories': []}

categories_map = {str(i): i for i in range(10)}

for key in categories_map:
    categoriy_info = {"id": categories_map[key], "name": key}
    annotations_info['categories'].append(categoriy_info)

ann_id = 1
for i, file_name in enumerate(imgs):

    image_file_name = file_name + '.jpg'
    xml_file_name = file_name + '.xml'
    image_file_path = os.path.join(imgs_dir, image_file_name)
    shutil.copy(image_file_path, os.path.join(img_save_dir, image_file_name))
    xml_file_path = os.path.join(xml_file_dir, xml_file_name)

    image_info = dict()
    image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    image_info = {'file_name': image_file_name, 'id': file_name, "image_id": file_name,
                  'height': height, 'width': width}
    annotations_info['images'].append(image_info)

    DOMTree = xml.dom.minidom.parse(xml_file_path)
    collection = DOMTree.documentElement

    names = collection.getElementsByTagName('name')
    names = [name.firstChild.data for name in names]

    xmins = collection.getElementsByTagName('xmin')
    xmins = [xmin.firstChild.data for xmin in xmins]
    ymins = collection.getElementsByTagName('ymin')
    ymins = [ymin.firstChild.data for ymin in ymins]
    xmaxs = collection.getElementsByTagName('xmax')
    xmaxs = [xmax.firstChild.data for xmax in xmaxs]
    ymaxs = collection.getElementsByTagName('ymax')
    ymaxs = [ymax.firstChild.data for ymax in ymaxs]

    object_num = len(names)

    for j in range(object_num):
        if names[j] in categories_map:
            x1, y1, x2, y2 = int(xmins[j]), int(ymins[j]), int(xmaxs[j]), int(ymaxs[j])
            x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 - 1, y2 - 1

            if x2 == width:
                x2 -= 1
            if y2 == height:
                y2 -= 1

            x, y = x1, y1
            w, h = x2 - x1 + 1, y2 - y1 + 1
            category_id = categories_map[names[j]]
            area = w * h
            annotation_info = {"id": ann_id, "image_id": file_name, "bbox": [x, y, w, h], "category_id": category_id,
                               "area": area, "iscrowd": 0}
            annotations_info['annotations'].append(annotation_info)
            ann_id += 1

json.dump(annotations_info, f, indent=4)
print('---整理后的标注文件---')
print('所有图片的数量：', len(annotations_info['images']))
print('所有标注的数量：', len(annotations_info['annotations']))
print('所有类别的数量：', len(annotations_info['categories']))
