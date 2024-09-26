import json
import base64
import io

import redis
from PIL import Image


def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')


def send_tasks(image_paths, prompt):
    r = redis.Redis(host='localhost', port=8866)
    for image_path in image_paths:
        #image to base64
        base64_string = image_to_base64(image_path)

        task = {'image': base64_string, 'prompt': prompt}
        r.lpush('encode_tasks', json.dumps(task))

def listen_for_results()
    r = redis.Redis(host='localhost', port=8866)
    while True:
        _, result = r.blpop('finished_images')
        data = json.loads(result)
        image_data = base64.b64decode(data['image'])
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
    
        if 'command' in data:
            if data['command'] == 'exit':
                print("Exiting listener...")
                break

send_tasks(["path_to_image1.jpg", "path_to_image2.jpg"], "xxxxxxxxx")
listen_for_results()