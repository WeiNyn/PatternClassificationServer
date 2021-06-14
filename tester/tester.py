import requests
from dataset import TesterDataset
from nonmask_dataset import NonMaskDataset
from tqdm import tqdm
import json

# dataset = TesterDataset(root_folder='/data/part_detect_smart_factory/result_gen_fabric_shape/')
# dataset = NonMaskDataset(root_folder='/data/part_detect_smart_factory/120_test_pattern_12_5_2021_cut/')
dataset = NonMaskDataset(root_folder='/data/container_testing/smart_factory_demo/IMGS_0706_crop')

true_count = 0
count = 0
false_cache = dict()

url = 'http://10.0.11.69:8890/demo-path'

headers = {
    'Content-Type': 'application/json'
}

for _ in range(5):
    for i in tqdm(range(len(dataset))):
        image, mask, label = dataset[i]
        
        payload = dict(
            image=image,
            # image_mask=mask,
            # output_dir='/data/HuyNguyen/Demo/output_dir/'
        )
        
        response = requests.request(
            'POST', url, headers=headers, data=json.dumps(payload)
        )
            
        if response.ok:
            count += 1
            result = json.loads(response.text)
            predicted_label = result['results']['pattern_id']
            
            if int(predicted_label) == label:
                true_count += 1
            
            else:
                false_cache[label] = false_cache.get(label, dict())
                false_cache[label][predicted_label] = false_cache[label].get(predicted_label, 0) + 1
                
        else:
            print(response.text)
print(count)
print(true_count)
print(float(true_count)/float(count))
print(false_cache)