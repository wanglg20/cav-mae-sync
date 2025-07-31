import csv
import json
import pandas as pd
import os


output_test_json = '/home/chenyingying/tmp/cav-mae-sync/src/data_info/vggsound_test.json'
output_train_json = '/home/chenyingying/tmp/cav-mae-sync/src/data_info/vggsound_train.json'
ori_csv = '/home/chenyingying/tmp/cav-mae-sync/src/data_info/vggsound.csv'

img_path = '/data/wanglinge/dataset/VGGSound/frames_16'
audio_path = '/data/wanglinge/dataset/VGGSound/audio'
train_img_path = os.path.join(img_path, 'train')
test_img_path = os.path.join(img_path, 'test')
train_audio_path = os.path.join(audio_path, 'train')
test_audio_path = os.path.join(audio_path, 'test')
raw_video_path = '/data/wanglinge/dataset/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video'
label_indices = '/home/chenyingying/tmp/cav-mae-sync/src/data_info/datafiles/class_labels_indices_vgg.csv'

indice_dict = {}
indice_info = pd.read_csv(label_indices)
indice_dict = dict(zip(indice_info['display_name'], indice_info['mid']))


input_filelist = pd.read_csv(ori_csv, 
                 names=['id', 'timestamp', 'label', 'split'],
                 header=None)
data_info = [(f"{row['id']}_{str(row['timestamp']).zfill(6)}", row['split'], row['label']) for _, row in input_filelist.iterrows()]

train_data = []
test_data = []
error_count = 0
error_label = set()

for video_id, split, label in data_info:
    video_path = os.path.join(raw_video_path, f"{video_id}.mp4")

    if split == 'train':
        img_dir = train_img_path
        audio_file = os.path.join(train_audio_path, f"{video_id}.wav")
        label = label.replace(',', '_')
        if label not in indice_dict:
            error_count += 1
            error_label.add(label)
            continue
        label = indice_dict[label]
        train_data.append({
            'video_id': video_id,
            'raw_video_path': video_path,
            'video_path': img_dir,
            'wav': audio_file,
            'labels': label
        })
    else:
        img_dir = test_img_path
        audio_file = os.path.join(test_audio_path, f"{video_id}.wav")
        label = label.replace(',', '_')
        if label not in indice_dict:
            error_count += 1
            error_label.add(label)
            continue
        label = indice_dict[label]
        test_data.append({
            'video_id': video_id,
            'raw_video_path': video_path,
            'video_path': img_dir,
            'wav': audio_file,
            'labels': label
        })

train_data_info = {}
train_data_info['data'] = train_data
test_data_info = {}
test_data_info['data'] = test_data

with open(output_train_json, 'w') as f:
    json.dump(train_data_info, f, indent=4)
with open(output_test_json, 'w') as f:
    json.dump(test_data_info, f, indent=4)
print(f"Total errors: {error_count}")
if error_count > 0:
    print(f"Error labels: {error_label}")
else:
    print("No errors found in labels.")

error_log = '/home/chenyingying/tmp/cav-mae-sync/src/data_info/vggsound_error_labels.txt'
with open(error_log, 'w') as f:
    for label in error_label:
        f.write(f"{label}\n")