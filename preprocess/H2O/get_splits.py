"""
* Copyright (c) 2023 OPPO. All rights reserved.
*
*
* Licensed under the Apache License, Version 2.0 (the "License"):
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and 
* limitations under the License.
"""

import os


def get_split_info(split_file):
    sample_info = []
    with open(split_file, 'r') as f:
        for line in f.readlines():
            img_file = line.strip()
            subject, scene, obj = img_file.split('/')[:3]
            sub = subject.split('subject')[-1]
            info = 'sub{}_{}_{}'.format(sub, scene, obj)
            if info not in sample_info:
                sample_info.append(info)
    return sample_info


def write_split(info, filepath):
    with open(filepath, 'w') as f:
        for sample in info:
            f.writelines(sample + '\n')


if __name__ == '__main__':
    data_root = '../../data/H2O/ego'

    dest_dir = '../../data/H2O/Ego3DTraj/splits'
    os.makedirs(dest_dir, exist_ok=True)
    
    split_file = os.path.join(data_root, 'label_split', 'pose_train.txt')
    train_info = get_split_info(split_file)  # 114
    write_split(train_info, os.path.join(dest_dir, 'train.txt'))

    split_file = os.path.join(data_root, 'label_split', 'pose_val.txt')
    val_info = get_split_info(split_file)    # 24
    write_split(val_info, os.path.join(dest_dir, 'val.txt'))

    split_file = os.path.join(data_root, 'label_split', 'pose_test.txt')
    test_info = get_split_info(split_file)   # 46
    write_split(test_info, os.path.join(dest_dir, 'test.txt'))
    
    print("Train samples: {}, val samples: {}, test samples: {}".format(len(train_info), len(val_info), len(test_info)))
