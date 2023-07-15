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
