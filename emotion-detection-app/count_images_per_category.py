import os

# Base directory for FER2013 dataset
base_dir = os.path.join('data', 'fer2013')

# Emotion categories
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Dataset splits
splits = ['train', 'test']

def count_images():
    for split in splits:
        print(f'--- {split.upper()} ---')
        for emotion in emotions:
            dir_path = os.path.join(base_dir, split, emotion)
            if os.path.exists(dir_path):
                num_files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
                print(f'{emotion.capitalize():<10}: {num_files}')
            else:
                print(f'{emotion.capitalize():<10}: Directory not found')
        print()

if __name__ == '__main__':
    count_images()
