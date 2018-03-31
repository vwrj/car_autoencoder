import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def is_cat(image_name):
    try:
        word_label = image_name.split('.')[-3]
    except IndexError:
        print(image_name)
    return word_label == 'cat'


def read_and_process_image(image_path, add_noise=False, gray_flag=False, img_size=32):
    if gray_flag:
        img_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img_data = cv2.imread(image_path)
    img_data = cv2.resize(img_data, (img_size, img_size))
    img_data = img_data.astype('float32') / 255.
    
    if add_noise:
        noise_factor = 0.05
        img_data = img_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=img_data.shape) 
        img_data = np.clip(img_data, 0., 1.)
        
    return img_data

def get_images_train(file, train_dir = None, refresh = False, add_noise = False, filter_func = None, gray_flag=False):
    
    if filter_func is None:
        def filter_func(a):
            return True
        
    if train_dir is None:
        train_dir = os.getcwd()
    
    p = os.path.join(os.getcwd(), file)
    if (not refresh and os.path.isfile(p)):
            train = np.load(file)
    else:
        train = []
        print("Fetching and processing images from directory: ", train_dir)
        for img in os.listdir(train_dir):
            if filter_func(img):  
                path = os.path.join(train_dir, img)
                img_data = read_and_process_image(path, add_noise, gray_flag)
                train.append(img_data)
        train = np.array(train)
        np.save(file, train)
        print('Saved training data into: ', file)
    print('Shape of training data:', train.shape)
    return train

def plot_side_by_side(img1, img2, titles = ["Image 1", "Image 2"]):
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(titles[0])

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(titles[1])
    plt.show()

def gray_scale(img):
    img = np.average(img, axis=2)
    return np.transpose([img, img, img], axes=[1,2,0])

    

    
