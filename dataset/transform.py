from timm.data import RandAugment, rand_augment_ops
import cv2
from PIL import Image
import random
import numpy as np


class transform():
    def __init__(self, dataset = "tusimple") -> None:
        self.transforms = [
            'AutoContrast',
            'Equalize',
            # 'Invert',
            'Posterize',
            'Solarize',
            'SolarizeAdd',
            'Color',
            'Contrast',
            'Brightness',
            'Sharpness', 
            'PosterizeIncreasing',
            'SolarizeIncreasing',
            'ColorIncreasing',
            'ContrastIncreasing',
            'BrightnessIncreasing',
            'SharpnessIncreasing',
            #'Cutout'  # NOTE I've implement this as random erasing separately
        ]
        self.augment = RandAugment(rand_augment_ops(transforms=self.transforms), num_layers=3)
        self.width = 1280 if dataset == 'tusimple' else 1640

    def __call__(self, image, lanes, vp, y, train):
        if(train):
            image = self.add_noise(image)
            image = self.randAugment(image)
            image, lanes, vp, y = self.flip_image(image, lanes, vp, y)
            image, lanes, vp = self.translate_x(image, lanes, vp)
            # image = self.AID(np.array(image), lanes, y)
            # image, lanes, vp = self.scale(image, lanes, vp)
            image = self.RandomErasing(np.array(image), probability=0.25, erasing_num = 1)
            
        else:
            image = np.array(image)
        
        return image, list(lanes), vp, list(y)

    def randAugment(self, image):
        return self.augment(image)
    
    def add_noise(self, img):
        if(random.random() > 0.5):
            return img
        img = np.array(img)
        # 随机使用不同的噪声
        noise_choice = ["sp_noise", 
                        # "gaussian_noise", 
                        # "poisson", 
                        # "speckle"
                        ]
        noise = random.choice(noise_choice)

        # 高斯噪声
        if noise == "gaussian_noise":
            mean = 0
            sigma = random.randint(10, 20)

            gauss = np.random.normal(mean, sigma, img.shape)
            noisy_img = img + gauss
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        # 椒盐噪声
        elif noise == "sp_noise":
            s_vs_p = 0.5
            amount = 0.02

            noisy_img = np.copy(img)
            num_salt = int(amount * img.size * s_vs_p)
            #设置添加噪声的坐标位置
            coords = [np.random.randint(0,i - 1, int(num_salt)) for i in img.shape]
            noisy_img[coords[0],coords[1],:] = [255,255,255]
            num_pepper = int(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in img.shape]
            noisy_img[coords[0],coords[1],:] = [0,0,0]

        elif noise == "poisson":
            noisy_poisson = np.random.poisson(lam=5, size=img.shape).astype(np.uint8)
            noisy_img = noisy_poisson + img

        elif noise == "speckle":
            gauss = np.random.randn(*img.shape)
            noisy_img = img + img * gauss
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_img)


    def RandomErasing(self, image, probability=1, min_area=0.02, max_area=0.1, length_ratio=0.33, erasing_value='random', erasing_num = 1):
        if random.uniform(0, 1) > probability:
            return image
        
        image_size = image.shape
        area = image_size[0] * image_size[1]
        
        for i in range(erasing_num):
            target_area = random.uniform(min_area, max_area) * area
            
            aspect_ratio = random.uniform(length_ratio, 1 / length_ratio)
            height_erasing = int(round(np.sqrt(target_area * aspect_ratio)))
            width_erasing = int(round(np.sqrt(target_area / aspect_ratio)))
        
            if height_erasing < image_size[0] and width_erasing < image_size[0]:
                x1 = random.randint(0, image_size[0] - height_erasing)
                y1 = random.randint(0, image_size[1] - width_erasing)
            # 高斯分布
                if  erasing_value == 'random':
                    erasing_value_matrix = np.random.normal(0,1,(height_erasing,width_erasing,3)) * 255
                else:
                    erasing_value_matrix = erasing_value

                image[x1 : x1+height_erasing, y1:y1 + width_erasing,:] = erasing_value_matrix

        return image
    
    def flip_image(self, image, lanes, vp, y):
        # 左右翻转
        if(random.random() > 0.5):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            vp[1:] = self.width - vp[1:]
            lanes = np.flip(lanes, 0)
            lanes[lanes>0] = self.width - lanes[lanes>0]

        # 上下翻转
        if(random.random() > 1):
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            y = 720 - y
            vp[0] = 720 - vp[0]
    
        return image, lanes, vp, y

    def AID(self, image, lanes, y):
        start_point = []
        for i in range(len(lanes)):
            if(len(lanes[i]) == 48):
                lanes_i = list(np.append(np.zeros(8), np.array(lanes[i])))
            else:
                lanes_i = lanes[i]
            lane = list(filter(lambda x : x[0]>0, zip(lanes_i, y)))
            if (lane != []):
                start_point.append(lane[-1])
 
        start_point = random.choices(start_point, k = 2)
        for i in range(2):
            cv2.circle(image, (np.array(start_point[i]).astype(int)), 30, (128, 128, 128), -1)
        
        return image
    
    def translate_x(self, image, lanes, vp):
        if(random.random() > 0):
            pixels = random.randint(-200, 200)
            image = image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor = (128, 128, 128))
            lanes[lanes>0] = lanes[lanes>0] - pixels
            lanes[lanes>self.width] = -2
            vp[1:] = vp[1:] - pixels
        
        if(random.random() > 1):
            pixels = random.randint(-2, 2)
            image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels * 10), fillcolor = (128, 128, 128))
            lanes = np.roll(lanes, -pixels, 1)
            vp[0] -= pixels * 10
            vp[1:] = np.roll(vp[1:], pixels, 0)

            if(pixels < 0):
                lanes[:, :-pixels] = -2
                vp[1:][:-pixels] = 640
            else:
                lanes[:, -pixels:] = -2
                vp[1:][:pixels] = 640
        
        return image, lanes, vp
    
    def scale(self, image, lanes, vp):
        image = Image.fromarray(image)
        width, height = image.size
        scale = 1 / 0.6
        delta_x = width * (1 - scale)/2
        delta_y = (720 - vp[0]) * (1 - scale)/2
        # lanes = affine(lanes, 0, (0, vp[0] -720), 0.6, (0, 0), height, width)

        image = image.transform(image.size, Image.AFFINE, (scale, 0, delta_x, 0, scale, delta_y), fillcolor = (128, 128, 128))
        new_vp = np.zeros(55)
        start_index = int((710 - 710 * 0.6)/10)
        for i in range(start_index, 55):
            cur_pos = 700 - i * 10
            a = int(cur_pos/0.6 % 10)
            b = int(cur_pos/0.6 / 10)
            vp[i] = a * vp[b] + (1 - a) * vp[b + 1]

        return np.array(image), lanes, vp

if __name__ == '__main__':
    image = Image.open(r'D:\Code\pytorch-auto-drive-master\dataset\tusimple\clips\0313-1\\60\\1.jpg')
    image = transform()(image)
    
    cv2.imshow('img', image)
    cv2.waitKey(0)