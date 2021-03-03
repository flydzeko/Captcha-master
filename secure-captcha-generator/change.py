import numpy as np
from PIL import ImageFilter
import math
from PIL import Image
import  random

def change(image):
    '''
    temp_file = "output/demo/temp_img/temp.jpg"
    temp_file1 = "output/demo/temp_img/temp1.jpg"
    image.save(temp_file)
    img = cv2.imread('output/demo/temp_img/temp.jpg', cv2.IMREAD_COLOR)
    '''
    # 进行仿射变换
    width,height =image.size
    # image = image.transform((width + 20, height + 10), Image.AFFINE, (1, -0.3, 0, -0.1, 1, 0), Image.BILINEAR)
    image = image.transform((width + 25, height + 15), Image.AFFINE, (1.3,-0.4, 0, -0.2, 0.8, 0), Image.BILINEAR)
    res = image.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 滤镜，边界加强
    res = image
    '''
    cv2.imwrite('output/demo/temp_img/temp1.jpg',img_output)
    res = Image.open(temp_file1)
    '''
    return res