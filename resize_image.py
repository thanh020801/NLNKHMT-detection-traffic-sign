from PIL import Image
import numpy as np

for i in range(49):
    img = Image.open('origin_data/1/0001_'+str(i)+'.jpg')
    img = img.convert("RGBA")
    resize_img = img.resize((210,210))
    resize_img.save('resize_data/1/0000_'+ str(i)+ '.png')
    # print(img.size)
