from PIL import Image as i
import os
from resizeimage import resizeimage as rs

def image_format_converter(directory):
    print("Converting all images to .jpg.........")
    j=0
    for filename in os.listdir(directory):
        j = j + 1
        file=i.open(directory+filename)
        new_width = 30
        new_height = 30
        file = file.resize((new_width, new_height), i.ANTIALIAS)
        file.convert("RGB").save(directory+"image%d.jpg"%j,"JPEG")
        file.close()
        os.remove(directory+filename)
    print("Conversion complete...\nFormatted %d images."%j)
    pass

if __name__=="__main__":
    image_format_converter("img_datasets/bat_test/")
    image_format_converter("img_datasets/sup_test/")