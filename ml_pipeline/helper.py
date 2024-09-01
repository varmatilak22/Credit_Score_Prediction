import os

def abs_image_path(img_name):
    current_dir = os.path.dirname(__file__)
    out_dir = os.path.dirname(current_dir)
    out_dir_2 = os.path.join(out_dir, 'assets')
    img_path = os.path.join(out_dir_2, img_name)
    return img_path
