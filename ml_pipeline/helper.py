import os

def abs_image_path(img_name):
    """
    Constructs the absolute path to an image file located in the 'assets' directory
    of the project.

    Args:
        img_name (str): The name of the image file.

    Returns:
        str: The absolute path to the image file.
    
    Example:
        >>> abs_image_path('logo.png')
        '/path/to/project/assets/logo.png'
    """
    # Get the current directory where this script is located
    current_dir = os.path.dirname(__file__)

    # Navigate one directory up from the current directory
    out_dir = os.path.dirname(current_dir)

    # Join the parent directory with the 'assets' directory
    out_dir_2 = os.path.join(out_dir, 'assets')

    # Construct the absolute path to the image file in the 'assets' directory
    img_path = os.path.join(out_dir_2, img_name)

    return img_path
