import io
import numpy as np
import matplotlib.cm as cm
from PIL import Image


def convert_array_to_PIL(myarray, cmap=cm.viridis):
    """Converts a NumPy array to a PIL Image.

    This function normalizes the input array to the range [0, 1], applies a colormap,
    and converts the result to a PIL Image.

    Args:
        myarray: The input NumPy array.
        cmap: The colormap to apply. Defaults to `matplotlib.cm.viridis`.

    Returns:
        A PIL Image object.
    """
    # Normalize the array to the range [0, 1]
    myarray = (myarray - np.min(myarray)) / (np.max(myarray) - np.min(myarray))

    # Apply the colormap to the normalized array
    myarray = cmap(myarray)

    # convert to PIL image for later download
    myarray = Image.fromarray(np.uint8(myarray * 255))

    return myarray


def convert_PIL_to_io(img, img_format="PNG"):
    """Converts a PIL Image to a byte string for download.

    This function takes a PIL Image object and converts it to a byte string in the
    specified image format.

    Args:
        img: The PIL Image object to convert.
        img_format: The desired image format. Defaults to "PNG". Other supported
        formats depend on the capabilities of the Pillow library.

    Returns:
        A byte string containing the image data in the specified format.
    """
    buf = io.BytesIO()
    img.save(buf, format=img_format)
    byte_im = buf.getvalue()
    return byte_im
