import io
import numpy as np
import matplotlib.cm as cm
from PIL import Image


def convert_array_to_PIL(myarray, cmap=cm.viridis):
    # Normalize the array to the range [0, 1]
    myarray = (myarray - np.min(myarray)) / (np.max(myarray) - np.min(myarray))

    # Apply the colormap to the normalized array
    myarray = cmap(myarray)

    # convert to PIL image for later download
    myarray = Image.fromarray(np.uint8(myarray * 255))

    return myarray


def convert_PIL_to_io(img, img_format="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=img_format)
    byte_im = buf.getvalue()
    return byte_im
