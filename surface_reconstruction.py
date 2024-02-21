

import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PImage
import plotly.graph_objects as go
from plotly import tools


def create_3d_surface(rgb_img, depth_img, paths=False, opacity=1.0, depth_cutoff=20, **kwargs):
    if paths: rgb_img, depth_img = read_images(rgb_img, depth_img)
    rgb_img = rgb_img.swapaxes(0, 1)[:, ::-1]
    depth_img = depth_img.swapaxes(0, 1)[:, ::-1]
    eight_bit_img = PImage.fromarray(rgb_img).convert('P', palette='WEB', dither=None)
    idx_to_color = np.array(eight_bit_img.getpalette()).reshape((-1, 3))
    colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
    depth_map = depth_img.copy().astype('float')
    depth_map[depth_map<depth_cutoff] = np.nan
    fig = go.Surface(
        z=depth_map,
        surfacecolor=np.array(eight_bit_img),
        cmin=0, 
        cmax=255,
        colorscale=colorscale,
        showscale=False,
        contours_z=dict(show=True, project_z=True, highlightcolor="limegreen"),
        opacity=opacity
        )
    fig = go.Figure(data=[fig],
                    layout_title_text="3D Surface"
                    )
    fig.update_layout(autosize=False, width=1000, height=1000)
    return fig

def read_images(image_path, depth_path):
    img = skimage.io.imread(image_path)
    d = skimage.io.imread(depth_path)
    return (img, d)