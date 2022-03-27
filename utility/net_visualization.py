import visualkeras
import tensorflow.keras.layers as layers
from collections import defaultdict
from PIL import ImageFont

from models import stixel_net


def main():
    model = stixel_net(visualize=True)

    color_map = defaultdict(dict)
    color_map[layers.Conv2D]['fill'] = 'white'
    color_map[layers.MaxPooling2D]['fill'] = 'gray'
    color_map[layers.Dropout]['fill'] = 'pink'
    color_map[layers.ELU]['fill'] = 'red'
    color_map[layers.Reshape]['fill'] = 'teal'

    # font = ImageFont.truetype("arial.ttf", 32)

    visualkeras.layered_view(model,
                             type_ignore=[layers.ELU, layers.Dropout],
                             color_map=color_map,
                             one_dim_orientation='y',
                             spacing=3,
                             scale_z=0.3,
                             scale_xy=2,
                             draw_funnel=False
                             ).show()

    model.summary()


if __name__ == '__main__':
    main()
