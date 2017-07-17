import numpy as np
from PIL import Image


def tile_weights(X, img_shape, tile_shape, tile_spacing=(0, 0)):
    columns = img_shape[0]*img_shape[1]
    rows = tile_shape[0]*tile_shape[1]
    shape = X.shape
    data = np.pad(X, ((0, rows-shape[0]), (0, columns-shape[1])), mode='constant', constant_values=((0, 0), (0, 0)))
    out = np.zeros(((img_shape[0]+tile_spacing[0])*tile_shape[0]-tile_spacing[0], (img_shape[1]+tile_spacing[1])*tile_shape[1]-tile_spacing[1]))
    for x, y in np.ndindex(tile_shape):
        img = data[y*tile_shape[0]+x].reshape(img_shape)
        out[x*(img_shape[0]+tile_spacing[0]):x*(img_shape[0]+tile_spacing[0])+img_shape[0],
            y * (img_shape[1] + tile_spacing[1]):y * (img_shape[1] + tile_spacing[1]) + img_shape[1]]=img
    return out.astype(np.uint8)


def save_weights(file, weights, shape=None, tile=None, spacing=(1,1)):
    """
    Saves weights as tiled image, where each tile represents hidden units weight. If number of hidden units or number
    of visible units doesn't fit into shape do automatic padding with black pixels. Automatically scales weights to
    (0,255) range.
    :param file: string, file name to save image
    :param weights: 2d-array, weights matrix
    :param shape: tuple, image shape
    :param tile: tiles shape
    :param spacing: spacing between tiles
    """
    weights =  np.transpose(weights)
    current_min = weights.min()
    current_max = weights.max()
    weights = 255 * (weights - current_min) / (current_max - current_min)
    image = Image.fromarray(
        tile_weights(
            X=weights,
            img_shape=shape,
            tile_shape=tile,
            tile_spacing=spacing
        )
    )
    image.save(file)


def save_hidden_state(file, hidden_state):
    """
    Just save hidden_state matrix to file as image
    :param file: string, file name to save image
    :param hidden_state: 2d array, hidden unit states across batch
    """
    hidden_state =  np.transpose(hidden_state)
    current_min = hidden_state.min()
    current_max = hidden_state.max()
    hidden_state = 255 * (hidden_state - current_min) / (current_max - current_min)
    image = Image.fromarray(hidden_state.astype(np.uint8))
    image.save(file)


weights = np.random.uniform(size=(230, 189))

save_weights("./test_r.jpg", weights, shape=(16,16), tile=(20, 10))