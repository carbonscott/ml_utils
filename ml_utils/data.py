import numpy as np

from skimage.transform import resize

class Pad:
    def __init__(self, size_y, size_x):
        self.size_y = size_y
        self.size_x = size_x

    def __call__(self, img):
        size_y = self.size_y
        size_x = self.size_x

        size_img_y, size_img_x = img.shape[-2:]

        dy_padded = max(size_y - size_img_y, 0)
        dx_padded = max(size_x - size_img_x, 0)
        pad_width = (
            (0, 0),
            (dy_padded // 2, dy_padded - dy_padded // 2),
            (dx_padded // 2, dx_padded - dx_padded // 2),
        )

        # Calculate the offset of the top left corner after padding
        top_left_offset = (dy_padded // 2, dx_padded // 2)

        img_padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)

        # Return both the padded image and the offset
        return img_padded, top_left_offset


class Resize:
    def __init__(self, size_y, size_x, anti_aliasing = True):
        self.size_y        = size_y
        self.size_x        = size_x
        self.anti_aliasing = anti_aliasing


    def __call__(self, img):
        size_y        = self.size_y
        size_x        = self.size_x
        anti_aliasing = self.anti_aliasing

        B = img.shape[0]

        img_resize = resize(img, (B, size_y, size_x), anti_aliasing = anti_aliasing)

        return img_resize


def generate_sample_data():
    import textwrap

    code = '''
    B, C, H, W = 10, 1, 1920, 1920
    batch_input = np.random.rand(B, C, H, W)

    H_unify, W_unify = 1024, 1024
    resizer = Resize(H_unify, W_unify)

    batch_input_unify = resizer(batch_input.reshape(B*C, H, W)).reshape(B, C, H_unify, W_unify)
    batch_input_unify_tensor = torch.from_numpy(batch_input_unify).to(torch.float)
    '''
    print(textwrap.dedent(code))
