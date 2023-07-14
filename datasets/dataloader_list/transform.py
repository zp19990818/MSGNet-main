import random
from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    # def __call__(self, img, mask, depth, flow):
    def __call__(self, img, mask, flow):
        assert img.size == mask.size
        # assert img.size == depth.size
        assert img.size == flow.size
        for t in self.transforms:
        #     img, mask, depth, flow = t(img, mask, depth, flow)
        # return img, mask, depth, flow
              img, mask, flow = t(img, mask,flow)
        return img, mask, flow

class RandomHorizontallyFlip(object):
    # def __call__(self, img, mask, depth, flow):
    def __call__(self, img, mask, flow):
        if random.random() < 0.5:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
                # depth.transpose(Image.FLIP_LEFT_RIGHT),
                flow.transpose(Image.FLIP_LEFT_RIGHT),
            )
        # return img, mask, depth, flow
        return img, mask, flow



class JointResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise RuntimeError("size参数请设置为int或者tuple")

    # def __call__(self, img, mask, depth, flow):
    def __call__(self, img, mask, flow):
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        # depth = depth.resize(self.size, Image.BILINEAR)
        flow = flow.resize(self.size, Image.BILINEAR)
        return img, mask, flow

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    # def __call__(self, img, mask, depth, flow):
    def __call__(self, img, mask, flow):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            img.rotate(rotate_degree, Image.BILINEAR),
            mask.rotate(rotate_degree, Image.NEAREST),
            # depth.rotate(rotate_degree, Image.BILINEAR),
            flow.rotate(rotate_degree, Image.BILINEAR),)