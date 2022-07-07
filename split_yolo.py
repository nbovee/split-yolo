from math import inf
from ScaledYOLOv4.models import yolo
from ScaledYOLOv4.models.yolo import Model
from ScaledYOLOv4.utils.general import check_anchor_order, make_divisible, check_file
from ScaledYOLOv4.utils.torch_utils import (
    time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device)


class SplitModel(Model):

    def forward(self, x, augment=False, profile=False, entry = 0, exit = inf):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train
    # return super().forward(x, augment, profile)

    def forward_once(self, x, profile=False):
    y, dt = [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

        if profile:
            try:
                import thop
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
            except:
                o = 0
            t = time_synchronized()
            for _ in range(10):
                _ = m(x)
            dt.append((time_synchronized() - t) * 100)
            print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output

    if profile:
        print('%.1fms total' % sum(dt))
    return x