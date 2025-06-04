from tinygrad import nn, Tensor


def pixel_shuffle(x, upscale_factor):
    batch_size, channels, height, width = x.shape
    r = upscale_factor
    out_channels = channels // (r * r)
    assert channels == out_channels * r * r, "Channels must be divisible by upscale_factor squared"

    x = x.reshape(batch_size, out_channels, r, r, height, width)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(batch_size, out_channels, height * r, width * r)
    return x

class PRelu:
    def __init__(self, num_parameters):
        self.weight = Tensor.ones(num_parameters)

    def __call__(self, x):
        return x.relu() + self.weight.reshape(1, -1, 1, 1) * x.neg().relu().neg()

class SRVGGNetCompact:
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)] # + prelu
        activation = PRelu(num_parameters=num_feat)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            activation = PRelu(num_parameters=num_feat)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))

    def __call__(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = pixel_shuffle(out, 4)

        _, _, h, w = x.shape
        base = x.interpolate(size=(h * 4, w * 4), mode="nearest")

        out += base
        return out