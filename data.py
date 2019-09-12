def to_img(x):
    x = x.view(x.size(0), 1, 96, 96)
    return x

num_epochs =1000
batch_size = 64
learning_rate = 1e-4


def plot_sample_img(img, name):
    img = img.view(1, 96, 96)
    save_image(img, './sample_{}.png'.format(name))


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)
