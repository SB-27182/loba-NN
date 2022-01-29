import torch


def frontDim_split(data, u):
    """Splits the data at dimension u, returning x of size [batch_size, dims-1], and u of size [batch_size, 1]"""
    n = data.shape[1]
    assert (u > 0)
    assert (n > 1)
    assert (u <= n)
    if (u - 1 == 0):
        split = torch.split(data, (1, n-1), dim=1)
        u = split[0]
        x = split[1]

    elif (n - u == 0):
        split = torch.split(data, (n-1, 1), dim=1)
        u = split[1]
        x = split[0]

    else:
        split = torch.split(data, (u-1, 1, n-u), dim=1)
        u = split[1]
        x = torch.cat((split[0], split[2]), dim=1)

    return (u, x)
