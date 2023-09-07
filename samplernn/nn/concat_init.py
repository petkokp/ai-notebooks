
def concat_init(tensor, inits):
    try:
        tensor = tensor.data
    except AttributeError:
        pass

    (length, fan_out) = tensor.size()
    fan_in = length // len(inits)

    chunk = tensor.new(fan_in, fan_out)
    for (i, init) in enumerate(inits):
        init(chunk)
        tensor[i * fan_in: (i + 1) * fan_in, :] = chunk
