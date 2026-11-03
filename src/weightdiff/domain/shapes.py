def is_wider(a_shape, b_shape):
    return (len(a_shape) == len(b_shape) and a_shape != b_shape
            and all(y >= x for x, y in zip(a_shape, b_shape)))
