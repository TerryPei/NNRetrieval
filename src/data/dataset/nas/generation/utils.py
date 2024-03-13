def base_convert(idx: int, base: int):
    next_level = idx // base
    if next_level != 0:
        return base_convert(next_level, base=base) + [idx % base]
    else:
        return [idx % base]
