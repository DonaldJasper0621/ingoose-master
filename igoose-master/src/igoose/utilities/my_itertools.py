import torch


def make_random_item_generator(source):
    for index in torch.randperm(len(source)).tolist():
        yield source[index]


def _make_infinite_item_generator(source):
    while True:
        is_empty = True

        for item in source:
            is_empty = False
            yield item

        if is_empty:
            raise ValueError("`source` is empty.")


def make_infinite_item_generator(source, maybe_random):
    if not maybe_random:
        yield from _make_infinite_item_generator(source)
        return

    try:
        num_items = len(source)
    except TypeError:
        yield from _make_infinite_item_generator(source)
    else:
        if not num_items:
            raise ValueError("`source` is empty.")

        while True:
            yield from make_random_item_generator(source)


def make_multi_sources_maybe_random_item_generator(sources, weights=None):
    maybe_random_item_generators = [
        make_infinite_item_generator(source, True) for source in sources
    ]
    if weights is None:
        weights = [1.0] * len(maybe_random_item_generators)
    categorical_distribution = torch.distributions.Categorical(
        probs=torch.tensor(list(weights), dtype=torch.float32)
    )
    while True:
        source_index = categorical_distribution.sample()
        yield next(maybe_random_item_generators[source_index])
