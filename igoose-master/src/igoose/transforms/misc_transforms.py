from typing import Any, Iterable, Mapping


class RenameKey:
    def __init__(self, old_key, new_keys: Iterable[str]):
        self._old_key = old_key
        self._new_keys = list(new_keys)

    def __call__(self, data_map: Mapping[str, Any]) -> dict[str, Any]:
        output_data_map = dict(data_map)

        value = output_data_map.pop(self._old_key)

        for new_key in self._new_keys:
            output_data_map[new_key] = value

        return output_data_map
