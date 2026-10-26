"""Small explicit registries: no experiment-runner changes needed for new plugins."""


class Registry(dict):
    def register(self, name):
        def decorator(value):
            if name in self:
                raise ValueError(f"Duplicate plugin: {name}")
            self[name] = value
            return value

        return decorator

    def create(self, name, *args, **kwargs):
        if name not in self:
            raise ValueError(f"Unknown plugin {name!r}; choices: {', '.join(self)}")
        try:
            return self[name](*args, **kwargs)
        except TypeError as error:
            raise ValueError(f"Invalid parameters for {name!r}: {error}") from error


DATASETS = Registry()
MODELS = Registry()
METHODS = Registry()
