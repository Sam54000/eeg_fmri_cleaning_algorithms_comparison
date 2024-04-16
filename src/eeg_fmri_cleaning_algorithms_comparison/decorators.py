import functools
from typing import Any
def pipe(func):  # noqa: ANN001
    """Decorator that pipes to the folder creation and saving methods.

    Args:
        func (_type_):

    Returns:
        _type_: _description_
    """
    @functools.wraps(func)
    def wrapper_decorator(self,  # noqa: ANN001
                          *args: tuple, 
                          **kwargs: dict[str, Any]) -> None:  # noqa: ANN002
        self._make_derivatives_saving_path()
        func(self,*args, **kwargs)
        self._save_raw()
        self._copy_sidecar()
    return wrapper_decorator