import functools
from pathlib import Path
from typing import Callable, Dict, Tuple, TypeVar, cast, Any

import bids
from simulated_data import DummyDataset

FunctionType = TypeVar('FunctionType', bound=Callable[..., Any])
def pipe(func: FunctionType) -> FunctionType:  # noqa: ANN001
    """Decorator that pipes to the folder creation and saving methods.

    Args:
        func (_type_):

    Returns:
        _type_: _description_
    """
    @functools.wraps(func)
    def wrapper_decorator(self: object,  # noqa: ANN001
                          *args: tuple, 
                          **kwargs: dict[str, Any]) -> None:  # noqa: ANN002
        func(self,*args, **kwargs)
        self._make_derivatives_saving_path()
        self._save_raw()
        self._copy_sidecar()
    return cast(FunctionType, wrapper_decorator)

def dummy_dataset(func: FunctionType) -> None:
    """Generate a dummy BIDS dataset for testing purpose.
    
    This decorator wraps a test method by creating a dummy BIDS dataset and 
    removing that temporary dataset after the function is executed.

    Args:
        func (_type_): _description_

    """
    @functools.wraps(func)
    def wrapper_decorator(self: object, 
                          *args: tuple, 
                          **kwargs: dict[str,Any]) -> None:  
        dataset_object = DummyDataset()
        eeg_dataset = dataset_object.create_eeg_dataset()
        bids_path = Path(eeg_dataset.bids_path)
        bids_layout = bids.layout.BIDSLayout(bids_path)
        bids_files = bids_layout.get(extension = '.vhdr')
        args = (bids_files, bids_path)
        func(self, *args, **kwargs)
        dataset_object.flush(check=False)
        return cast(FunctionType, wrapper_decorator)