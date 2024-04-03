
import os
import shutil
from pathlib import Path

import asrpy
import bids
import mne
import numpy as np
import pyprep
from bids.layout import BIDSLayout
from eeg_fmri_cleaning.main import clean_bcg, clean_gradient
from eeg_fmri_cleaning.utils import read_raw_eeg
from EEG_quality_assessment.signal_metrics import SignalMetrics

from .utils import (
    asr_cleaner,
    cbin_cleaner,
    copy_sidecar,
    is_task_checker,
    make_directory,
    pyprep_cleaner,
    write_report,
)

data_path = '/projects/EEG_FMRI/bids_eeg/BIDS/NEW/RAW'
derivatives_path = '/projects/EEG_FMRI/bids_eeg/BIDS/NEW/DERIVATIVES'

layout = BIDSLayout(data_path)
file_list = layout.get(return_type='file', extension='.set')

def cbin_cleaner_pipeline(file: str | os.PathLike, 
                          derivatives_path: str | os.PathLike, 
                          saving_path_name: str | os.PathLike) -> mne.io.Raw:
    """Combine the cbin_cleaner function with side stuff to save the file.

    Args:
        file (str | os.PathLike): The file name to clean
        derivatives_path (str | os.PathLike): The path to save the cleaned file
        saving_path_name (str | os.PathLike): The name of the folder to save 
                                              the cleaned file

    Returns:
        mne.io.Raw: The cleaned file
    """
    raw = cbin_cleaner(file)
    saving_file_path_cbin_cleaner = make_directory(
        file, 
        os.path.join(derivatives_path, saving_path_name)
                        )
    
    raw.save(os.path.join(
        saving_file_path_cbin_cleaner,
        os.path.split(file)[1]
        )
    )
    copy_sidecar(file, saving_file_path_cbin_cleaner)
    return raw

def pyprep_cleaner_pipeline(raw: mne.io.Raw,
                            reading_filename: str | os.PathLike,
                            derivatives_path: str | os.PathLike, 
                            saving_path_name: str | os.PathLike) -> mne.io.Raw:
    """Applies the PyPREP cleaner algorithm.

    Args:
        raw (mne.io.Raw): The raw EEG data to be cleaned.
        derivatives_path (str|os.PathLike): The path to the directory where 
                                            the cleaned data will be saved.
        saving_path_name (str|os.PathLike): The name of the file to be saved.

    Returns:
        mne.io.Raw: The cleaned raw EEG data.
    """
    raw = pyprep_cleaner(raw)
    saving_file_path_prep = make_directory(
        reading_filename, 
        os.path.join(derivatives_path, saving_path_name)
    )
    
    raw.save(os.path.join(
        saving_file_path_prep,
        os.path.split(reading_filename)[1]
    ))
    copy_sidecar(reading_filename, saving_file_path_prep)
    return raw

def asr_cleaner_pipeline(raw: mne.io.Raw,
                         reading_filename: str | os.PathLike,
                         derivatives_path: str | os.PathLike, 
                         saving_path_name: str | os.PathLike) -> mne.io.Raw:
    """Applies the ASR cleaner algorithm.

    Args:
        raw (mne.io.Raw): The raw EEG data to be cleaned.
        reading_filename (str | os.PathLike): The name of the file to be cleaned.
        derivatives_path (str | os.PathLike): The path to the directory where
                                              the cleaned data will be saved.
        saving_path_name (str | os.PathLike): The name of the file to be saved.

    Returns:
        mne.io.Raw: The cleaned raw EEG data.
    """
    raw = asr_cleaner(raw)
    saving_file_path_asr = make_directory(
        reading_filename, 
        os.path.join(derivatives_path, derivatives_path)
                        )
    
    raw.save(os.path.join(
        saving_file_path_asr,
        os.path.split(reading_filename)[1]
        )
    )
    copy_sidecar(reading_filename, saving_file_path_asr)
    return raw

def main():
    for file in file_list:
        if is_task_checker(file):
            try:
                raw = cbin_cleaner_pipeline(file, derivatives_path, 'CBIN_CLEANER')
                raw_copy = raw.copy()
                raw = asr_cleaner_pipeline(raw_copy, 
                                           derivatives_path, 
                                           'CBIN_CLEANER_ASR')
                raw = pyprep_cleaner_pipeline(raw, 
                                              derivatives_path, 
                                              'CBIN_CLEANER_PREP')
                raw = asr_cleaner_pipeline(raw, 
                                           derivatives_path, 
                                           'CBIN_CLEANER_PREP_ASR')
                
                write_report(file,
                    f"{derivatives_path}/processed.txt")
            except Exception as e:
                write_report(f"{file}: {str(e)}", 
                    f"{derivatives_path}/not_processed.txt")
                continue

if __name__ == '__main__':
    main()