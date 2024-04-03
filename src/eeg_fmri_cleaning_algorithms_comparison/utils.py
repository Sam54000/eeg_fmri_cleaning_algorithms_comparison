
import mne
import numpy as np
import os
from pathlib import Path
from EEG_quality_assessment.signal_metrics import SignalMetrics
import asrpy
import pyprep
import shutil

from eeg_fmri_cleaning.utils import read_raw_eeg
from eeg_fmri_cleaning.main import (
    clean_gradient,
    clean_bcg
)
import bids
from bids.layout import BIDSLayout

def is_task_checker(BIDSFile):
    return 'checker' in BIDSFile.get_entities()['task']

def write_report(message, filename):
    with open(filename, 'a') as f:
        f.write(message)
        f.write('\n')

def cbin_cleaner(file):
    raw = read_raw_eeg(file.path)
    if file.get_entities()['task'] == 'checker':
        raw = clean_gradient(raw)
    raw = clean_bcg(raw)
    return raw

def asr_cleaner(raw):
    asr = asrpy.ASR()
    asr.fit(raw)
    raw = asr.transform(raw)
    return raw

def copy_sidecar(file, where_to_copy):
    filename_base, filename_extension = os.path.splitext(file.path)
    sidecar_filename = filename_base + '.json'
    if os.path.isfile(sidecar_filename):
        shutil.copyfile(sidecar_filename, where_to_copy)
     
def make_directory(file, derivative_path):
    entities = bids.layout.parse_file_entities(file)
    file_path = Path(derivative_path)
    file_path.joinpath(entities['subject'],
                       entities['session'],
                       entities['datatype'])
    file_path.mkdir(parents=True, exist_ok=True)

    return file_path
    
def pyprep_cleaner(raw):
    montage = mne.channels.make_standard_montage('easycap-M10')
    raw.apply_montage(montage)
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(60, raw.info["sfreq"]/ 2, 60),
    }
    prep = pyprep.PrepPipeline(raw,
                               prep_params, 
                               channel_wise=True)
    prep.fit()
    
    return prep.raw
        