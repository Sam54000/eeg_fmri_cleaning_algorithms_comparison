#!/opt/anaconda3/envs/mne/bin/python3 -S  python  #
# -*- coding: utf-8 -*-
# ===============================================================================
# Author: Dr. Samuel Louviot, PhD
# Institution: Nathan Kline Institute
#              Child Mind Institute
# Address: 140 Old Orangeburg Rd, Orangeburg, NY 10962, USA
#          215 E 50th St, New York, NY 10022
# Date: 2024-04-04
# email: samuel DOT louviot AT nki DOT rfmh DOT org
# ===============================================================================
# LICENCE GNU GPLv3:
# Copyright (C) 2024  Dr. Samuel Louviot, PhD
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ===============================================================================

"""Cleaning pipelines for EEG data.

This module contains several methods to clean the EEG data recorded during fMRI.
The different methods consist of cleaning the gradient and BCG artifacts with 
the homemade pipelines (called here CBIN-CLEANER). Once this first step is done,
the data can be further cleaned by either using ASR and/or PyPrep algorithms.
"""

import os
import shutil
from pathlib import Path

import asrpy
import bids
import mne
import numpy as np
import pyprep
from decorators import pipe
from eeg_fmri_cleaning.main import clean_bcg, clean_gradient
from eeg_fmri_cleaning.utils import read_raw_eeg


def write_report(message: str, filename: str | os.PathLike) -> None:
    """Append a message to a txt file.

    Args:
        message (str): The message to append.
        filename (str | os.PathLike): The file to append the message.
    """
    if isinstance(filename, os.PathLike) or isinstance(filename, str):
        with open(filename, "a") as f:
            f.write(message)
            f.write("\n")
    else:
        raise ValueError("The filename must be a string or a Path object.")


class CleanerPipelines:
    """Class to clean the EEG data using different algorithms."""
    def __init__(self, BIDSFile: bids.layout.models.BIDSFile) -> None:  # noqa: D107
        self.BIDSFile = BIDSFile
        self.entities = BIDSFile.get_entities()
        self.rawdata_path = Path(BIDSFile.path)
        self.process_history = []

    def read_raw(self: "CleanerPipelines") -> "CleanerPipelines":
        """Read the raw EEG data using MNE."""
        try:
            self.raw = read_raw_eeg(self.BIDSFile.path)
        except Exception as e:
            print(f"Error while reading the raw data: {e}")
        return self

    def _make_derivatives_saving_path(
        self: "CleanerPipelines") -> "CleanerPipelines":
        """Create the path to save the cleaned files in the BIDS format.

        It is a file specific path that is generated based on the BIDSFile
        object.

        Args:
            added_folder (str, optional): The folder to be added after the 
                                          derivatives one.
        """
        reading_path = Path(self.BIDSFile.path)
        path_parts = list(reading_path.parts)
        rawdata_dirname = [name for name in path_parts if "raw" in name.lower()][0]
        path_parts[path_parts.index(rawdata_dirname)] = "DERIVATIVES"
        added_folder = "_".join(self.process_history)
        path_parts.insert(path_parts.index("DERIVATIVES") + 1, added_folder)
            
        filename = Path(*path_parts)
        self.derivatives_path = filename.parent
        self.derivatives_path.mkdir(parents=True, exist_ok=True)
        return self

    def _copy_sidecar(self: "CleanerPipelines") -> None:
        """Copy the sidecar file to the derivative folder.

        Args:
            BIDSFile (bids.layout.models.BIDSFile): The BIDSFile object.
            where_to_copy (str | os.PathLike): The folder to copy the sidecar file.
        """
        filename_base, _ = os.path.splitext(self.BIDSFile.path)
        sidecar_filename = filename_base + ".json"
        if os.path.isfile(sidecar_filename):
            shutil.copyfile(sidecar_filename, self.derivatives_path)

    def _save_raw(self: "CleanerPipelines") -> "CleanerPipelines":
        """Save the cleaned raw EEG data in the BIDS format."""
        saving_filename = os.path.splitext(self.BIDSFile.path)[0] + ".fif"
        self.raw.save(saving_filename, overwrite=True)
        return self

    @pipe
    def clean_gradient(self: "CleanerPipelines") -> "CleanerPipelines":
        """Clean the gradient artifacts from the EEG data."""
        self.raw = clean_gradient(self.raw)
        self.process_history.append("GRAD")
        return self

    @pipe
    def clean_bcg(self: "CleanerPipelines") -> "CleanerPipelines":
        """Clean the BCG artifacts from the EEG data."""
        self.raw = clean_bcg(self.raw)
        self.process_history.append("BCG")
        return self

    def _task_is(self, task_name: str) -> bool:
        return self.BIDSFile.get_entities()["task"] == task_name

    @pipe
    def run_pyprep(self: "CleanerPipelines") -> "CleanerPipelines":
        """Clean the EEG data using the PyPrep algorithm.

        Args:
            raw (mne.io.Raw): The raw EEG data.

        Returns:
            mne.io.Raw: The cleaned EEG data.
        """
        montage = mne.channels.make_standard_montage("easycap-M10")
        self.raw.apply_montage(montage)
        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": np.arange(60, self.raw.info["sfreq"] / 2, 60),
        }
        prep = pyprep.PrepPipeline(self.raw, prep_params, channel_wise=True)
        prep.fit()
        self.raw = prep.raw
        self.process_history.append("PREP")
        return self

    @pipe
    def run_asr(self) -> mne.io.Raw:
        """Clean the EEG data using the ASR algorithm.

        Args:
            raw (mne.io.Raw): The raw EEG data.

        Returns:
            mne.io.Raw: The cleaned EEG data.
        """
        asr = asrpy.ASR()
        asr.fit(self.raw)
        self.raw = asr.transform(self.raw)
        self.process_history.append("ASR")
        return self