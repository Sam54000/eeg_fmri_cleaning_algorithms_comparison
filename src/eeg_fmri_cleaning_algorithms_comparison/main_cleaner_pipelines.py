#!/usr/bin/env -S  python  #
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
import os
import argparse

import bids

from cleaner_pipelines import CleanerPipelines


args = argparse.ArgumentParser(description="Run the CBIN-CLEANER pipeline on EEG data.")
args.add_argument("--subject", 
                  type=str,
                  help="The subject to process.",
                  required=True)

input_args = args.parse_args()
def run_cbin_cleaner(BIDSFile: bids.layout.models.BIDSFile) -> None:  # noqa: D103
    cleaner = CleanerPipelines(BIDSFile)
    cleaner.read_raw()
    if cleaner._task_is("checker"):
        cleaner.clean_gradient().clean_bcg()
    elif cleaner._task_is("checkeroff"):
        cleaner.clean_bcg()
    return cleaner

def run_cbin_cleaner_asr(BIDSFile: bids.layout.models.BIDSFile) -> None:  # noqa: D103
    cleaner = run_cbin_cleaner(BIDSFile)
    cleaner.run_asr()

def run_cbin_cleaner_pyprep_asr(BIDSFile: bids.layout.models.BIDSFile) -> None:  # noqa: D103
    cleaner = run_cbin_cleaner(BIDSFile)
    cleaner.run_pyprep()
    cleaner.run_asr()
    
def main(kwargs):
    data_path = "/projects/EEG_FMRI/bids_eeg/BIDS/NEW/RAW"


    layout = bids.BIDSLayout(data_path)
    file_list = layout.get(
        extension=".set",
        **kwargs,
        )

    for BIDSFile_object in file_list:
        if any([BIDSFile_object.task == "checker",
                BIDSFile_object.task == "checkeroff"]):
            for pipeline_function in [run_cbin_cleaner, 
                                    run_cbin_cleaner_asr, 
                                    run_cbin_cleaner_pyprep_asr]:
                pipeline_function(BIDSFile_object)
    
if __name__ == "__main__":
    print(input_args)
    main(input_args.__dict__)