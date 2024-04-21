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
from pathlib import Path
import argparse

import bids

from cleaner_pipelines import CleanerPipelines
parser = argparse.ArgumentParser(description="Run the cleaning pipelines")
parser.add_argument("reading_path", type=str, help="Path to the BIDS dataset")
args = parser.parse_args()

def run_cbin_cleaner(cleaner) -> None:  # noqa: D103
    cleaner.read_raw()
    if cleaner._task_is("checker"):
        cleaner.run_clean_gradient_and_bcg()
        return "run_clean_gradient_and_bcg"
    elif cleaner._task_is("checkeroff"):
        cleaner.run_clean_bcg()
        return "run_clean_bcg"
        

def run_cbin_cleaner_asr(cleaner) -> None:  # noqa: D103
    cleaner = run_cbin_cleaner(cleaner)
    cleaner.run_asr()


def run_cbin_cleaner_pyprep_asr(cleaner) -> None:  # noqa: D103
    cleaner = run_cbin_cleaner(cleaner)
    cleaner.run_pyprep()
    cleaner.run_asr()
    
def main(reading_path):
   
    layout = bids.BIDSLayout(reading_path)
    file_list = layout.get(extension=".set")

    for BIDSFile_object in file_list:
        cleaner = CleanerPipelines(BIDSFile_object)
        if any([BIDSFile_object.task == "checker",
                BIDSFile_object.task == "checkeroff"]):
            try:
                pipeline_function = run_cbin_cleaner(cleaner)
                pipeline_function = run_cbin_cleaner_asr(cleaner)
                pipeline_function = run_cbin_cleaner_pyprep_asr(cleaner)

            except Exception as e:
                message = f"""filename: {str(BIDSFile_object.filename)}
                process:{pipeline_function}
                error:{str(e)}

                """
                cleaner.write_report(message)

if __name__ == "__main__":
    main(args.reading_path)