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

from cleaner_pipelines import write_report, CleanerPipelines


class MainCleanerPipelines:
    def __init__(self, BIDSFile: bids.layout.models.BIDSFile):
        self.BIDSFile = BIDSFile
            
    def run_cbin_cleaner(self) -> None:  # noqa: D103
        cleaner = CleanerPipelines(self.BIDSFile)
        cleaner.read_raw()
        if cleaner._task_is("checker"):
            cleaner.clean_gradient().clean_bcg()
        elif cleaner._task_is("checkeroff"):
            cleaner.clean_bcg()
        return cleaner

    def run_cbin_cleaner_asr(self) -> None:  # noqa: D103
        cleaner = self.run_cbin_cleaner(self.BIDSFile)
        cleaner.run_asr()

    def run_cbin_cleaner_pyprep_asr(self) -> None:  # noqa: D103
        cleaner = self.run_cbin_cleaner(self.BIDSFile)
        cleaner.run_pyprep()
        cleaner.run_asr()
    
def main(reading_path):
   
    root = reading_path.parent
    derivatives_path = root.joinpath('DERIVATIVES')
    report_filename = derivatives_path.joinpath('report.txt')


    layout = bids.BIDSLayout(reading_path)
    file_list = layout.get(extension=".set")

    for BIDSFile_object in file_list:
        if any([BIDSFile_object.task == "checker",
                BIDSFile_object.task == "checkeroff"]):
            #
            # The problem is here don't make a list of functions
            #
            for pipeline_function in ["run_cbin_cleaner", 
                                    "run_cbin_cleaner_asr", 
                                    "run_cbin_cleaner_pyprep_asr"]:
                try:
                    getattr(
                        MainCleanerPipelines(
                            BIDSFile_object), pipeline_function)()

                except Exception as e:
                    message = f"""filename: {str(BIDSFile_object.filename)}
                    process:{pipeline_function}
                    error:{str(e)}

                    """
                    write_report(message, report_filename)