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
import bids


if __name__ == "__main__":
    data_path = "/projects/EEG_FMRI/bids_eeg/BIDS/NEW/RAW"

    layout = bids.BIDSLayout(data_path)
    file_list = layout.get(extension=".set")

    for filename, BIDSFile_object in file_list.items():
        cleaner = CleanerPipelines(BIDSFile_object)
        cleaner._read_raw()
        cleaner.run_cbin_pipeline().run_asr_pipeline()
        cleaner._read_raw().run_cbin_pipeline().run_prep_pipeline().run_asr_pipeline()