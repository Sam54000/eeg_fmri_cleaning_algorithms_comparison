import mne
import unittest
import bids
from bids.layout import BIDSLayout
from eeg_fmri_cleaning_algorithms_comparison import (
    write_report,
    CleanerPipelines,
)

class TestUtils(unittest.TestCase):
    def __init__(self):
        data_path = '/projects/EEG_FMRI/bids_eeg/BIDS/NEW/RAW'
        layout = BIDSLayout(data_path)
        files = layout.get_files()
        test_filenames, test_fileobjects = files.items()
        self.test_fileobject = test_fileobjects[0]
        
    def test_read_raw(self):
        raw = CleanerPipelines(self.test_fileobject)._read_raw()
        assert isinstance(raw, mne.io.Raw)

        # Add assertions to check the result
        
    def test_find_real_channel_name(self):
        raw = ...  # Create a mock instance of mne.io.Raw
        name = "ecg"
        result = utils.find_real_channel_name(raw, name)
        # Add assertions to check the result
        
    def test_map_channel_type(self):
        raw = ...  # Create a mock instance of mne.io.Raw
        result = utils.map_channel_type(raw)
        # Add assertions to check the result
        
    def test_set_channel_types(self):
        raw = ...  # Create a mock instance of mne.io.Raw
        channel_map = {"channel1": "type1", "channel2": "type2"}
        result = utils.set_channel_types(raw, channel_map)
        # Add assertions to check the result
        
    def test_input_interpreter(self):
        input_string = "1,2,3,4,5"
        input_param = "param"
        max_value = 10
        result = utils.input_interpreter(input_string, input_param, max_value)
        # Add assertions to check the result
        
    def test_read_raw_eeg(self):
        filename = "/path/to/file"
        preload = True
        result = utils.read_raw_eeg(filename, preload)
        # Add assertions to check the result
        
    def test_numerical_explorer(self):
        directory = "/path/to/directory"
        prefix = "prefix"
        result = utils.numerical_explorer(directory, prefix)
        # Add assertions to check the result

if __name__ == "__main__":
    unittest.main()
