import os
import tempfile
from pathlib import Path

import bids
import pytest
import simulated_data
from typing import Dict, Any
import pytest
from typing import Generator

import eeg_fmri_cleaning_algorithms_comparison.cleaner_pipelines as cp

@pytest.fixture
def temp_bids_files() -> Generator[Any, Any, Any]:
    dataset_object = simulated_data.DummyDataset()
    dataset_object.create_eeg_dataset()
    bids_path = dataset_object.bids_path
    bids_layout = bids.layout.BIDSLayout(bids_path)
    bids_files = bids_layout.get(extension = '.vhdr')
    print(bids_files)
    temp_dataset = {'bids_files': bids_files, 'bids_path': bids_path}
    return temp_dataset


def test_append_message_to_txt_file() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        message = "This is a test message"
        filename = "test.txt"
        file_path = os.path.join(temp_dir, filename)

        # Call the function under test
        cp.write_report(message, file_path)

        # Check that the file was created and contains the correct message
        assert os.path.exists(file_path)
        with open(file_path, "r") as file:
            mapping =  dict.fromkeys(range(32))
            output_str = file.read()
            res = output_str.translate(mapping)
            assert res == message

def test_filename_argument_is_none() -> None:
    # Call the function under test with a filename argument
    try:
        cp.write_report("This is a test message", None)
    except ValueError as e:
        assert str(e) == "The filename must be a string or a Path object."

def test_make_saving_path(temp_bids_files) -> None:
    bids_file = temp_bids_files['bids_files'][0]
    bids_path = temp_bids_files['bids_path']
    cleaner = cp.CleanerPipelines(bids_file)
    cleaner.process_history = list()
    process = list()
    procedures = ['GRAD', 'ASR', 'PYPREP']
    for procedure in procedures:
        cleaner.process_history.append(procedure) 
        process.append(procedure)
        if len(process) > 1:
            added_folder = "_".join(process)
        else:
            added_folder = process[0]
        cleaner._make_derivatives_saving_path()
        expected_path = bids_path.parent.joinpath(
                f'DERIVATIVES/{added_folder}',
                'sub-001',
                'ses-001',
                'eeg'
            )
        assert str(cleaner.derivatives_path) == str(expected_path)
        
def test_sidecare_copied_at_correct_location(temp_bids_files):
    bids_file = temp_bids_files['bids_files'][0]
    cleaner = cp.CleanerPipelines(bids_file)
    cleaner.process_history = list()
    procedures = ['GRAD', 'ASR', 'PYPREP']
    for procedure in procedures:
        cleaner.process_history.append(procedure) 
        cleaner._make_derivatives_saving_path()
        cleaner._copy_sidecar()

        path = cleaner.derivatives_path

        expected_filename = path.joinpath(
            'sub-001_ses-001_task-test_run-001_eeg.json'
            )

        print(expected_filename)

        assert os.path.exists(str(expected_filename))
    
def test_save_raw_method(temp_bids_files):
    cleaner = cp.CleanerPipelines(temp_bids_files['bids_files'][0])
    cleaner.raw = simulated_data.simulate_eeg_data()
    cleaner.process_history = list()
    procedures = ['GRAD', 'ASR', 'PYPREP']
    for procedure in procedures:
        cleaner.process_history.append(procedure) 
        cleaner._make_derivatives_saving_path()
        cleaner._copy_sidecar()
        cleaner._save_raw()
        path = cleaner.derivatives_path

        expected_filename = Path(
            os.path.join(
                str(path),
                'sub-001_ses-001_task-test_run-001_eeg.fif'
            )
        )
        assert os.path.isfile(expected_filename)

def test_decorator_pipe(temp_bids_files):
    cleaner = cp.CleanerPipelines(temp_bids_files['bids_files'][0])
    cleaner.raw = simulated_data.simulate_eeg_data()
    procedures = 'TEST_PIPE'
    cleaner.function_testing_decorator()

    expected_saving_path = Path(
        os.path.join(
            temp_bids_files['bids_path'].parent,
            'DERIVATIVES',
            procedures,
            'sub-001',
            'ses-001',
            'eeg'
        )
    )
    eeg_filename = 'sub-001_ses-001_task-test_run-001_eeg.fif'
    json_filename = 'sub-001_ses-001_task-test_run-001_eeg.json'
    expected_eeg_filename = os.path.join(expected_saving_path, eeg_filename)
    expected_json_filename = os.path.join(expected_saving_path, json_filename)
    assert os.path.isfile(expected_eeg_filename)
    assert os.path.isfile(expected_json_filename)