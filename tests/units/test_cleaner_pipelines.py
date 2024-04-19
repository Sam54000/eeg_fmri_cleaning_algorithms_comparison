import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import bids
import mne
import pytest
import simulated_data

import eeg_fmri_cleaning_algorithms_comparison.cleaner_pipelines as cp


@pytest.fixture
def temp_bids_files() -> Generator[Any, Any, Any]:
    cwd = Path.cwd()
    test_output_path = cwd.joinpath('tests','outputs')
    dataset_object = simulated_data.DummyDataset(root=test_output_path)
    dataset_object.create_eeg_dataset()
    bids_path = dataset_object.bids_path
    bids_layout = bids.layout.BIDSLayout(bids_path)
    bids_files = bids_layout.get(extension = '.vhdr')
    temp_dataset = {'bids_files': bids_files, 'bids_path': bids_path}
    return temp_dataset

@pytest.fixture
def predifined_cleaner(testing_path):
    dataset_object = simulated_data.DummyDataset(root=testing_path)
    dataset_object.create_eeg_dataset(
        fmt='eeglab',
        sampling_frequency=5000,
        duration = 25,
        events_kwargs=dict(
            name = 'R128',
            number = 10,
            start = 1,
            stop = 21,
    )
    )
    bids_path = dataset_object.bids_path
    bids_layout = bids.layout.BIDSLayout(bids_path)
    bids_files = bids_layout.get(extension = '.set')
    cleaner = cp.CleanerPipelines(bids_files[0])
    return cleaner

@pytest.fixture
def testing_path():
    cwd = Path.cwd()
    output_dir = cwd.joinpath('tests','outputs') 
    return output_dir

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

def test_run_clean_gradient(predifined_cleaner):
    predifined_cleaner.read_raw()
    predifined_cleaner.clean_gradient()
    assert isinstance(predifined_cleaner.raw, mne.io.BaseRaw)
    assert len(predifined_cleaner.raw.annotations.description) == 10
    assert isinstance(predifined_cleaner, cp.CleanerPipelines)
    
def test_run_clean_bcg(predifined_cleaner):
    predifined_cleaner.read_raw()
    predifined_cleaner.clean_clean_bcg()
    assert isinstance(predifined_cleaner.raw, mne.io.BaseRaw)
    assert len(predifined_cleaner.raw.annotations.description) == 10
    assert isinstance(predifined_cleaner, cp.CleanerPipelines)

def test_run_pyprep(predifined_cleaner):
    predifined_cleaner.read_raw()
    predifined_cleaner.run_pyprep()
    assert isinstance(predifined_cleaner.raw, mne.io.BaseRaw)
    assert len(predifined_cleaner.raw.annotations.description) == 10
    assert isinstance(predifined_cleaner, cp.CleanerPipelines)

def test_run_asr(predifined_cleaner):
    predifined_cleaner.read_raw()
    predifined_cleaner.run_asr()
    assert isinstance(predifined_cleaner.raw, mne.io.BaseRaw)
    assert len(predifined_cleaner.raw.annotations.description) == 10
    assert isinstance(predifined_cleaner, cp.CleanerPipelines)

def test_chain(predifined_cleaner):
    predifined_cleaner.read_raw().run_clean_gradient().run_clean_bcg().run_pyprep().run_asr()
    assert isinstance(predifined_cleaner.raw, mne.io.BaseRaw)
    assert len(predifined_cleaner.raw.annotations.description) == 10
    assert isinstance(predifined_cleaner, cp.CleanerPipelines)