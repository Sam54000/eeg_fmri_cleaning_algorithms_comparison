import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
import numpy as np

import bids
import mne
import pytest
import simulated_data

import eeg_fmri_cleaning_algorithms_comparison.cleaner_pipelines as cp

@pytest.fixture
def dataset_structure() -> Generator[Dict[str, Any], None, None]:
    cwd = Path.cwd()
    output_dir = cwd.joinpath('tests','outputs') 
    dataset_object = simulated_data.DummyDataset(root=output_dir,
                                                 flush=False)
    return dataset_object

@pytest.fixture
def light_dataset() -> Generator[Any, Any, Any]:
    cwd = Path.cwd()
    output_dir = cwd.joinpath('tests','outputs') 
    dataset_object = simulated_data.DummyDataset(root=output_dir,
                                                 task='test',
                                                 flush=True)
    dataset_object.create_eeg_dataset(light = True, fmt='eeglab')
    yield dataset_object

@pytest.fixture(scope='class')
def heavy_dataset() -> Generator[Any, Any, Any]:
    cwd = Path.cwd()
    output_dir = cwd.joinpath('tests','outputs') 
    dataset_object = simulated_data.DummyDataset(root=output_dir,
                                                 flush=False)
    dataset_object.create_eeg_dataset(
        fmt='eeglab',
        n_channels = 16,
        sampling_frequency=5000,
        duration = 25,
        misc_channels = ['ecg'],
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
    cleaner.read_raw()

    yield cleaner


def test_append_message_to_txt_file(light_dataset) -> None:
    bids_path = light_dataset.bids_path
    bids_layout = bids.layout.BIDSLayout(bids_path)
    bids_files = bids_layout.get(extension = '.set')
    cleaner = cp.CleanerPipelines(bids_files[0])
    print(cleaner.derivatives_path)
    message = "This is a test message"
    cleaner.write_report(message)
    with open(cleaner.derivatives_path.joinpath('report.txt'), 'r') as f:
        assert f.read() == message+"\n"
        


def test_filename_argument_is_none() -> None:
    # Call the function under test with a filename argument
    try:
        cp.write_report("This is a test message", None)
    except ValueError as e:
        assert str(e) == "The filename must be a string or a Path object."

def test_make_derivatives_path(light_dataset) -> None:
    bids_path = light_dataset.bids_path
    bids_layout = bids.layout.BIDSLayout(bids_path)
    bids_files = bids_layout.get(extension = '.set')
    cleaner = cp.CleanerPipelines(bids_files[0])
    cwd = Path.cwd()
    test_output_path = cwd.joinpath('tests','outputs')
    for folder in test_output_path.iterdir():
        if 'temporary_directory_generated_' in folder.name:
            temporary_directory = folder
            break
    expected_path = temporary_directory.joinpath('DERIVATIVES')
    
    assert str(cleaner.derivatives_path) == str(expected_path)

def test_make_process_path(light_dataset) -> None:
    bids_path = light_dataset.bids_path
    bids_layout = bids.layout.BIDSLayout(bids_path)
    bids_files = bids_layout.get(extension = '.set')
    cleaner = cp.CleanerPipelines(bids_files[0])
    cwd = Path.cwd()
    test_output_path = cwd.joinpath('tests','outputs')
    for folder in test_output_path.iterdir():
        if 'temporary_directory_generated_' in folder.name:
            temporary_directory = folder
            break
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
        cleaner._make_process_path()
        expected_path = temporary_directory.joinpath(
                'DERIVATIVES',
                added_folder,
            )
        assert str(cleaner.process_path) == str(expected_path)

def test_make_subject_session_path(light_dataset) -> None:
    bids_path = light_dataset.bids_path
    bids_layout = bids.layout.BIDSLayout(bids_path)
    bids_files = bids_layout.get(extension = '.set')
    cleaner = cp.CleanerPipelines(bids_files[0])
    cwd = Path.cwd()
    test_output_path = cwd.joinpath('tests','outputs')
    for folder in test_output_path.iterdir():
        if 'temporary_directory_generated_' in folder.name:
            temporary_directory = folder
            break
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
        cleaner._make_process_path()
        cleaner._make_subject_session_path()
        expected_path = temporary_directory.joinpath(
                'DERIVATIVES',
                added_folder,
                'sub-001',
                'ses-001'
            )
        assert str(cleaner.subject_session_path) == str(expected_path)

def test_make_modality_path(light_dataset) -> None:
    bids_path = light_dataset.bids_path
    bids_layout = bids.layout.BIDSLayout(bids_path)
    bids_files = bids_layout.get(extension = '.set')
    cleaner = cp.CleanerPipelines(bids_files[0])
    cwd = Path.cwd()
    test_output_path = cwd.joinpath('tests','outputs')
    for folder in test_output_path.iterdir():
        if 'temporary_directory_generated_' in folder.name:
            temporary_directory = folder
            break
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
        cleaner._make_process_path()
        cleaner._make_subject_session_path()
        cleaner._make_modality_path()
        expected_path = temporary_directory.joinpath(
                'DERIVATIVES',
                added_folder,
                'sub-001',
                'ses-001',
                'eeg'
            )
        assert str(cleaner.modality_path) == str(expected_path)
        
def test_task_is_test(light_dataset) -> None:
    bids_path = light_dataset.bids_path
    bids_layout = bids.layout.BIDSLayout(bids_path)
    bids_files = bids_layout.get(extension = '.set')
    cleaner = cp.CleanerPipelines(bids_files[0])
    cleaner.process_history = list()
    procedures = ['GRAD', 'ASR', 'PYPREP']
    for procedure in procedures:
        cleaner.process_history.append(procedure) 
        assert cleaner._task_is('test')
        
def test_sidecare_copied_at_correct_location(light_dataset):
    bids_path = light_dataset.bids_path
    bids_layout = bids.layout.BIDSLayout(bids_path)
    bids_files = bids_layout.get(extension = '.set')
    bids_file = bids_files[0]
    cleaner = cp.CleanerPipelines(bids_file)
    cleaner.process_history = list()
    procedures = ['GRAD', 'ASR', 'PYPREP']
    for procedure in procedures:
        cleaner.process_history.append(procedure) 
        cleaner._make_process_path()
        cleaner._make_subject_session_path()
        cleaner._make_modality_path()
        cleaner._copy_sidecar()

        path = cleaner.modality_path

        expected_filename = path.joinpath(
            'sub-001_ses-001_task-test_run-001_eeg.json'
            )

        print(expected_filename)

        assert expected_filename.exists()
    
def test_save_raw_method(light_dataset):
    bids_path = light_dataset.bids_path
    bids_layout = bids.layout.BIDSLayout(bids_path)
    bids_files = bids_layout.get(extension = '.set')
    cleaner = cp.CleanerPipelines(bids_files[0])
    cleaner.raw = simulated_data.simulate_light_eeg_data()
    cleaner.process_history = list()
    procedures = ['GRAD', 'ASR', 'PYPREP']
    for procedure in procedures:
        cleaner.process_history.append(procedure) 
        cleaner._make_process_path()
        cleaner._make_subject_session_path()
        cleaner._make_modality_path()
        cleaner._copy_sidecar()
        cleaner._save_raw()

        expected_filename = cleaner.modality_path.joinpath(
                'sub-001_ses-001_task-test_run-001_eeg.fif'
            )
        assert os.path.isfile(expected_filename)

def test_decorator_pipe(light_dataset):
    bids_path = light_dataset.bids_path
    bids_layout = bids.layout.BIDSLayout(bids_path)
    bids_files = bids_layout.get(extension = '.set')
    cleaner = cp.CleanerPipelines(bids_files[0])
    cleaner.raw = simulated_data.simulate_light_eeg_data()
    procedures = 'TEST_PIPE'
    cleaner.function_testing_decorator()

    expected_saving_path = Path(
        os.path.join(
            bids_path.parent,
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

class TestRunsCleanerPipelines:
    def test_run_clean_gradient(self, heavy_dataset):
        heavy_dataset.run_clean_gradient()
        cwd = Path.cwd()
        testing_path = cwd.joinpath('tests','outputs')
        for content in testing_path.iterdir():
            if 'temporary_directory_generated_' in content.name:
                temporary_directory = content
                break
        expected_saving_path = Path(
            os.path.join(
                temporary_directory,
                'DERIVATIVES',
                'GRAD',
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
        assert isinstance(heavy_dataset.raw, mne.io.BaseRaw)
        assert len(heavy_dataset.raw.annotations.description) == 10
        assert isinstance(heavy_dataset, cp.CleanerPipelines)
        assert heavy_dataset.raw.get_data().shape == (17, 125000)
        assert np.max(heavy_dataset.raw.get_data()) > 0
        
    def test_run_clean_bcg(self, heavy_dataset):
        heavy_dataset.run_clean_bcg()
        cwd = Path.cwd()
        testing_path = cwd.joinpath('tests','outputs')
        for content in testing_path.iterdir():
            if 'temporary_directory_generated_' in content.name:
                temporary_directory = content
                break
        expected_saving_path = Path(
            os.path.join(
                temporary_directory,
                'DERIVATIVES',
                'BCG',
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
        assert isinstance(heavy_dataset.raw, mne.io.BaseRaw)
        assert len(heavy_dataset.raw.annotations.description) == 10
        assert isinstance(heavy_dataset, cp.CleanerPipelines)

    def test_run_pyprep(self, heavy_dataset):
        #Pyprep crashes because it doesn't like my simulated data.
        #It thinks that there is too many bad electrodes.
        heavy_dataset.run_pyprep(montage_name='biosemi16')
        assert isinstance(heavy_dataset.raw, mne.io.BaseRaw)
        assert len(heavy_dataset.raw.annotations.description) == 10
        assert isinstance(heavy_dataset, cp.CleanerPipelines)

    def test_run_asr(self, heavy_dataset):
        heavy_dataset.run_asr()
        cwd = Path.cwd()
        testing_path = cwd.joinpath('tests','outputs')
        for content in testing_path.iterdir():
            if 'temporary_directory_generated_' in content.name:
                temporary_directory = content
                break
        expected_saving_path = Path(
            os.path.join(
                temporary_directory,
                'DERIVATIVES',
                'BCG',
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
        assert isinstance(heavy_dataset.raw, mne.io.BaseRaw)
        assert len(heavy_dataset.raw.annotations.description) == 10
        assert isinstance(heavy_dataset, cp.CleanerPipelines)

    def test_chain(self, heavy_dataset):
        heavy_dataset.run_clean_gradient_and_bcg()
        cwd = Path.cwd()
        testing_path = cwd.joinpath('tests','outputs')
        for content in testing_path.iterdir():
            if 'temporary_directory_generated_' in content.name:
                temporary_directory = content
                break
        expected_saving_path = Path(
            os.path.join(
                temporary_directory,
                'DERIVATIVES',
                'GRAD_BCG',
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
        assert isinstance(heavy_dataset.raw, mne.io.BaseRaw)
        assert len(heavy_dataset.raw.annotations.description) == 10
        assert isinstance(heavy_dataset, cp.CleanerPipelines)
        assert heavy_dataset.raw.get_data().shape == (17, 6250)
        assert np.max(heavy_dataset.raw.get_data()) > 0