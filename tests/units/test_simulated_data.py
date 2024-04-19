
# Generated by CodiumAI
import mne
import pytest
import pandas as pd
import os
from pathlib import Path
from simulated_data import simulate_eeg_data, DummyDataset

@pytest.fixture
def raw_data():
    return simulate_eeg_data()

@pytest.fixture
def testing_path():
    cwd = Path.cwd()
    output_dir = cwd.joinpath('tests','outputs') 
    return output_dir
    
def test_returns_instance_of_rawarray():
    result = simulate_eeg_data()
    assert isinstance(result, mne.io.RawArray)

# The function is called with n_channels = 0.
def test_called_with_n_channels_zero():
    with pytest.raises(ValueError):
        simulate_eeg_data(n_channels=0)

def test_dummy_dataset_called_with_zeros():
    with pytest.raises(ValueError) as e:
        dataset = DummyDataset(n_subjects=0, 
                     n_sessions=0, 
                     n_runs=0)
        print(e)
        
def test_participant_metadata():
    dataset = DummyDataset(n_subjects = 5)
    dataset.create_participants_metadata()
    assert isinstance(dataset.participant_metadata, pd.DataFrame)
    assert dataset.participant_metadata.shape[0] == 5
    nan_mask = dataset.participant_metadata.isna()
    for column in dataset.participant_metadata.columns:
        assert not any(nan_mask[column].values)

def test_add_participant_metadata():
    dataset = DummyDataset(n_subjects = 5)
    dataset.create_participants_metadata()
    dataset._add_participant_metadata(
        participant_id = 'sub-06',
        age = 26,
        sex = 'M',
        handedness = 'R'
    )
    assert isinstance(dataset.participant_metadata, pd.DataFrame)
    assert dataset.participant_metadata.shape[0] == 6
    nan_mask = dataset.participant_metadata.isna()
    for column in dataset.participant_metadata.columns:
        assert not any(nan_mask[column].values)

def test_generate_label():
    dataset = DummyDataset(root = './')
    for i in range(1,12):
        labels = dataset._generate_label('subject', i, 'TEST')
        assert labels == f'sub-TEST{i:03d}'
    labels = dataset._generate_label('subject', 1)
    assert labels == 'sub-001'
    labels = dataset._generate_label('session', 1)
    assert labels == 'ses-001'
    labels = dataset._generate_label('run', 1)
    assert labels == 'run-001'
    
def test_create_modality_agnostic_dir(testing_path):
    dataset = DummyDataset(root = testing_path)
    path = dataset.create_modality_agnostic_dir()
    asserting_path = testing_path.joinpath('RAW', 'sub-001', 'ses-001')
    assert isinstance(path[0], Path)
    assert str(path[0]) == str(asserting_path)

def test_extract_entities_from_path(testing_path):
    dataset = DummyDataset(root = testing_path)
    asserting_path = testing_path.joinpath('RAW', 'sub-001', 'ses-001')
    entities = dataset._extract_entities_from_path(asserting_path)
    assert entities == {'subject': 'sub-001', 'session': 'ses-001'}

def test_create_sidecar_json(testing_path):
    dataset = DummyDataset(root = testing_path)
    eeg_filename = 'sub-001_ses-001_task-test_run-001_eeg.vhdr'
    base_eeg_filename, _ = os.path.splitext(eeg_filename)
    eeg_path = testing_path.joinpath('RAW', 
                                   'sub-001', 
                                   'ses-001', 
                                   'eeg')
    eeg_path.mkdir(parents=True, exist_ok=True)
    eeg_full_path = eeg_path.joinpath(eeg_filename)
    dataset._create_sidecar_json(eeg_full_path)
    asserting_path = eeg_path.joinpath(base_eeg_filename + '.json')
    assert asserting_path.exists()

def test_eeg_dataset(testing_path):
    dataset = DummyDataset(root = testing_path)
    dataset.create_eeg_dataset()
    asserting_path = testing_path.joinpath('RAW', 'sub-001', 'ses-001', 'eeg')
    eeg_filenames = ['sub-001_ses-001_task-test_run-001_eeg.vhdr',
                     'sub-001_ses-001_task-test_run-001_eeg.vmrk',
                     'sub-001_ses-001_task-test_run-001_eeg.eeg',
                     'sub-001_ses-001_task-test_run-001_eeg.json']
    assert asserting_path.is_dir()
    for filename in eeg_filenames:
        eeg_path = asserting_path.joinpath(filename)
        assert eeg_path.exists()

def test_eeg_dataset_annotations(testing_path):
    dataset = DummyDataset(root = testing_path)
    dataset.create_eeg_dataset(
        fmt = 'eeglab',
        duration = 10,
        events_kwargs = dict(
            name = 'testing_event',
            number = 3,
            start = 2,
            stop = 8
        )
    )

    testing_path = testing_path.joinpath('RAW', 'sub-001', 'ses-001', 'eeg')
    testing_eeg_name = 'sub-001_ses-001_task-test_run-001_eeg.set'
    filename = testing_path.joinpath(testing_eeg_name)
    raw = mne.io.read_raw_eeglab(filename)
    annotations = raw.annotations
    assert len(annotations.onset) == 3
    assert annotations.description[0] == 'testing_event'    
