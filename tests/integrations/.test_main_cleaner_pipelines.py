import pytest
import simulated_data
import eeg_fmri_cleaning_algorithms_comparison.main_cleaner_pipelines as mcp
from pathlib import Path

def 
def test_main_path_integrity():
    cwd = Path.cwd()
    test_output_path = cwd.joinpath('tests/outputs')
    data = simulated_data.DummyDataset(
        n_subjects=2, 
        n_sessions=2, 
        n_runs=1,
        task='checker',
        root = test_output_path
        )
    
    data._populate_labels()
    data.create_eeg_dataset(
        fmt = 'eeglab', 
        n_channels = 16,
        duration = 60,
        events_kwargs = 
        {
            'name': 'R128',
            'number': 10,
            'start': 5,
            'stop': 30
        }
                            )

    for content in test_output_path.iterdir():
        if 'temporary_directory_generated_' in content.name:
            temporary_directory = content
            break
    
    files_to_check = list()
    for data_folder in ['RAW', 'DERIVATIVES']:
        for subject in data.subjects:
            for session in data.sessions:
                for run in data.runs:
                    files_to_check.append(
                        temporary_directory.joinpath(
                        data_folder,
                        subject,
                        session,
                        f'{subject}_{session}_task-test_run-{run}_eeg.set'
                    )
                    )
    data.files_to_check = files_to_check
    mcp.main(data.bids_path)
    for file in data.files_to_check:
        print(file)
        assert file.exists()

def test_main_path_when_error():
    data = simulated_data.dummydataset(
        n_subjects=2, 
        n_sessions=2, 
        n_runs=1,
        task='checker',
        root = '../outputs',
        )
    
    data.create_eeg_dataset(
        fmt = 'eeglab', 
        n_channels = 16,
        duration = 60,
                            )

    test_output_path = Path('test/outputs')
    for content in test_output_path.iterdir():
        if 'temporary_directory_generated_' in content.name:
            temporary_directory = content
            break
    
    files_to_check = list()
    for data_folder in ['RAW', 'DERIVATIVES']:
        for subject in data.subjects:
            for session in data.sessions:
                for run in data.runs:
                    files_to_check.append(
                        temporary_directory.joinpath(
                        data_folder,
                        subject,
                        session,
                        f'{subject}_{session}_task-test_run-{run}_eeg.set'
                    )
                    )
    data.files_to_check = files_to_check
    kwargs = {'subject': 'sub-001', 'rawpath': '../outputs/raw'}
    mcp.main(kwargs)
    for file in data.files_to_check:
        print(file)
        assert not file.exists()
