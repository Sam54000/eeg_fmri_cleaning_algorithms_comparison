import pytest
import simulated_data
import eeg_fmri_cleaning_algorithms_comparison.main_cleaner_pipelines as mcp
import eeg_fmri_cleaning_algorithms_comparison.cleaner_pipelines as cp
import bids
from pathlib import Path

@pytest.fixture(scope='class')
def dataset():
    cwd = Path.cwd()
    test_output_path = cwd.joinpath('tests','outputs')
    data = simulated_data.DummyDataset(
        n_subjects=2, 
        n_sessions=2, 
        n_runs=1,
        task='checker',
        root = test_output_path,
        flush = False
        )
    
    data._populate_labels()
    data.create_eeg_dataset(
        fmt = 'eeglab', 
        n_channels = 16,
        duration = 25,
        sampling_frequency = 5000,
        misc_channels = ['ecg'],
        events_kwargs = 
        {
            'name': 'R128',
            'number': 10,
            'start': 5,
            'stop': 30
        }
                            )
    return data
class TestFunctions:
    def test_run_cbin_cleaner(self,dataset):
        layout = bids.layout.BIDSLayout(dataset.bids_path)
        files = layout.get(extension='.set')
        
        for file in files:
            cleaner = cp.CleanerPipelines(file)
            mcp.run_cbin_cleaner(cleaner)
            
        cwd = Path.cwd()
        test_output_path = cwd.joinpath('tests','outputs')
        for content in test_output_path.iterdir():
            if 'temporary_directory_generated_' in content.name:
                temporary_directory = content
                break
        
        files_to_check = list()
        for subject in dataset.subjects:
            for session in dataset.sessions:
                for run in dataset.runs:
                    files_to_check.append(
                        temporary_directory.joinpath(
                        'DERIVATIVES',
                        'GRAD_BCG',
                        subject,
                        session,
                        f'{subject}_{session}_task-checker_run-{run}_eeg.fif'
                    )
                    )
                    assert files_to_check.exists()

class TestMain:
    def test_main_raw_path_integrity(self,dataset):
        cwd = Path.cwd()
        test_output_path = cwd.joinpath('tests','outputs')
        for content in test_output_path.iterdir():
            if 'temporary_directory_generated_' in content.name:
                temporary_directory = content
                break
        
        files_to_check = list()
        for subject in dataset.subjects:
            for session in dataset.sessions:
                for run in dataset.runs:
                    files_to_check.append(
                        temporary_directory.joinpath(
                        'RAW',
                        subject,
                        session,
                        f'{subject}_{session}_task-test_run-{run}_eeg.set'
                    )
                    )
        mcp.main(dataset.bids_path)
        for file in files_to_check:
            print(file)
            assert file.exists()
    
    def test_main_derivatives_path_integrity(self,dataset):
        cwd = Path.cwd()
        test_output_path = cwd.joinpath('tests','outputs')
        for content in test_output_path.iterdir():
            if 'temporary_directory_generated_' in content.name:
                temporary_directory = content
                break
        
        files_to_check = list()
        additional_folders = ['GRAD', 'GRAD_BCG', 'GRAD_BCG_ASR']
        for folder in additional_folders:
            for subject in dataset.subjects:
                for session in dataset.sessions:
                    for run in dataset.runs:
                        files_to_check.append(
                            temporary_directory.joinpath(
                            'DERIVATIVES',
                            folder,
                            subject,
                            session,
                            f'{subject}_{session}_task-test_run-{run}_eeg.fif'
                        )
                        )
        mcp.main(dataset.bids_path)
        for file in files_to_check:
            print(file)
            assert file.exists()
    
    def test_main_report_exists(self,dataset):
        cwd = Path.cwd()
        test_output_path = cwd.joinpath('tests','outputs')
        for content in test_output_path.iterdir():
            if 'temporary_directory_generated_' in content.name:
                temporary_directory = content
                break
        
        report_path = temporary_directory.joinpath('DERIVATIVES',
                                                   'report.txt')
        mcp.main(dataset.bids_path)
        assert report_path.exists()
    