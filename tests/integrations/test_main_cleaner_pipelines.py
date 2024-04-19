import pytest
import simulated_data
import eeg_fmri_cleaning_algorithms_comparison.main_cleaner_pipelines as mcp
from pathlib import Path

@pytest.fixture
def dataset():
    data = simulated_data.DummyDataset(
        n_subjects=2, 
        n_sessions=2, 
        n_runs=1,
        root = '../outputs',
        )
    
    data.create_eeg_dataset(
        fmt = 'eeglab', 
        n_channels = 16,
        duration = 60,
        events_kwars = 
        {
            'name': 'R128',
            'number': 10,
            'start': 5,
            'stop': 30
        }
                            )
    return data

def test_main_happy_path_path_integrity_of_rawdata(dataset):
    kwargs = {'subject': 'sub-001', 'rawpath': '../outputs/RAW'}
    output_dir = Path('../outputs')
    
    #search for the temporary directory generated by the pipeline
    for content in output_dir.iterdir():
        if "temporary_directory_generated_" in content.name:
            temporary_dir = content
            break
    
    raw_path = temporary_dir.joinpath('RAW')
    for subject in dataset.subjects:
    files_to_check =
    
    
    mcp.main(kwargs)
    
