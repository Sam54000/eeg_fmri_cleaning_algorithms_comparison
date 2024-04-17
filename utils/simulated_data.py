import json
import os
import shutil
import tempfile

import mne
import numpy as np
from path_handler import DirectoryTree


def simulate_eeg_data(
    n_channels: int = 32,
    duration_seconds: int = 2,
    sampling_frequency: int = 256) -> mne.io.RawArray:
    """Simulate EEG data.

    This is just to perform unittest, it is basically a random distirbution.
    I will need to think about simulated actual EEG data.

    Args:
        n_channels (int): _description_
        duration_seconds (int): _description_
        sampling_frequency (int): _description_

    Returns:
        mne.io.RawArray
    """
    n_samples = duration_seconds * sampling_frequency
    data = np.random.rand(n_channels, n_samples)
    info = mne.create_info(n_channels, sampling_frequency, ch_types="eeg")
    return mne.io.RawArray(data, info)
class DummyDataset:
    """A class to create a dummy BIDS dataset for EEG data.
    
    This class creates a dummy BIDS dataset in order to test pipelines.
    The dataset is generated in the temporary folder of the computer.
    Once test are done on the dataset, it is possible to remove it from memory
    using the flush method. 
    """
    def __init__(self, 
                 n_subjects: int = 1, 
                 n_sessions: int = 1, 
                 n_runs: int = 1,
                 data_folder: str = "RAW") -> None:
        """Initialize the DummyDataset object.

        Args:
            n_subjects (int, optional): The number of subject to simulate. 
                                        Defaults to 1.
            n_sessions (int, optional): The number of sessions to simulate. 
                                        Defaults to 1.
            n_runs (int, optional): The number of run to simulate. 
                                    Defaults to 1.
            data_folder (str, optional): The location of the data 
                                         (source, rawdata, derivatives). 
                                         Defaults to "RAW".
        """
        self.n_subjects = n_subjects
        self.n_sessions = n_sessions
        self.n_runs = n_runs
        self.data_folder = data_folder
        self.bids_path = None
        self.root = tempfile.mkdtemp()
    
    def _create_sidecar_json(self, 
                            eeg_filename: str | os.PathLike,
                            eeg_folder: str | os.PathLike) -> None:
        """Create a sidecar JSON file for the EEG data.

        Args:
            eeg_filename (str | os.PathLike): The EEG data file name.
            eeg_folder (str | os.PathLike): The folder containing the EEG data.
        """
        json_filename = os.path.splitext(eeg_filename)[0] + ".json"
        json_saving_path = os.path.join(eeg_folder, json_filename)
        
        json_content = {
          "SamplingFrequency":2400,
          "Manufacturer":"Brain Products",
          "ManufacturersModelName":"BrainAmp DC",
          "CapManufacturer":"EasyCap",
          "CapManufacturersModelName":"M1-ext",
          "PowerLineFrequency":50,
          "EEGReference":"single electrode placed on FCz",
          "EEGGround":"placed on AFz",
          "SoftwareFilters":{
              "Anti-aliasing filter":{
              "half-amplitude cutoff (Hz)": 500,
              "Roll-off": "6dB/Octave"
              }
          },
          "HardwareFilters":{
              "ADC's decimation filter (hardware bandwidth limit)":{
              "-3dB cutoff point (Hz)":480,
              "Filter order sinc response":5
              }
          },
          }

        with open(json_saving_path, 'w') as json_file:
            json.dump(json_content, json_file, indent=4)
        return self
        
    def flush(self, check: bool = True) -> None:
        """Remove the temporary directory from memory."""
        tree = DirectoryTree(self.root)
        if check:
            print("The following directory will be removed:")
            tree.print_tree()
        else:
            print("Removing the temporary directory...")
            print("Content being removed:")
            tree.print_tree()

            shutil.rmtree(self.root,
                          ignore_errors=True,
                          onerror=None)
            post_removal_checker = os.path.exists(self.root)
            if post_removal_checker:
                print("The tree was not removed.")
            else:
                print("The tree was successfully removed.")
            
        return self
            
    def create_eeg_dataset(self) -> str:
        """Create temporary BIDS dataset.

        Create a dummy BIDS dataset for EEG data with multiple subjects, sessions, 
        and runs.

        Args:
            n_subjects (int, optional): Number of subjects to create. Defaults to 1.
            n_sessions (int, optional): Number of sessions per subject. Defaults to 1.
            n_runs (int, optional): Number of runs per session. Defaults to 1.

        Returns:
            str: The path of the temporary BIDS dataset.
        """
        # Create a temporary directory for the BIDS dataset
        self.bids_path = os.path.join(self.root,
                                     self.data_folder)
        os.mkdir(self.bids_path)

        # Define the necessary BIDS files for dataset description
        dataset_description = {
            "Name": "THIS IS A DUMMY DATASET",
            "BIDSVersion": "1.9.0",
            "License": "CC0",
            "Authors": ["Jane Doe", "John Doe"]
        }

        # Write dataset_description.json
        with open(
            os.path.join(self.bids_path, "dataset_description.json"), 'w'
            ) as desc_file:
            json.dump(dataset_description, desc_file, indent=4)

        # Initialize participants data
        participants_data = "participant_id\tage\n"

        # Generate subjects, sessions, and runs
        for sub_id in range(1, self.n_subjects + 1):
            subject_label = f"sub-{sub_id:02d}"
            participants_data += f"{subject_label}\t{20 + sub_id}\n"  # Just example ages

            for ses_id in range(1, self.n_sessions + 1):
                session_label = f"ses-{ses_id:02d}"
                
                for run_id in range(1, self.n_runs + 1):
                    run_label = f"run-{run_id:02d}"
                    eeg_dir = os.path.join(self.bids_path, 
                                        subject_label, 
                                        session_label, "eeg")
                    os.makedirs(eeg_dir, exist_ok=True)  # Create necessary directories

                    # Define file names for EEG data files
                    eeg_filenames = [
                        f"{subject_label}_{session_label}_task-test_{run_label}_eeg.vhdr",
                        f"{subject_label}_{session_label}_task-test_{run_label}_eeg.eeg",
                        f"{subject_label}_{session_label}_task-test_{run_label}_eeg.vmrk"
                    ]

                    # Create dummy EEG files
                    for eeg_file in eeg_filenames:
                        with open(os.path.join(eeg_dir, eeg_file), 'w') as file:
                            file.write("Dummy EEG file content\n")
                    
                    # Create sidecar JSON file
                    self._create_sidecar_json(eeg_filenames[0], eeg_dir)

        # Write participants.tsv
        with open(os.path.join(self.bids_path, "participants.tsv"), 'w') as part_file:
            part_file.write(participants_data)

        print(f"Temporary BIDS dataset created at {self.bids_path}")
        return self

    def print_bids_tree(self) -> None:
        """Print the BIDS dataset tree."""
        tree = DirectoryTree(self.bids_path)
        tree.print_tree()
        return self
