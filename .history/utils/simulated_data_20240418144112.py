import json
import os
import shutil
import tempfile
from pathlib import Path

import mne
import numpy as np
from path_handler import DirectoryTree
import pandas as pd


def get_var_name(var):
    for name, value in locals().items():
        if value is var:
            return name
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
    if n_channels <= 0:
        raise ValueError("The number of channels must be greater than 0.")

    if duration_seconds <= 0:
        raise ValueError("The duration must be greater than 0.")
    
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
                 data_folder: str = "RAW",
                 root: str | os.PathLike = None) -> None:
        """Initialize the DummyDataset object.

        Args:
            n_subjects (int, optional): The number of subject to simulate. 
                Defaults to 1.
                
            n_sessions (int, optional): The number of sessions to simulate. 
                Defaults to 1.
                
            n_runs (int, optional): The number of run to simulate. 
                Defaults to 1.
                
            data_folder (str, optional): The location of the data 
                source, rawdata, derivatives).  Defaults to "RAW".
                
            root (str | os.PathLike, optional): The root directory to create 
                the temporary dataset. If None, the dataset is created in the 
                 temporary directory of the system. Defaults to None.
        """
        arguments_to_check = [n_subjects, n_sessions, n_runs]
        arguments_name = [
            get_var_name(argument) 
            for argument in arguments_to_check
        ]
        conditions = [
            not isinstance(argument, int) or argument < 1
            for argument in arguments_to_check
        ]
        if any(conditions):
            error_message = "The number of "
            if sum(conditions) == 1:
                error_message += arguments_name[conditions.index(True)]
            else:
                error_message += " and ".join(
                    [arguments_name[i] 
                     for i, condition in enumerate(conditions) 
                     if condition
                     ]
            )
            error_message += " must be an integer greater than 0."
            raise ValueError(error_message)
        
        self.n_subjects = n_subjects
        self.n_sessions = n_sessions
        self.n_runs = n_runs
        self.data_folder = data_folder
        self.bids_path = None
        if root:
            self.root = Path(root)
        else:
            self.root = Path(tempfile.mkdtemp())
    
    def _add_participant_metadata(
        self, 
        participant_id: str,
        age: int,
        sex: str,
        handedness: str
        ) -> None:
        
        if not hasattr(self, 'participant_metadata'):
            self._create_participants_metadata()

        temp_df = pd.DataFrame(
            {
                "participant_id": participant_id,
                "age": age,
                "sex": sex,
                "handedness": handedness
            },
            index=[0]
        )

        self.participant_metadata = pd.concat(
            [self.participant_metadata, temp_df],
            ignore_index=True
        )
    
    def create_participant_metadata(self) -> None:
        holder = {
            "participant_id": [],
            "sex": [],
            "age": [],
            "handedness": []
        }
        for subject_number in range(1, self.n_subjects + 1):
            holder['age'].append(np.random.randint(18, 60))
            holder['sex'].append(np.random.choice(['M', 'F']))
            holder['handedness'].append(
                np.random.choice(['right', 'left', 'ambidextrous'])
                )
            holder['participant_id'].append(
                self._generate_label('subject', subject_number)
                )
        
        self.participant_metadata = pd.DataFrame(holder)
        return self
    
    def _save_participant_metadata(self) -> None:
        self.participant_metadata.to_csv(
            os.path.join(self.bids_path, "participants.tsv"),
            sep="\t",
            index=False
        )
                                  
        
    def _create_bids_folder(self: 'DummyDataset') -> 'DummyDataset':
        """Create a BIDS compliant starting folder structure.
        
        The root path created is the system's temporary directory or an
        arbitrary path defined by the user. To make it BIDS compliant, it needs
        to have a 'source' and/or a 'rawdata' and/or a 'derivatives' folder. This
        method creates the bids_path attribute which is a path compliant with 
        the BIDS standard.

        Returns:
            DummyDataset
        """
        self.bids_path = self.root.joinpath(self.data_folder)
        self.bids_path.mkdir(parents=True, exist_ok=True)
        return self
    
    def _generate_label(
        self: 'DummyDataset',
        label_type: str = 'subject',
        label_number: int = 1,
        label_str_id: str = None,
    ) -> str:
        """Generate a BIDS compliant label.
        
        The BIDS standard requires a specific format for the labels of the 
        subject, session, and run folders. This method generates the label
        based on the label_type, label_number, label_str_id, and zero_padding
        parameters.

        Args:
            self (DummyDataset): _description_
            label_type (str, optional): The type of label to generate. It can be
                'subject', 'session', or 'run'. Defaults to 'subject'.
                
            label_number (int, optional): The number of the label. Defaults to 1.
                
            label_str_id (str, optional): The string identifier to add to the label.
                Defaults to None.
                
            zero_padding (int, optional): The number of 0 to add before the label
                number. Defaults to 2.

        Returns:
            str: The generated label.
        """
        label_prefix = label_type[:3] + '-'
        if not label_str_id:
            label_str_id = ''
        label = f"{label_prefix}{label_str_id}{label_number:03d}"
        return label
        
    def _generate_folder_path(self: 'DummyDataset',
                        folder_type: str,
                        folder_number: int = 1,
                        folder_str_id: str = None,
                        zero_padding: int = 2) -> Path:
        """Create multiple BIDS compliant folders.
        
        The BIDS structure requires the structure to be an iterative
        successsion of subject/session folders. This method creates the
        necessary folders (subject or session defined in 'folder_type') 
        and returns the last one created. 

        Args:
            self (DummyDataset): _description_
            folder_type (str): Define the type of folder to create. It can be
                'subject' or 'session'.
                
            folder_number (int, optional): The subject or session number.
                Default to 1.
                
            folder_str_id (str, optional): The subject's or session's string 
                identifier that would precede the subject's or session's number.  
                Defaults to None.

            zero_padding (int, optional): How many 0 to add before the subject's
            or session's number. Defaults to 2.

        Raises:
            ValueError: If the folder_type is not 'subject' or 'session'.

        Returns:
            Path: The path of the folder created
        """
        if folder_type == 'subject':
            parent_path = self.bids_path
        elif folder_type == 'session':
            parent_path = self.subject_path
        else:
            raise ValueError(f"Invalid folder type: {folder_type}")
        
        for folder_id in range(1, folder_number + 1):
            folder_label = self._generate_label(
                label_type = folder_type,
                label_number = folder_id,
                label_str_id = folder_str_id,
            )
            folder_path = parent_path.joinpath(folder_label)
            
        return folder_path
    def _create_sidecar_json(self, 
                             eeg_filename: str | os.PathLike) -> None:
        """Create a sidecar JSON file for the EEG data.

        Args:
            eeg_filename (str | os.PathLike): The EEG data file name.
            
            eeg_folder (str | os.PathLike): The folder containing the EEG data.
        """
        json_filename = Path(os.path.splitext(eeg_filename)[0])
        json_filename.with_suffix('.json')
        
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

        with open(json_filename, 'w') as json_file:
            json.dump(json_content, json_file, indent=4)
    
    def _create_dataset_description(self) -> None:
        """Create the dataset_description.json file."""
        self.dataset_description = {
            "Name": "THIS IS A DUMMY DATASET",
            "BIDSVersion": "1.9.0",
            "License": "CC0",
            "Authors": ["Jane Doe", "John Doe"]
        }

        with open(
            os.path.join(self.bids_path, "dataset_description.json"), 'w'
            ) as desc_file:
            json.dump(self.dataset_description, desc_file, indent=4)
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
            
    def create_eeg_dataset(self,
                           fmt: str = 'brainvision') -> str:
        """Create temporary BIDS dataset.
        
        Create a dummy BIDS dataset for EEG data with multiple subjects, sessions, 
        and runs.

        data_type (str, optional) 'fif' | 'brainvision' | 'edf' | 'eeglab': 
            The format of the EEG data to simulate. Defaults to 'brainvision'.
            

        Returns:
            str: The path of the temporary BIDS dataset.
        """
        # Define the necessary BIDS files for dataset description
        self._create_bids_folder()
        self._create_dataset_description()
        self.create_participants_metadata()

        for subject_number in range(1, self.n_subjects + 1):
            arguments = ['subject', subject_number]
            participant_id = self._generate_label(*arguments)
            self.subject_path = self._generate_folder_path(*arguments)

            for session_number in range(1, self.n_sessions + 1):
                arguments = ['session', session_number]
                session_label = self._generate_label(*arguments)
                self.session_path = self._generate_folder_path(*arguments)
                
                for run_number in range(1, self.n_runs + 1):
                    run_label = self._generate_label('run', run_number)
                    eeg_directory = self.session_path.joinpath('eeg')
                    eeg_directory.mkdir(parents=True, exist_ok=True)

                    # Define file names for EEG data files
                    if fmt == 'brainvision':
                        extension = '.vhdr'
                    elif fmt == 'edf':
                        extension = '.edf'
                    elif fmt == 'eeglab':
                        extension = '.set'
                    elif fmt == 'fif':
                        extension = '.fif'
                        
                    base_eeg_filename = "_".join([
                        participant_id,
                        session_label,
                        run_label,
                        'task-test',
                        'eeg'
                    ])

                    eeg_filename = base_eeg_filename + extension
                    eeg_absolute_filename = eeg_directory.joinpath(eeg_filename)

                    raw = simulate_eeg_data()
                    mne.export.export_raw(
                        fname = eeg_absolute_filename,
                        raw=raw,
                        fmt=fmt,
                    )

                    # Create sidecar JSON file
                    self._create_sidecar_json(eeg_filename)


        self._save_participant_metadata()
        print(f"Temporary BIDS EEG dataset created at {self.bids_path}")
        self.print_bids_tree()
        return self

    def print_bids_tree(self) -> None:
        """Print the BIDS dataset tree."""
        tree = DirectoryTree(self.bids_path)
        tree.print_tree()
        return self