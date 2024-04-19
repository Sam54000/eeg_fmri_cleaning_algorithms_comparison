import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import mne
import numpy as np
import pandas as pd
from mne import annotation_from_events, create_info, make_fixed_length_events
from mne.io import RawArray
from mne.utils import check_random_state
from path_handler import DirectoryTree


def simulate_eeg_data(
    n_channels: int = 32,
    duration_seconds: int = 2,
    sampling_frequency: int = 256,
    events_name: Optional[str] = 'R128'
) -> RawArray:
    """Simulate EEG data.

    This function generates simulated EEG data.

    Args:
        n_channels (int): The number of EEG channels.
        duration_seconds (int): The duration of the EEG data in seconds.
        sampling_frequency (int): The sampling frequency of the EEG data.
        events_name (str, optional): The name of the events. Defaults to 'R128'.

    Returns:
        RawArray: The simulated EEG data.
    """
    if n_channels <= 0:
        raise ValueError("The number of channels must be greater than 0.")

    if duration_seconds <= 0:
        raise ValueError("The duration must be greater than 0.")

    n_samples = duration_seconds * sampling_frequency

    data = np.random.rand(n_channels, n_samples)
    info = create_info(n_channels, sampling_frequency, ch_types="eeg")
    raw = RawArray(data, info)
    if events_name:
        events = make_fixed_length_events(
            raw, 
            id=1, 
            duration=0, 
            overlap=0, 
            start=0
        )
        raw.add_events(events)
        annotations = annotation_from_events(
            events=events, 
            sfreq=sampling_frequency,
            event_desc=events_name
        )
        raw.set_annotations(annotations)
    
    return raw


class DummyDataset:
    """A class to create a dummy BIDS dataset for EEG data.
    
    This class creates a dummy BIDS dataset in order to test pipelines.
    The dataset is generated in the temporary folder of the computer.
    Once tests are done on the dataset, it is possible to remove it from memory
    using the flush method. 
    """
    def __init__(
        self, 
        n_subjects: int = 1, 
        n_sessions: int = 1, 
        n_runs: int = 1,
        sessions_label_str: Optional[str] = None,
        subjects_label_str: Optional[str] = None,
        data_folder: str = "RAW",
        root: Optional[Union[str, os.PathLike]] = None,
        testing: bool = True
    ) -> None:
        """Initialize the DummyDataset object.

        Args:
            n_subjects (int, optional): The number of subjects to simulate. 
                Defaults to 1.
            n_sessions (int, optional): The number of sessions to simulate. 
                Defaults to 1.
            n_runs (int, optional): The number of runs to simulate. 
                Defaults to 1.
            sessions_label_str (str, optional): The string identifier to add to 
                the session label. Defaults to None.
            subjects_label_str (str, optional): The string identifier to add to 
                the subject label. Defaults to None.
            data_folder (str, optional): The location of the data 
                source (rawdata, derivatives). Defaults to "RAW".
            root (str | os.PathLike, optional): The root directory to create 
                the temporary dataset. If None, the dataset is created in the 
                temporary directory of the system. Defaults to None.
            testing (bool, optional): Whether the dataset is being used for 
                testing purposes. Defaults to True.
        """
        arguments_to_check = [n_subjects, n_sessions, n_runs]
        arguments_name = ['subjects', 'sessions', 'runs']
        print(arguments_name)
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
        self.sessions_label_str = sessions_label_str
        self.subjects_label_str = subjects_label_str
        if testing and root:
            self.root = Path(root)
        else:
            self.temporary_directory = tempfile.TemporaryDirectory(
                suffix='suf',
                prefix='temporary_directory_generated_', 
                dir=root, 
                ignore_cleanup_errors=False,
                delete=True
            )
            self.root = Path(self.temporary_directory.name)
        self.bids_path = self.root.joinpath(self.data_folder)
    
    def _add_participant_metadata(
        self, 
        participant_id: str,
        age: int,
        sex: str,
        handedness: str
    ) -> None:
        """Add participant metadata to the dataset.

        Args:
            participant_id (str): The participant ID.
            age (int): The age of the participant.
            sex (str): The sex of the participant.
            handedness (str): The handedness of the participant.
        """
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
    
    def create_participants_metadata(self) -> 'DummyDataset':
        """Create participant metadata for the dataset.

        Returns:
            DummyDataset: The DummyDataset object.
        """
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
        """Save the participant metadata to a file."""
        saving_filename = self.bids_path.joinpath("participants.tsv")
        self.participant_metadata.to_csv(
            saving_filename,
            sep="\t",
            index=False
        )
        
    def _generate_label(
        self: 'DummyDataset',
        label_type: str = 'subject',
        label_number: int = 1,
        label_str_id: Optional[str] = None,
    ) -> str:
        """Generate a BIDS compliant label.
        
        The BIDS standard requires a specific format for the labels of the 
        subject, session, and run folders. This method generates the label
        based on the label_type, label_number, and label_str_id parameters.

        Args:
            label_type (str, optional): The type of label to generate. It can be
                'subject', 'session', or 'run'. Defaults to 'subject'.
            label_number (int, optional): The number of the label. Defaults to 1.
            label_str_id (str, optional): The string identifier to add to the label.
                Defaults to None.

        Returns:
            str: The generated label.
        """
        label_prefix = label_type[:3] + '-'
        if not label_str_id:
            label_str_id = ''
        label = f"{label_prefix}{label_str_id}{label_number:03d}"
        return label
        
    def create_modality_agnostic_dir(
        self: 'DummyDataset'
    ) -> List[Path]:
        """Create multiple BIDS compliant folders.
        
        The BIDS structure requires the structure to be an iterative
        succession of subject/session folders. This method creates the
        necessary folders (subject or session defined in 'folder_type') 
        and returns the last one created. 

        Returns:
            List[Path]: The paths of the created folders.
        """
        path_list = list()
        
        for subject_number in range(1, self.n_subjects + 1):
            subject_folder_label = self._generate_label(
                label_type='subject',
                label_number=subject_number,
                label_str_id=self.subjects_label_str
            )
            
            for session_number in range(1, self.n_sessions + 1):
                session_folder_label = self._generate_label(
                    label_type='session',
                    label_number=session_number,
                    label_str_id=self.sessions_label_str
                )
                path = self.bids_path.joinpath(
                    subject_folder_label, 
                    session_folder_label
                )
                path_list.append(path)
                path.mkdir(parents=True, exist_ok=True)
        
        return path_list
    
    def _extract_entities_from_path(
        self, 
        path: Union[str, os.PathLike]
    ) -> str:
        """Extract the entities from a path.

        Args:
            path (str | os.PathLike): The path to extract the label from.

        Returns:
            str: The extracted label.
        """
        path = Path(path)
        parts = path.parts
        entities = {
            'subject': [part for part in parts if 'sub-' in part][0],
            'session': [part for part in parts if 'ses-' in part][0],
        }

        return entities
    
    def _create_sidecar_json(
        self, 
        eeg_filename: Union[str, os.PathLike]
    ) -> None:
        """Create a sidecar JSON file for the EEG data.

        Args:
            eeg_filename (str | os.PathLike): The EEG data file name.
        """
        json_filename = Path(os.path.splitext(eeg_filename)[0])
        json_filename = json_filename.with_suffix('.json')
        
        json_content = {
          "SamplingFrequency": 2400,
          "Manufacturer": "Brain Products",
          "ManufacturersModelName": "BrainAmp DC",
          "CapManufacturer": "EasyCap",
          "CapManufacturersModelName": "M1-ext",
          "PowerLineFrequency": 50,
          "EEGReference": "single electrode placed on FCz",
          "EEGGround": "placed on AFz",
          "SoftwareFilters": {
              "Anti-aliasing filter": {
              "half-amplitude cutoff (Hz)": 500,
              "Roll-off": "6dB/Octave"
              }
          },
          "HardwareFilters": {
              "ADC's decimation filter (hardware bandwidth limit)": {
              "-3dB cutoff point (Hz)": 480,
              "Filter order sinc response": 5
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
    
    def flush(self, check: bool = True) -> None:
        """Remove the temporary directory from memory.

        Args:
            check (bool, optional): Whether to check the directory before removal.
                Defaults to True.
        """
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
    
    def create_eeg_dataset(
        self,
        fmt: str = 'brainvision'
    ) -> str:
        """Create temporary BIDS dataset.
        
        Create a dummy BIDS dataset for EEG data with multiple subjects, sessions, 
        and runs.

        Args:
            fmt (str, optional): The format of the EEG data to simulate. 
                Defaults to 'brainvision'.

        Returns:
            str: The path of the temporary BIDS dataset.
        """
        path_list = self.create_modality_agnostic_dir()
        self._create_dataset_description()
        self.create_participants_metadata()
         
        for path in path_list:
            for run_number in range(1, self.n_runs + 1):
                run_label = self._generate_label('run', run_number)
                eeg_directory = path.joinpath('eeg')
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
                    
                entities = self._extract_entities_from_path(path)

                eeg_filename = "_".join([
                    entities['subject'],
                    entities['session'],
                    'task-test',
                    run_label,
                    'eeg',
                ])

                eeg_filename += extension
                eeg_absolute_filename = eeg_directory.joinpath(eeg_filename)

                raw = simulate_eeg_data()
                mne.export.export_raw(
                    fname=eeg_absolute_filename,
                    raw=raw,
                    fmt=fmt,
                    overwrite=True
                )

                # Create sidecar JSON file
                self._create_sidecar_json(eeg_absolute_filename)


        self._save_participant_metadata()
        print(f"Temporary BIDS EEG dataset created at {self.bids_path}")
        self.print_bids_tree()
        return self
    
    def print_bids_tree(self) -> None:
        """Print the BIDS dataset tree."""
        tree = DirectoryTree(self.bids_path)
        tree.print_tree()
