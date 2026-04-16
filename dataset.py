import os
import glob
import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_utils import parse_metadata

class EEGDataset(Dataset):
    """
    Dataset for loading EEG EDF files and segmenting them into windows.
    Each window is labeled as Seizure (1) or Normal (0).
    """
    def __init__(self, data_root, metadata, patient_id=None, window_sec=2.0, step_sec=1.0, max_files=None):
        self.data_root = data_root
        self.metadata = metadata
        self.window_sec = window_sec
        self.step_sec = step_sec
        
        # Filter files by patient if provided
        self.files = []
        for fname, meta in metadata.items():
            if patient_id is None or meta["patient"] == patient_id:
                # Try multiple possible paths due to potential naming inconsistencies (e.g. PN01.edf vs PN01-1.edf)
                possible_paths = [
                    os.path.join(data_root, fname, fname),
                    os.path.join(data_root, fname.replace(".edf", "-1.edf"), fname.replace(".edf", "-1.edf")),
                    os.path.join(data_root, fname, fname.replace(".edf", "-1.edf")),
                ]
                
                found = False
                for fpath in possible_paths:
                    if os.path.exists(fpath):
                        self.files.append((fpath, meta))
                        found = True
                        break
                
                if not found:
                    # Fallback: search for any .edf in a folder matching the start of the filename
                    base_name = fname.split('.')[0] # e.g. PN01
                    folders = glob.glob(os.path.join(data_root, f"{base_name}*"))
                    for folder in folders:
                        edfs = glob.glob(os.path.join(folder, "*.edf"))
                        if edfs:
                            self.files.append((edfs[0], meta))
                            break
        
        if max_files:
            self.files = self.files[:max_files]
            
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        """Segments files into windows and assigns labels."""
        if not self.files:
            print("Warning: No files found for this dataset!")
            return
            
        print(f"Preparing samples for {len(self.files)} files...")
        
        for fpath, meta in self.files:
            try:
                # Load EDF file
                raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
                sfreq = raw.info['sfreq']
                data = raw.get_data() # (channels, samples)
                
                # Take 20 channels
                data = data[:20, :] 
                
                # Z-score normalization
                means = np.mean(data, axis=1, keepdims=True)
                stds = np.std(data, axis=1, keepdims=True) + 1e-6
                data = (data - means) / stds
                
                win_pts = int(self.window_sec * sfreq)
                step_pts = int(self.step_sec * sfreq)
                
                reg_start_sec = meta["reg_start"]
                seizures = meta["seizures"]
                
                # Slide the window
                for start in range(0, data.shape[1] - win_pts, step_pts):
                    end = start + win_pts
                    win_start_time = reg_start_sec + (start / sfreq)
                    win_end_time = reg_start_sec + (end / sfreq)
                    
                    label = 0
                    for s in seizures:
                        if not (win_end_time < s["start"] or win_start_time > s["end"]):
                            label = 1
                            break
                    
                    self.samples.append((data[:, start:end].astype(np.float32), label))
                raw.close()
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), y

def get_eeg_data(data_root, metadata_file, patient_id=None, max_files=None):
    metadata = parse_metadata(metadata_file)
    dataset = EEGDataset(data_root, metadata, patient_id=patient_id, max_files=max_files)
    return dataset

def get_dataloader(dataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    # If dataset is empty, DataLoader will fail. We return a dummy if needed?
    # No, better to let it fail or handle in caller.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
