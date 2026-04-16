import re
import os
from datetime import datetime

def parse_time(time_str):
    """Parses time strings like 19.39.33 or 19:58:36 into seconds from 00:00:00."""
    time_str = time_str.strip().replace('.', ':')
    try:
        t = datetime.strptime(time_str, "%H:%M:%S")
        return t.hour * 3600 + t.minute * 60 + t.second
    except ValueError:
        return None

def parse_metadata(file_path):
    """
    Parses epilipsysample.txt to extract patient metadata.
    Returns a dict mapping filename to its metadata (reg_start, seizures).
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    metadata = {}
    current_patient = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect patient blocks (PN00, PN01, PN05)
        if re.match(r'^PN\d+$', line):
            current_patient = line
            i += 1
            continue
            
        # Detect file information
        if "File name:" in line:
            filename = line.split("File name:")[1].strip()
            # If extension is missing, add it
            if not filename.endswith(".edf"):
                filename += ".edf"
                
            reg_start_line = lines[i+1].strip()
            if "Registration start time:" in reg_start_line:
                reg_start_str = reg_start_line.split("Registration start time:")[1].strip()
                reg_start_sec = parse_time(reg_start_str)
                
                seizures = []
                # Look ahead for seizures linked to this file
                j = i + 2
                while j < len(lines) and "File name:" not in lines[j] and not re.match(r'^PN\d+$', lines[j]):
                    if "Seizure start time:" in lines[j]:
                        start_str = lines[j].split("Seizure start time:")[1].strip()
                        end_str = lines[j+1].split("Seizure end time:")[1].strip()
                        seizures.append({
                            "start": parse_time(start_str),
                            "end": parse_time(end_str)
                        })
                        j += 1
                    elif "Start time:" in lines[j] and current_patient == "PN01": # PN01 has a different format
                        start_str = lines[j].split("Start time:")[1].strip()
                        end_str = lines[j+1].split("End time:")[1].strip()
                        seizures.append({
                            "start": parse_time(start_str),
                            "end": parse_time(end_str)
                        })
                        j += 1
                    j += 1
                
                metadata[filename] = {
                    "patient": current_patient,
                    "reg_start": reg_start_sec,
                    "seizures": seizures
                }
        i += 1
        
    return metadata

if __name__ == "__main__":
    # Test the parser
    import json
    meta = parse_metadata("epilipsysample.txt")
    print(json.dumps(meta, indent=2))
