from pandas._libs.tslibs.offsets import Day # Import Day offset from pandas tslibs
from easydict import EasyDict as edict # Import EasyDict for dictionary-like object access
from iv2_utils.iv2 import * # Import all from iv2_utils.iv2 (assuming it contains json_read, pickle_read, pickle_write)
import pandas as pd # Import pandas for data manipulation (though seems unused directly in this snippet)
import numpy as np # Import numpy for numerical operations
import copy # Import copy for deep copying objects
import os # Import os for interacting with the operating system (file/directory operations)

# Mapping of single-character indicators to subfolder names
pred_mapping = {'a': 'augment', 'b': 'bf50', 't': 'act75'}
# Variable 'averaged' seems defined but not used in this snippet
averaged = 100

def generate_settings(file_path):
    """
    Generate settings from the given file path.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    edict: A dictionary containing the subfolder and window size.
    """
    original_path = file_path # Store original path for error message
    if '/' in file_path:
        file_path = file_path.split('/')[-1] # Get just the filename if path includes directories

    assert '.json' in file_path
    file_path = file_path.split('.')[0] # Remove the .pkl extension

    indicator = file_path[0] # Get the first character as the indicator
    try:
        window_size = int(file_path[1:]) # Get the rest of the filename as the window size (integer)
    except:
        print(f"{original_path} doesn't work") # Print error if converting window size fails
        raise Exception # Raise an exception

    # Return settings as an EasyDict
    return edict({'subfolder': pred_mapping[indicator], 'window_size': window_size})

def generate_settings_logits(file_path):
    """
    Generate settings for logits from the given file path.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    edict: A dictionary containing the subfolder.
    """
    if '/' in file_path:
        file_path = file_path.split('/')[-1] # Get just the filename

    if '.pkl' not in file_path and '.json' not in file_path:
        print(file_path) # Print filename if no .pkl extension
    assert('.pkl' in file_path or '.json' in file_path) # Assert that the file has a .pkl extension
    file_path = file_path.split('.')[0] # Remove the .pkl extension

    indicator = file_path[0] # Get the first character as the indicator

    # Return settings (only subfolder) as an EasyDict
    return edict({'subfolder': pred_mapping[indicator]})

class Accessor:
    """
    A class to manage access to data with different categories and window sizes.

    Attributes:
    name (str): The name of the accessor.
    access (edict): A dictionary to store data categorized by window size.
    varieties (edict): A dictionary to store the list of window sizes for each category.
    """
    def __init__(self, name):
        self.access = edict() # Initialize access dictionary (category -> window_size -> DataWrapper)
        self.varieties = edict() # Initialize varieties dictionary (category -> list of window_sizes)
        self.name = name # Store the name of the accessor

    def add(self, category, window_size, data):
        """
        Add data to the specified category and window size.

        Parameters:
        category (str): The category to add the data to.
        window_size (int): The window size for the data.
        data (list): The data to add.
        """
        if category not in self.access:
            self.access[category] = edict() # Create category dict if it doesn't exist

        if category not in self.varieties:
            self.varieties[category] = [] # Create varieties list if it doesn't exist

        l_datawrapper = DataWrapper() # Create a new DataWrapper instance
        for i in data: l_datawrapper.append(i) # Append data items to the DataWrapper
        l_datawrapper.set_metadata(edict({'name': self.name})) # Set metadata for the DataWrapper

        self.access[category][str(window_size)] = l_datawrapper # Store the DataWrapper by category and window_size (as string key)
        self.varieties[category].append(window_size) # Add the window size to the list of varieties for this category

    def get(self, category, window_size):
        """
        Get data from the specified category and window size.

        Parameters:
        category (str): The category to get the data from.
        window_size (int): The window size for the data.

        Returns:
        DataWrapper: The data wrapper containing the data.
        """
        try:
            return self.access[category][str(window_size)] # Return the DataWrapper for the given category and window_size
        except:
            raise NotImplementedError("Doesn't exist, try again.") # Raise error if category or window_size not found
            return -1 # This line is unreachable due to the raise statement

    def __str__(self) -> str:
        return str(self.varieties) # String representation shows the available varieties (window sizes per category)

    def __call__(self, category, window_size):
        return self.get(category, window_size) # Allows accessing data using the object instance directly like a function call

class LogitData:
    """
    A class to manage logit data.

    Attributes:
    name (str): The name of the logit data.
    access (edict): A dictionary to store logit data categorized by category.
    """
    def __init__(self, name):
        self.access = edict() # Initialize access dictionary (category -> DataWrapper)
        self.name = name # Store the name of the logit data object

    def add(self, category, logits):
        """
        Add logits to the specified category.

        Parameters:
        category (str): The category to add the logits to.
        logits (list): The logits to add.
        """
        l_datawrapper = DataWrapper(logits) # Create a DataWrapper with the logits
        l_datawrapper.set_metadata(edict()) # Set empty metadata initially
        l_datawrapper.set_metavar('name', self.name) # Set the name in metadata
        self.access[category] = l_datawrapper # Store the DataWrapper by category

    def get(self, category):
        """
        Get logits from the specified category.

        Parameters:
        category (str): The category to get the logits from.

        Returns:
        DataWrapper: The data wrapper containing the logits.
        """
        try:
            return self.access[category] # Return the DataWrapper for the given category
        except:
            raise NotImplementedError("Doesn't exist, try again.") # Raise error if category not found
            return -1 # This line is unreachable

    def __call__(self, category):
        return self.get(category) # Allows accessing logits using the object instance directly like a function call

class DataWrapper(list):
    """
    A class to wrap data with metadata. Inherits from list.

    Attributes:
    meta (edict): The metadata for the data.
    """
    def set_metadata(self, metadata):
        self.meta = metadata # Set the metadata

    def get_metadata(self):
        return self.meta # Get the metadata

    def set_metavar(self, var, new_val):
        self.meta[var] = new_val # Set a specific variable within the metadata

    @property
    def name(self):
        return self.meta.name # Property to easily access 'name' from metadata

    @property
    def data(self):
        return self.meta.data # Property to easily access 'data' from metadata

    @property
    def window_size(self):
        return self.meta.window_size # Property to easily access 'window_size' from metadata

def load_data():
    """
    Load data from the 'jar' directory.

    Returns:
    edict: A dictionary containing the loaded data.
        - data.act75 and data.bf50 contain data used for eval only.
        - data.act75_full and data.bf50_full contain the phrases used for text-video similarity.
    """
    # List directories in 'jar', filtering out common non-data files/dirs
    models = list(filter(lambda x: x not in ['config.py', '.DS_Store', '__init__.py', '__pycache__'], os.listdir('jar')))
    data = edict() # Initialize an EasyDict to store all loaded data

    data.bf50 = [] # Initialize list for bf50 data
    # Read BF50.json and append (video_id, frames) tuples
    for video, phrase, frames in json_read('rustyjar/BF50.json'):
        data.bf50.append((int(video.split('/')[-1].split('.')[0]), frames))

    data.bf50_full = json_read('rustyjar/BF50.json') # Load the full BF50.json data

    # Assign bf50 data to 'augment' and 'regular' categories
    data.augment = data.regular = data.bf50

    data.act75 = [] # Initialize list for act75 data
    # Read ACT75.json and append (video_id, frames) tuples
    for video, phrase, frames in json_read('rustyjar/ACT75.json'):
        data.act75.append((int(video.split('/')[-1].split('.')[0]), frames))

    data.act75_full = json_read('rustyjar/ACT75.json') # Load the full ACT75.json data

    # Iterate through detected model directories in 'jar'
    for model in models:
        if model[0] == '.': continue # Skip hidden directories
        data[model] = Accessor(model) # Create an Accessor object for each model
        # Iterate through files within the model directory
        for file in os.listdir(os.path.join('jar', model)):
            if file[0] == '.': continue # Skip hidden files
            settings = generate_settings(file) # Generate settings (subfolder, window_size) from filename
            if file.endswith('json'):
                read_data = json_read(os.path.join('jar', model, file))
            else:
                read_data = pickle_read(os.path.join('jar', model, file)) # Read data (logits) from the pickle file

            data[model].add(settings.subfolder, settings.window_size, read_data) # Add the loaded data to the Accessor
            # Set additional metadata on the newly added DataWrapper
            data[model](settings.subfolder, settings.window_size).set_metadata(
                edict({
                    'name': model, # Model name
                    'data': data[settings.subfolder], # Reference to the base data (bf50 or act75)
                    'window_size': settings.window_size # Window size
                })
            )

    return data # Return the EasyDict containing all loaded data

def load_logits():
    """
    Load logits from the 'rustyjar' directory.

    Returns:
    edict: A dictionary containing the loaded logits.
    """
    # List directories in 'rustyjar' (assumed to be model directories)
    models = [name for name in os.listdir('rustyjar') if os.path.isdir(os.path.join('rustyjar', name))]

    logits = edict() # Initialize an EasyDict to store loaded logits

    # Iterate through model directories in 'rustyjar'
    for model in models:
        if model[0] == '.': continue # Skip hidden directories
        logits[model] = LogitData(model) # Create a LogitData object for each model
        # Iterate through files within the model directory
        for file in os.listdir(os.path.join('rustyjar', model)):
            # print(file) # Debug print (commented out)
            if file[0] == '.': continue # Skip hidden files
            settings = generate_settings_logits(file) # Generate settings (subfolder) from filename

            # Read data (logits)
            if file.endswith('json'):
                read_data = json_read(os.path.join('rustyjar', model, file))
            else:
                read_data = pickle_read(os.path.join('rustyjar', model, file)) # Read data (logits) from the pickle file

            logits[model].add(settings.subfolder, read_data) # Add the loaded logits to the LogitData object

    return logits # Return the EasyDict containing all loaded logits

# split is 'r', 'a', etc.
# model name is 'ViCLIP', 'B14', etc.
# logits_list is a list of logits (presumably [([score, frame_idx], ...), ...])
def synthesize_logits(model_name, split, logits_list, window_size = 8):
    """
    Synthesize logits for a given model and split.
    Processes a list of logits, applies a sliding window average, and finds the peak.

    Parameters:
    model_name (str): The name of the model.
    split (str): The split identifier.
    logits_list (list): A list of logits, where each logit entry is a list of (score, frame_index) tuples.
    window_size (int): The window size for synthesis (averaging).

    Returns:
    list: A list of predicted frame indices (peak locations) for each item in logits_list.
    """
    result = [] # Initialize list to store the resulting synthesized frame indices
    for logits in logits_list: # Iterate through each item in the input list (each item is a list of (score, frame_index) tuples)
        logits_c = copy.deepcopy(logits) # Create a deep copy to avoid modifying the original list
        logits_c.sort(key = lambda x: x[1]) # Sort the logits by frame index
        new_logits = [] # Initialize list to store averaged logits and their center frame index
        add = (window_size -8) // 2 # Calculate the number of frames to add on each side for averaging (relative to an assumed base window of 8)
        # Iterate through the sorted logits, considering the window size
        for j in range(add, len(logits_c) - add):
            a_range = list(range(j - add, j + add + 1)) # Define the index range for the current window
            a_range = [logits_c[x][0] for x in a_range] # Get the scores within the current window range
            new_logits.append((np.mean(a_range).item(), j + 1)) # Calculate the mean score, store it with the center frame index (j+1 assuming 1-based indexing)

        new_logits.sort(key = lambda x: -x[0]) # Sort the new_logits list by mean score in descending order
        if len(new_logits) == 0:
            new_logits.append((0, 4)) # If no logits were processed (e.g., list too short for window), add a default result (score 0, frame 4)
        result.append(new_logits[0][1]) # Append the frame index (which had the highest average score) to the result list

    output_path = os.path.join('jar', model_name, f'{split}{window_size}.json') # Construct the output file path
    json_write(result, output_path) # Write the synthesized results (list of frame indices) to a pickle file
    print("Wrote output to", output_path) # Print confirmation message
    return result # Return the list of synthesized frame indices
