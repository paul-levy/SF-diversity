import numpy as np
from collections import defaultdict
''' I am using this script to test the timing between the spikes from expo and the spike times in the output of the sorting from the Plexon offline sorter.
    The question to answer is two-fold: one, is there a simple relationship between the spike times in the two files, and the second is what is the time offset/relationship!
'''

def process_blocks(tree):
    """Extract block parameters
    Go through the XML extracting information from <block> subsections that
    define any given expo program
    Parameters
    ----------
    tree: xml.etree.ElementTree.ElementTree
        output of a parsed xml document using xml library
    Returns
    ----------
    output: defaultdict
        a 3 element dictionary with IDs, Names, and Routines. As in the output
        of readExpoXML. Some differences are that there are no routineMaps, or
        routinesEntries. These are summary information that can be added if
        needed via additional modules.
    """
    block = tree.findall("Blocks/Block")
    stored_routine_information = []
    block_information = defaultdict(list)
    for block_elements in block:
        #first find info for block IDs and names
        for block_key,block_value in block_elements.items():
            block_information[block_key].append(block_value)
        #next find routine IDs, Names, Labels, and Params
        routine_parameters = [];
        routine_information = defaultdict(list)
        for routines in block_elements:
            #find base routine info (IDs, Names, Labels)
            for routine_key,routine_value in routines.items():
                routine_information[routine_key].append(routine_value)
            #find routine Params
            stored_parameters = defaultdict(list);
            for params in routines:
                for param_key,param_value in params.items():
                    stored_parameters[param_key].append(param_value)
            routine_parameters.append(stored_parameters)
        #add parameter information to routine_information and store
        routine_information['Params'] = routine_parameters
        stored_routine_information.append(routine_information)
    IDs = list(map(int,block_information['ID']))
    Names = block_information['Name']
    output = defaultdict(list)
    output['IDs'] = IDs;
    output['Names'] = Names;
    output['Routines'] = stored_routine_information
    return output

def process_passes(tree):
    """Extract pass parameters
    Go through the XML extracting information from <pass> subsections that
    define any given expo program
    Parameters
    ----------
    tree: xml.etree.ElementTree.ElementTree
        output of a parsed xml document using xml library
    Returns
    ----------
    output: defaultdict
        a 6 element dictionary with ID, SlodID, BlockID, Start Time, End Time,
        Events.  Each set of keys in the dictionary maps to a set of values
        equivalent to the length of the number of passes. This is organized
        similarly to the output of readExpoXML.
    """
    passes = tree.findall("Passes/Pass")
    stored_pass_events = [];
    stored_pass_basics = defaultdict(list)
    for pass_elements in passes:
        event_store = defaultdict(list)
        #find core information about passes (BlockIDs,Start/End times, Pass, Slot,
        #Block IDs)
        for pass_key,pass_value in pass_elements.items():
            stored_pass_basics[pass_key].append(pass_value)
        #now find <event> information, like specific orientation etc.
        for events in pass_elements:
            for event_key,event_value in events.items():
                event_store[event_key].append(event_value)
        stored_pass_events.append(event_store)
    #populate dictionary for output
    output = stored_pass_basics
    #add event information to output dictionary
    output['events'] = stored_pass_events
    return output

def process_frametimes(tree):
    """Extract tick time parameters (frametimes)
    Go through the XML extracting information from <timeline> subsections that
    define any given expo program
    Parameters
    ----------
    tree: xml.etree.ElementTree.ElementTree
        output of a parsed xml document using xml library
    Returns
    ----------
    output: defaultdict
        a 7 element dictionary with start_times, flush_times, end_times,
        expo_start_time, expo_end_time, tBO, and nTicks Events. This is
        organized similarly to the output of readExpoXML, with the exception
        that the superfluous displayTimes is no longer extracted
    """
    timeline = tree.findall("timeline")
    timeline = timeline[0];
    basic_info = timeline.attrib
    stored_tick_information = defaultdict(list)
    for ticks in timeline:
        for tick_key,tick_values in ticks.items():
            stored_tick_information[tick_key].append(int(tick_values)) #NOTE, converting things to integers.
    #subtract start time from start, flush and end times and convert to 1/10 ms precision
    stored_tick_information.update({'start':[(x-int(basic_info['startTime']))/100000 for x in stored_tick_information['start']]})
    stored_tick_information.update({'flush':[(x-int(basic_info['startTime']))/100000 for x in stored_tick_information['flush']]})
    stored_tick_information.update({'end':[(x-int(basic_info['startTime']))/100000 for x in stored_tick_information['end']]})
    output = stored_tick_information
    nticks = len(stored_tick_information['start'])
    output['nticks'] = nticks;
    output['expo_start_time'] = 0
    output['expo_end_time'] = (int(basic_info['endTime'])-int(basic_info['startTime']))/100000
    return output

def process_spikes(tree):
    # NOTE: Hacky - should be updated to be aligned with coding style above
    """Extract spikes from xml file
    Parameters
    ----------
    tree: xml.etree.ElementTree.ElementTree
        output of a parsed xml document using xml library
    Returns
    ----------
    output: defaultdict
        a 7 element dictionary with start_times, flush_times, end_times,
        expo_start_time, expo_end_time, tBO, and nTicks Events. This is
        organized similarly to the output of readExpoXML, with the exception
        that the superfluous displayTimes is no longer extracted
    """
    root = tree.getroot();
    spks = root.find('Spikes')

    n_spks = int(spks.attrib['Total']); # attrib is a dictionary with one key - total # of spikes
    spks_chil = spks.getchildren(); # should have only one child
    spks_chil_attrib = spks_chil[0].attrib;
    # assumes only one spike template and channel
    ID     = int(spks_chil_attrib['ID']);
    chan   = int(spks_chil_attrib['Channel']);
    ms_to_s = np.power(10, 4); # this is the conversion factor from spike time values in the xml (msec) to actual time in s
    times_in_ms  = np.fromstring(spks_chil_attrib['Times'], sep=','); # go from string (with separator ",") to array of time values
    times_in_s = times_in_ms/ms_to_s;

    output = defaultdict(list);
    output['n_spikes'] = n_spks;
    output['ID'] = ID;
    output['ID'] = chan;
    output['spike_times'] = times_in_s;

    return output;

def process_passes(tree):    
    ''' Returns
    '''
    root = tree.getroot();
    passes = root.find('Passes');

    ms_to_s = 1e4;

    start_times = [float(i.attrib['StartTime']) for i in passes.getchildren()];
    end_times = [float(i.attrib['EndTime']) for i in passes.getchildren()];
    durations = [(i-j)/ms_to_s for i,j in zip(end_times, start_times)];

    blockIDs = [float(i.attrib['BlockID']) for i in passes.getchildren()];

    output = defaultdict(list);
    output['blockIDs'] = blockIDs;
    output['durations'] = durations;

    return output;
