import numpy as np
import xml_parse
import pdb

def xml_to_spiketimes(filepath):
  ''' Calling code written by Manu, this should take the full path to an xml file for a given experiment and get the spike times
  '''
  import xml.etree.ElementTree as ET
  tree = ET.parse(filepath)

  spikes = xml_parse.process_spikes(tree);
  return spikes['spike_times'];

def xml_to_passes(filepath):
  import xml.etree.ElementTree as ET
  tree = ET.parse(filepath)

  passes = xml_parse.process_passes(tree);
  return passes;

def get_times():
  fileNames = ['m658r9#1[ori16]', 'm658r9#2[sf11]'];

  fileNames_plex = [str(x+ '_div10.txt') for x in fileNames];
  fileNames_expo = [str(x+ '.xml') for x in fileNames];

  ### load the expo files
  times_expo = [xml_to_spiketimes(x) for x in fileNames_expo];
  passes     = [xml_to_passes(x) for x in fileNames_expo];
  durations  = [np.array(x['durations'])[np.where(np.array(x['blockIDs'])>0)] for x in passes]; # blockID = 0 is the initialization pass, don't want that

  ### load the sorted (plexon) files
  times_plex = [np.loadtxt(x, skiprows=1, delimiter=',', usecols=(0,)) for x in fileNames_plex];

  return times_expo, times_plex, durations, fileNames;

'''
Comment on expo spike times: in sfMixLGNLoad, we call "GetSpikeTimes.m" with both the
'startTime' and 'endTime' arguments equal to 0.001*latency, where latency is in ms.
Thus, if latency is 40 ms, then this is 0.04 seconds; this value is added to both the beginning
and end of the spike times - so if the trial takes place from 1-2 seconds, then we look for spikes
between 1.04 - 2.04 seconds, but the numbers are then reset to be between [0, dur] seconds, where dur is the duration of the trial
'''
