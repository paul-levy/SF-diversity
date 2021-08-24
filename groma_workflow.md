# How to use Groma
As of m680ish and beyond, all of the acute recordings are stored on Groma. This will explain how to access that data and transfer it to the usual directories for analysis.

### Getting the core experiments
You can browse the files on groma by "Go--Connect to Server" and setting
'''
afp://groma.cns.nyu.edu
'''

as the server directory. From here, the data is stored on /acute.

Most recently, we've been running the Bauman+Bonds-inspired experiments, with the root name sfBB_*. To see those files, simply run an _ls_ command from within the corresponding experiment direcory (e.g. m685_V1V2/expo_files/recordings/). To copy those files over, it's best to use
'''
scp FILE_ROOT DIR_TO_USE
'''
where FILE_ROOT is most recently *sfBB*xml (we only need the xml files) and DIR_TO_USE is plevy@publio.cns.nyu.edu:/users/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/V1_BB/recordings/

### Converting the XML into NPY, creating the datalist

On any of the "usual" machines (e.g. Publio, Sesto), go to the usual SF_diversity directory, then ExpoAnalysisTools/python. There, load the conda environment (lcv-python [salp] NOT pytorch [sapy], due to numpy/pickling issues), call python3.6, and load read_sfBB.py. Once you're there, you'll call read_sfBB_all, specifying the path of the data (should be ../../V1_BB/recordings/, relative to ExpoAnalysisTools/python/) and the output datalist name. You can also specify whether or not to update the datalist at all, or to overwrite existing npy files. This should be converted into a command line call rather than necessiating the loading of the module from python (shrug).

### Getting the basic programs for each cell

At this point, you can already analyze the sfBB* experiments using the existing code (e.g. fits with model_responses_pytorch.py, plots with plot_sfBB.py, or even the Jupyter sandbox_sfBB.ipynb). But, to further round-out the analysis, it's good to have the associated basic analysis programs for a given cell.

To do this, log into **arindal** (not zemina, due to python2 vs python3 issues) as stimulus, and navigate to /v/analysis/paul/py_helper/. You'll first need to transfer the dataList to the same directory. Then, load the right conda environment (_conda activate mountainlab_, which is aliased as _caml_), start python, and import build_basics_list.py. You'll also need to load the datalist which you've transfered. Since there's no helper_fcns there, just use numpy, (i.e. datalist = np.load(dl_name, encoding='latin1').item()). When using the SCP helper function, you MUST transfer to sesto, since there are the ssh keys from zemina to sesto that will avoid having to enter the password for each file. See the functions in build_basics_list.py (as in /v/analysis/...) for details.

### Updating the datalist to gather the associated basic programs

Then, go back to your usual machine (e.g. sesto, publio), and run the code in buildDataList from the section with the comment labeled IF MANUAL EDIT OF DATALIST...


