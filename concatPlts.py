# importing required modules
import sys
import PyPDF2
import os
import pdb

def PDFrotate(origFileName, newFileName, rotation):
 
    # creating a pdf File object of original pdf
    pdfFileObj = open(origFileName, 'rb')
     
    # creating a pdf Reader object
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
 
    # creating a pdf writer object for new pdf
    pdfWriter = PyPDF2.PdfFileWriter()
     
    # rotating each page
    for page in range(pdfReader.numPages):
 
        # creating rotated page object
        pageObj = pdfReader.getPage(page)
        pageObj.rotateClockwise(rotation)
 
        # adding rotated page object to pdf writer
        pdfWriter.addPage(pageObj)
 
    # new pdf file object
    newFile = open(newFileName, 'wb')
     
    # writing rotated pages to new file
    pdfWriter.write(newFile)
 
    # closing the original pdf file object
    pdfFileObj.close()
     
    # closing the new pdf file object
    newFile.close()

def PDFmerge(pdfs, output): 
    # creating pdf file merger object
    pdfWriter = PyPDF2.PdfFileWriter()
     
    # appending pdfs one by one
    for pdf in pdfs:
      f = open(pdf, 'rb');
      pdfReader = PyPDF2.PdfFileReader(f);
      #number_of_pages = pdfReader.getNumPages();
      #print(number_of_pages);
      try: # try to get the page
        curr_page = pdfReader.getPage(0); # first page [0]; second page [1]; etc
        pdfWriter.addPage(curr_page);
      except: # but it's ok if we have to skip (e.g. PDF with fewer conditions/pages)
        pass;
    
    # writing combined pdf to output pdf file
    with open(output, 'wb') as f:
        pdfWriter.write(f)

def PDFwrapper(folder, output):
    
    all_files = sorted(os.listdir(os.path.abspath(folder)));
    to_merge = [];
    hid_file = '.';
    for f in all_files:
        # vvv adjust here!!! vvv
        if 'cell' in f and f[0] != hid_file and 'allCons' in f: # allCons_log
            to_merge.append(str(folder + f));

    PDFmerge(to_merge, str(folder + output));

    return to_merge;

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print('Need two arguments! Folder to look into, output file name');
    
    folder = sys.argv[1];
    output_name = sys.argv[2];

    PDFwrapper(folder, output_name);
