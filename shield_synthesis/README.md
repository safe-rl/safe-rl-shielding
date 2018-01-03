# Shield Synthesis Tool

## Requirements:
  - Install CU Decision Diagram Package from http://vlsi.colorado.edu/~fabio/ (libcudd and libobj)
  - Either set the *CUDD_PATH* environment variable or in the Makefile
  
## Usage:
  - Use **make** first
  - Afterwards, e.g.: ./shield_synthesizer 3 test.dfa *(Generating a shield for specification file **test.dfa** and ranking size 3)*
  - For multiple specification files, simply provide all files seperated with a space
  
## Notes:
  - This tool is not designed to be userfriendly for now. In case there is an assertion error or a segmentation fault, it is most likely due to errors in the specification files. No warranties however.