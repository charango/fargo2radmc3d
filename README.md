# FARGO2RADMC3D
Python program that computes synthetic maps of continuum emission and of polarised scattered light with RADMC3D from the results of 2D gas+dust hydrodynamical simulations carried out with the code Dusty FARGO-ADSG. The program can be easily adapted to use the outputs of a different code.

The program works with both Python 2.X and Python 3.X. It requires a parameter file, called 'params.dat', to tell the code what to do (e.g., compute a continuum emission map at 0.9 mm wavelength) and what parameters to use (e.g., the dust's total mass, size distribution etc.). The parameter file needs to be in the parent directory of your working directory.

Dust opacities can be computed with bhmie.py, which is a python version of the Bohren & Huffman Mie code ported by Cornelis Dullemond, and with makedustopac.py, also written by Cornelis Dullemond. Both python programs can be found in the opacity subroutines provided by RADMC3D (v 0.41). An .tar.gz archive containing optical constants (.lnk files) and pre-computed dust continuum opacities (.inp files) is also provided.

Please have a look at the manual (5 pages) to check how the code works and what the different options and parameters in params.dat are.
