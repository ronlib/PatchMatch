=====================================
README:   Inpainting based PatchMatch
=====================================

@Author: Younesse ANDAM

@Contact: younesse.andam@gmail.com

Description: This project is a personal implementation of an algorithm called PATCHMATCH that restores missing areas in an image.
The algorithm is presented in the following paper
 PatchMatch  A Randomized Correspondence Algorithm
               for Structural Image Editing
   by C.Barnes,E.Shechtman,A.Finkelstein and Dan B.Goldman
   ACM Transactions on Graphics (Proc. SIGGRAPH), vol.28, aug-2009

 For more information please refer to
 http://www.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/index.php

Copyright (c) 2010-2011

Requirements
============

To run the project you need to install Opencv library and link it to your project.
Opencv can be download it here
http://opencv.org/downloads.html

How to use
===========

python3 setup.py build_ext --inplace
python3 main.py -f images/forest.bmp


Enjoy!!
