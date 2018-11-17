require 'os'
require 'torch'
require 'image'
require 'patch2vec'


patch2vec_init('model_750_50000_32_180804.t7')

nnluainpaint = require("libpatchmatch2")

if arg[1] and arg[2] then
   nnluainpaint.compare_nn_l2(arg[1], arg[2])
else
   print('Usage: th compare_algorithms.lua <first image file> <second image file>')
end
