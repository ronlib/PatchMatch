require 'os'
require 'torch'
require 'image'
require 'patch2vec'


patch2vec_init('model16_570.t7')

nnluainpaint = require("libpatchmatch2")

if arg[1] and arg[2] then
   --nnluainpaint.compare_nn_l2("Jackass_3D_0995_scaled.bmp", "Jackass_3D_1026_scaled.bmp")
   nnluainpaint.compare_nn_l2(arg[1], arg[2])
else
   print('Usage: th compare_algorithms.lua <first_file> <second_file>')
end
