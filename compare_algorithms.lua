require 'os'
require 'torch'
require 'image'
require 'patch2vec'


patch2vec_init('model16_570.t7')

nnluainpaint = require("libpatchmatch2")
nnluainpaint.compare_nn_l2("Jackass_3D_0995_scaled.bmp", "Jackass_3D_1026_scaled.bmp")
