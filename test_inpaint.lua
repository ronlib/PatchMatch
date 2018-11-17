require 'os'
require 'torch'
require 'image'
require 'patch2vec'


patch2vec_init('model_750_50000_32_180804.t7')

nnluainpaint = require("libpatchmatch2")
nnluainpaint.inpaint('forest-hole.bmp', 'forest-mask.bmp', 1, 2, 0, 100)
