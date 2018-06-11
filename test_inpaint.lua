require 'os'
require 'torch'
require 'image'
require 'patch2vec'


patch2vec_init('model16_570.t7')

nnluainpaint = require("libpatchmatch2")
nnluainpaint.inpaint('forest-hole.bmp', 'forest-mask.bmp')
