require 'os'
require 'torch'
require 'image'
require 'patch2vec'


patch2vec_init('model_750_50000_32_180804.t7')

nnluainpaint = require("libpatchmatch2")
nnluainpaint.inpaint('/home/ron/studies/project/PatchMatch/image_files/forest/forest.bmp',
                     '/home/ron/studies/project/PatchMatch/image_files/forest/forest_mask.bmp',
                     5,         -- nn_iters
                     12,         -- inpaint_border
                     0,         -- inpaint_min_level
                     50         -- threshold
)
