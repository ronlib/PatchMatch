require 'os'
require 'torch'
require 'image'
require 'patch2vec'


patch2vec_init('model_750_50000_32_180804.t7')

nnluainpaint = require("libpatchmatch2")
nnluainpaint.inpaint(
   -- '/home/ron/studies/project/PatchMatch/birds.png',
   -- '/home/ron/studies/project/PatchMatch/birds-mask.png',
   '/home/ron/studies/project/PatchMatch/image_files/forest/forest.bmp',
   '/home/ron/studies/project/PatchMatch/forest-mask-50-points.bmp',
   -- '/home/ron/studies/project/PatchMatch/image_files/forest/forest_mask.bmp',
   1,         -- nn_dist
   32,        -- patch_w, must match the neural network in use
   1,         -- inpaint_add_completion_term
   1,         -- inpaint_use_full_image_coherence
   7,         -- nn_iters
   8,         -- inpaint_border
   0,         -- inpaint_min_level
   50         -- threshold
)
