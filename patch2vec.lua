require 'check_for_cuda'
require 'dpnn'

if (loadrequire('cunn') == 0) then
   require 'nn'
   cuda = 0
else
   require 'cunn'
   require 'cudnn'
   cuda = 1
end

require 'image'


function patch2vec_init(nn_file_path)
   if (cuda == 1) then
      NN = torch.load(nn_file_path):cuda()
      print 'Using CUDA'
   else
      NN = torch.load(nn_file_path)
      print 'Not using CUDA'
   end
end

function create_tensor_from_image_storage(storage, H, W, num_channels)
   -- local t = torch.ByteTensor(storage):float()/255
   torch.setnumthreads(1)
   local t
   if (cuda == 1) then
      t = torch.ByteTensor(storage):float():cuda()/255
   else
      t = torch.ByteTensor(storage):float()/255
   end
   t = t:view(H, W, num_channels):transpose(2,3):transpose(1,2)
   t = t:contiguous()
   -- Adding 1 to fit to the NN input dimensions
   t = t:view(1, num_channels, H, W)
   return t
end

function compute_patch2vec(patch1_storage_obj, H, W, num_channels)
   local ffi = require 'ffi'
   local patch1 = create_tensor_from_image_storage(patch1_storage_obj, H, W, num_channels)
   local v1 = NN:forward(patch1):clone():float()
   return v1:storage(), tonumber(ffi.cast('intptr_t', v1:data()))
end

function compute_patches_distance_NN(patch1_storage_obj, patch2_storage_obj, H, W, num_channels)
   local patch1 = create_tensor_from_image_storage(patch1_storage_obj, H, W, num_channels)
   local patch2 = create_tensor_from_image_storage(patch2_storage_obj, H, W, num_channels)
   local v1 = NN:forward(patch1):clone()
   local v2 = NN:forward(patch2):clone()
   -- Distance is 1-similarity
   local distance = torch.sqrt((v1-v2)*(v1-v2))
   return distance
end

-- This function receives storage object, its current dimensions, and the new
-- dimensions the image should be scaled to
-- H, W - the current dimensions
-- sH, sW - the scaled image dimensions

function scale_image(image_byte_storage, H, W, Hs, Ws, num_channels)
   local ffi = require 'ffi'
   local t = torch.ByteTensor(image_byte_storage):float()/255

   -- Transforming the image to the format of 'image' library
   t = t:view(H, W, num_channels)
   t = t:transpose(1,3):transpose(2,3):contiguous()
   local t2 = image.scale(t, Ws, Hs)*255

   -- Transforming it back to what the c code image format
   t2 = t2:byte()
   t2 = t2:transpose(2,3):transpose(1,3)
   t2 = t2:contiguous()

   -- Returning t2:storage() in order to keep a reference to it back in the
   -- calling code
   return t2:storage(), tonumber(ffi.cast('intptr_t', t2:data()))
end
