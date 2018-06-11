require 'check_for_cuda'

if (loadrequire('cunn') == 0) then
   require 'nn'
   require 'dpnn'
else
   require 'cunn'
   require 'cudpnn'
end

require 'image'


function patch2vec_init(nn_file_path)
	 NN = torch.load(nn_file_path):cuda()
end

function create_tensor_from_image_storage(storage, H, W, num_channels)
	 -- local t = torch.ByteTensor(storage):float()/255
	 local t = torch.ByteTensor(storage):cuda():float()/255
	 t = t:view(num_channels, H, W)
	 t = t:transpose(2,3):transpose(1,2)
	 t = t:contiguous()
	 -- Adding 1 to fit to the NN input dimensions
	 t = t:view(1, num_channels, H, W)
	 return t
end

function compute_patches_distance_NN(patch1_storage_obj, patch2_storage_obj, H, W, num_channels)
	 local patch1 = create_tensor_from_image_storage(patch1_storage_obj, H, W, num_channels)
	 local patch2 = create_tensor_from_image_storage(patch2_storage_obj, H, W, num_channels)
	 local v1 = NN:forward(patch1):clone()
	 local v2 = NN:forward(patch2):clone()
	 -- Distance is 1-similarity
	 local distance = 1-v1*v2
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
