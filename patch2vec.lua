require 'torch'
require 'nn'
require 'dpnn'


function patch2vec_init(nn_file_path)
	 NN = torch.load(nn_file_path)
end

function create_tensor_from_image_storage(storage, H, W, num_channels)
	 local t = torch.ByteTensor(storage):float()/255
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
