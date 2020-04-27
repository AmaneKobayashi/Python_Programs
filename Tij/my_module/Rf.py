def Rf(dens,diff):
	import numpy as np
	import tifffile

	np_diff_amp=np.where(diff<0.0, 0.0, diff)
	np_diff_amp=np.sqrt(np_diff_amp)
	np_diff_amp_abs=np.absolute(np_diff_amp)

	R_structure_factor = np.fft.fftn(dens,norm="ortho")
	R_amp = np.absolute(R_structure_factor)
	R_amp = np.fft.fftshift(R_amp)
	R_amp=np.fliplr(R_amp)
	R_amp_square=np.square(R_amp)
	R_amp_abs=np.absolute(R_amp)
	
#	tifffile.imsave("R_amp.tif" ,R_amp)
#	tifffile.imsave("np_diff_amp.tif" ,np_diff_amp)

	R_amp_abs_np_diff_amp_abs=R_amp_abs*np_diff_amp_abs

	R_amp_square_pos=np.zeros(dens.shape,dtype="float32")
	R_amp_abs_pos=np.zeros(dens.shape,dtype="float32")
	R_amp_abs_np_diff_amp_abs_pos=np.zeros(dens.shape,dtype="float32")
	np_diff_amp_pos=np.zeros(dens.shape,dtype="float32")
	np_diff_amp_abs_pos=np.zeros(dens.shape,dtype="float32")
	
	R_amp_square_pos[np_diff_amp%2>0.0]=R_amp_square[np_diff_amp%2>0.0]
	R_amp_abs_pos[np_diff_amp%2>0.0]=R_amp_abs[np_diff_amp%2>0.0]
	R_amp_abs_np_diff_amp_abs_pos[np_diff_amp%2>0.0]=R_amp_abs_np_diff_amp_abs[np_diff_amp%2>0.0]
	np_diff_amp_pos[np_diff_amp%2>0.0]=np_diff_amp[np_diff_amp%2>0.0]
	np_diff_amp_abs_pos[np_diff_amp%2>0.0]=np_diff_amp_abs[np_diff_amp%2>0.0]

	amp2_sum=np.sum(R_amp_square_pos)
	amp_x_diff_amp_sum=np.sum(R_amp_abs_np_diff_amp_abs_pos)
	diff_amp_sum=np.sum(np_diff_amp_abs_pos)
	scale_factor=amp_x_diff_amp_sum/amp2_sum

	diff_amp_scale=np.sum(np.absolute(np_diff_amp_abs_pos-scale_factor*R_amp_abs_pos))
	R_factor=diff_amp_scale/diff_amp_sum
	
	return R_factor,scale_factor
