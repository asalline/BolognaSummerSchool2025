# This file contains a function for reconstructing sinogram data for the 2DeteCT dataset.
'''Modified from: https://github.com/mbkiss/2DeteCTcodes '''


import astra
import imageio
import warnings
from typing import Any

import glob # Used for browsing through the directories.
import os # Used for creating directories.
import shutil # Later used for copying files.
import time # Used for keeping processing time.

#This may be needed if Astra and pytorch cause a conflict
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

# Data directories.
# We enter here some intrinsic details of the dataset needed for our reconstruction scripts.
# Set the variable "base_data_dir_str" to the path where the dataset is stored on your own workstation.
base_data_dir_str = 'C:/Users/ahauptma18/Dropbox/Bologna_summerschool/2DeteCT_slices1-1000/'

# Set the variable "save_dir_str" to the path where you would like to store the
# reconstructions you compute.
save_dir_str = 'C:/Users/ahauptma18/Dropbox/Bologna_summerschool/2DeteCTcodes-main/recs/' 

# User defined settings.

# Select the ID(s) of the slice(s) you want to reconstruct.
#import random
slice_id = range(1,2)
# Adjust this to your liking.

# Select which modes you want to reconstruct.
mode = 2 #These are all modes available.

# Define whether you have a GPU for computations available and if you like specify which one to use.
use_GPU = True
device = 'CUDA'
#gpuIndex = 0 # Set the index of the GPU+ card used.
#astra.set_gpu_index(gpuIndex)

# Pre-processing parameters.
binning = 1 # Manual selection of detector pixel binning after acqusisition.
excludeLastPro = True # Exclude last projection angle which is often the same as the first one.
'''subsampling potentially critical if limited GPU memory available'''
angSubSamp = 10 # Define a sub-sampling factor in angular direction.
# (all reference reconstructions are computed with full angular resolution).
maxAng = 360 # Maximal angle in degrees - for reconstructions with limited angle (standard: 360).

# Correction profiles.
# The detector is slightly shifted with respect to the ASTRA geometry specified.
# Furthermore, the detector has been changed shortly before 20220531 (between slices 2830 and 2831).
# The full correction profiles can be found below.
corr_profiles = dict()
corr_profiles['20220407_RvL'] = {'det_tan': 24.19, 'src_ort': -5.67, 'axs_tan': -0.5244, 'det_roll': -0.015}
corr_profiles['20220531_RvL'] = {'det_tan': 24.4203, 'src_ort': -6.2281, 'axs_tan': -0.5010, 'det_roll': -0.262}
# This array contains the simplified horizontal correction shift for both geometries.
corr = np.array([1.00, 0.0]) # Found the optimal shifts to be
# [2.75, 1.00] for (2048,2048). subsampling yields 
# [1.00, 0.00] for (1024,1024). 

# File names in dataset structure.
sino_name = 'sinogram.tif'
dark_name = 'dark.tif'
flat_name = ('flat1.tif', 'flat2.tif')
slcs_name ="slice{:05}"

# Reference information.
sino_dims = (3601,1912) # Dimensions of the full sinograms.
detPix = 0.0748 # Physical size of one detector pixel in mm.
# Early OOD scans: 5521 - 5870 
# Late OOD scans: 5871 - 6370

#
downFactor=2 #downsampling of output size
# Reconstruction parameters.
recSz = (int(2048/downFactor),int(2048/downFactor)) # Used reconsttuction area to create as little model-inherent artifacts within the FOV.



# Keep track of the processing time per reconstruction job.
t = time.time();
print('Starting reconstruction job...', flush=True)



for i_slc in slice_id:

        i_mode=2
    # for i_mode in modes:

        # Load and pre-process data.
        print('SLICE ----------> ', str(i_slc))
        # Get the current path for respective slice and mode within the dataset structure.
        current_path = base_data_dir_str + slcs_name.format(i_slc) + '/mode{}/'.format(i_mode)

        # load flat-field and dark-fields.
        # There are two flat-field images (taken before and after the acquisition of ten slices),
        # we simply average them.
        dark = imageio.imread(glob.glob(current_path + dark_name)[0]) 
        flat1 = imageio.imread(glob.glob(current_path + flat_name[0])[0])
        flat2 = imageio.imread(glob.glob(current_path + flat_name[1])[0])
        flat = np.mean(np.array([ flat1, flat2 ]), axis=0 )

        # Read in the sinogram.
        sinogram = imageio.imread(glob.glob(current_path + sino_name)[0])
        sinogram =  np.ascontiguousarray(sinogram)
        
        # Change data type of the giles from uint16 to float32
        sinogram = sinogram.astype('float32')
        dark = dark.astype('float32')
        flat = flat.astype('float32')
        
        # Down-sample the sinogram as well as the dark and flat field
        # for i in np.arange(sino_dims[0]):
        sinogram = (sinogram[:,0::2]+sinogram[:,1::2])
        dark = (dark[0,0::2]+dark[0,1::2])
        flat = (flat[0,0::2]+flat[0,1::2])
            
        print('Shape of down-sampled sinogram:', sinogram.shape)
        print('Shape of down-sampled dark field:', dark.shape)
        print(dark[0],dark[-1],dark[-2])
        print('Shape of down-sampled flat field:', flat.shape)
        print(flat[0],flat[-1],flat[-2])

        # Subtract the dark field, devide by the flat field,
        # and take the negative log to linearize the data according to the Beer-Lambert law.
        data = sinogram - dark
        data = data/(flat-dark)

        # Exclude last projection if desired.
        if excludeLastPro:
            data = data[0:-1,:]

        # Create detector shift via linear grid interpolation.
        if i_slc in range(1,2830+1) or i_slc in range(5521,5870+1):
            detShift = corr[0] * detPix
        else:
            detShift = corr[1] * detPix

        detGrid = np.arange(0,956) * detPix
        detGridShifted = detGrid + detShift
        detShiftCorr = interp1d(detGrid, data, kind='linear', bounds_error=False, fill_value='extrapolate')
        data = detShiftCorr(detGridShifted)

        # Clip the data on the lower end to 1e-6 to avoid division by zero in next step.
        data = data.clip(1e-6, None)
        print("Values have been clipped from [", np.min(data), ",", np.max(data),"] to [1e-6,None]")

        
            
        # Take negative log.
        data = np.log(data)
        data = np.negative(data)
        data = np.ascontiguousarray(data)
       

        # Create array that stores the used projection angles.
        angles = np.linspace(0,2*np.pi, 3601) # 3601 = full width of sinograms.

        # Apply exclusion of last projection if desired.
        if excludeLastPro:
            angles = angles[0:-1]
            print('Excluded last projection.')

        # Apply angular subsampling.
        data = data[0::angSubSamp,:]
        angles = angles[0::angSubSamp]
        angInd = np.where(angles<=(maxAng/180*np.pi))
        angles = angles[angInd]
        data = data[:(angInd[-1][-1]+1),:]

        print('Data shape:', data.shape)
        print('Length angles:', len(angles))

        print('Loading and pre-processing done', flush=True)


        print('Computing reconstruction for slice', i_slc, '...', flush=True)

        # Create ASTRA objects for reconstruction.
        detSubSamp = 2
        binning = 1
        detPixSz = detSubSamp * binning * detPix
        SOD = 431.019989 
        SDD = 529.000488

        # Scale factor calculation.
        # ASTRA assumes that the voxel size is 1mm.
        # For this to be true we need to calculate a scale factor for the geometry.
        # This can be done by first calculating the 'true voxel size' via the intersect theorem
        # and then scaling the geometry accordingly.

        # Down-sampled width of the detector.
        nPix = 956
        det_width = detPixSz * nPix

        # Physical width of the field of view in the measurement plane via intersect theorem.
        FOV_width = det_width * SOD/SDD
        print('Physical width of FOV (in mm):', FOV_width)

        # True voxel size with a given number of voxels to be used.
        nVox = 1024
        voxSz = FOV_width / nVox
        print('True voxel size (in mm) for', nVox, 'voxels to be used:', voxSz)

        # Scaling the geometry accordingly.
        scaleFactor = 1./voxSz
        print('Self-calculated scale factor:', scaleFactor)
        SDD = SDD * scaleFactor
        SOD = SOD * scaleFactor
        detPixSz = detPixSz * scaleFactor

        # Create ASTRA objects.
        projGeo = astra.create_proj_geom('fanflat', detPixSz, 956, angles, SOD, SDD - SOD)
        volGeo = astra.create_vol_geom(recSz[0], recSz[1])
        recID = astra.data2d.create('-vol', volGeo)
        sinoID = astra.data2d.create('-sino', projGeo, data)
        projID   = astra.create_projector('cuda', projGeo, volGeo)
        

       
        # reconstruct
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ProjectorId'] = projID
        cfg['ProjectionDataId'] = sinoID
        cfg['ReconstructionDataId'] = recID
        
        # cfg['FilterType']='hann' #Filter type
        # cfg['FilterD']=0.6 #frequency parameter
        fbp_id = astra.algorithm.create(cfg)
        astra.algorithm.run(fbp_id)
        V = astra.data2d.get(recID)
        
        plt.gray()
        plt.imshow(V)
        plt.show()
        
        # garbage disposal
        astra.data2d.delete([sinoID, recID])
        astra.projector.delete(projID)
        astra.algorithm.delete(fbp_id)
        V = np.maximum(V,0)
        



        # Save reconstruction.
        imageio.imwrite(str(save_dir_str + 'slice' + str(i_slc).zfill(5) + '/' 'mode' + str(i_mode)+'/reconstruction.tif'),(V.astype(np.float32)).reshape(recSz))


