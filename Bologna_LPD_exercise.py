import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import astra


def random_ellipses(num_of_ellipses: int,
                    size: tuple) -> torch.Tensor:
    
    # Pre-allocate PyTorch tensor for the ellipses
    ellipses = torch.zeros(size=((num_of_ellipses, ) + (size[0], size[1])))

    # Create linear spaces for the ellipses to be put into.
    # Note that x is a row vector and y is a column vector.
    x = torch.linspace(start=-size[0], end=size[1], steps=size[1])
    y = torch.linspace(start=-size[0], end=size[1], steps=size[0])[:,None]

    # Loop over the amount of ellipses needed.
    for k in range(num_of_ellipses):
        # Find a random points for center coordinates, height and width.
        x0 = np.random.randint(low=-size[0], high=size[0])
        y0 = np.random.randint(low=-size[1], high=size[1])
        a = np.random.randint(low=-size[0], high=size[0])
        b = np.random.randint(low=-size[1], high=size[1])

        # Use ellipse function to create them.
        ellipses[k,:,:] = ((x-x0)/a)**2 + ((y-y0)/b)**2 <= 1

        # Force the ellipse value to be random constant from uniform distribution
        # for each ellipse.
        ellipses[k,:,:][ellipses[k,:,:] == 1] = torch.rand(1)

    # Sum all ellipses together and normalize the data to be in [0,1].
    ellipses = torch.sum(ellipses, dim=0)
    ellipses /= torch.max(ellipses)
    
    return ellipses

class LPD_transform:
    def __init__(self,
                proj_geom,
                vol_geom): 
    
        self.proj_geom = proj_geom
        self.vol_geom = vol_geom

    def forw(self, reconstruction):
        proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
        rec_id = astra.data2d.create('-vol', self.vol_geom)
        
        f_sinogram = astra.create_sino(reconstruction, proj_id)[1]
        # sinogram_id = astra.data2d.create('-sino', self.proj_geom, f_sinogram-sinogram[0,0,:,:].cpu().detach().numpy())
        # print('FORW2')

        return f_sinogram
    
    def back(self, sinogram):
        proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
        rec_id = astra.data2d.create('-vol', self.vol_geom)
        FBP = astra.astra_dict('FBP_CUDA')
        sinogram_id = astra.data2d.create('-sino', self.proj_geom, sinogram)
        # Create FBP reconstruction.
        cfg = FBP
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['option'] = { 'FilterType': 'none' }

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        rec = astra.data2d.get(rec_id)
        # rec = torch.as_tensor(np.maximum(0, rec))

        astra.data2d.delete(sinogram_id)
        astra.projector.delete(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)

        # print('BACK2')
       
        return rec


# Function for ASTRA operators to be backpropagated. Do not change.
# based on
# https://github.com/odlgroup/odl/blob/25ec783954a85c2294ad5b76414f8c7c3cd2785d/odl/contrib/torch/operator.py#L33
class NumpyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, forward_fun, backward_fun):
        ctx.forward_fun = forward_fun
        ctx.backward_fun = backward_fun

        x_np = x.detach().cpu().numpy()
        # y_np = np.stack([ctx.forward_fun(x_np_i) for x_np_i in x_np])
        y_np= ctx.forward_fun(x_np)
        # y_np = y_np.astype(np.float32)
        y = torch.from_numpy(y_np).to(x.device)
        ctx.save_for_backward(y)
        # print('FORW1')
        return y

    @staticmethod
    def backward(ctx, y):
        y_np = y.detach().cpu().numpy()
        # x_np = np.stack([ctx.backward_fun(y_np_i) for y_np_i in y_np])
        x_np = ctx.backward_fun(y_np)
        # x_np = x_np.astype(np.float32)
        x = torch.from_numpy(x_np).to(y.device)
        # print('BACK1')
        return x, None, None
    
    

class LPDIterModule(torch.nn.Module):
    def __init__(self, circ_PAT_Op, adjoint=False):
        super().__init__()
        self.circPAT = circ_PAT_Op
        self.adjoint = adjoint

    def forward(self, x):
        # print('HERE')
        # x.shape: (N, C, H, W) or (N, C, D, H, W)
        forward_fun = (self.circPAT.back if self.adjoint else
                        self.circPAT.forw)
        # note: backward_fun is only an approximation to the transposed jacobian
        backward_fun = (self.circPAT.forw if self.adjoint else
                        self.circPAT.back)
       
        xi=x[0,0]
       
        y = NumpyFunction.apply(xi, forward_fun, backward_fun)
        y=y[None,None,:]
        return y
    

class LPDStep(nn.Module):
    def __init__(self,
                 primal_in_channels: int,
                 dual_in_channels: int,
                 out_channels: int,
                 n_iter: int,
                 device='cuda') -> torch.Tensor:
        super().__init__()

        self.primal_in_channels = primal_in_channels
        self.dual_in_channels = dual_in_channels
        self.out_channels = out_channels
        self.n_iter = n_iter
        self.device = device
        self.relu = nn.ReLU()

        # Define learnable step-size parameter initialized as zeros.
        self.step_length = nn.Parameter(torch.zeros(1,1,1,1))
        
        self.primal_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.primal_in_channels, 
                      out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, 
                      out_channels=self.out_channels, kernel_size=(3,3), padding=1),
        )

        ### TASK: Define dual network for the Learned Primal-Dual algorithm




        self.to(device)

    def forward(self,
                reconstruction: torch.Tensor,
                sinogram: torch.Tensor,
                h: torch.Tensor,
                forw_op,
                back_op) -> torch.Tensor:

        ### TASK: Implement forward operator for Learned Primal-Dual





class LPD(nn.Module):
    def __init__(self,
                 primal_in_channels: int,
                 dual_in_channels: int,
                 out_channels: int,
                 n_iter: int,
                 forw,
                 back,
                 device='cuda') -> torch.Tensor:
        super().__init__()
        
        self.primal_in_channels = primal_in_channels
        self.dual_in_channels = dual_in_channels
        self.out_channels = out_channels
        self.n_iter = n_iter
        self.forw = forw
        self.back = back
        self.device = device
        self.relu = nn.ReLU()

        for k in range(self.n_iter):
            step = LPDStep(primal_in_channels=self.primal_in_channels,
                           dual_in_channels=self.dual_in_channels,
                           out_channels=self.out_channels,
                           n_iter=self.n_iter,
                           device=self.device)
            setattr(self, f'step{k}', step)

    def forward(self,
                reconstruction: torch.Tensor,
                sinogram: torch.Tensor,
                vol_geom,
                proj_geom) -> torch.Tensor:
        
        h = torch.zeros(sinogram.shape).to(self.device)

        for k in range(self.n_iter):
            step = getattr(self, f'step{k}')
            reconstruction, h = step(reconstruction, sinogram, h, self.forw, self.back)

        return self.relu(reconstruction)
    




device = 'cuda' if torch.cuda.is_available() else 'cpu'



### TASK: Generate validation data set from the "random_ellipses" function and save them.
### Consider using image size of (128,128), amount of validation images of 20, and
### choose amount of ellipses in each image to be more than 20.






### TASK: With ASTRA/other package, turn ellipse data to sinograms and add Gaussian noise (2% for example) to those.
### Create for exmple sparse setting with 500 beams and 360° rotation with 90 angles.
### Reconstruct the images from NOISY sinogram data, and use noisy sinograms and noisy reconstructions
### as an input of the network.
### If using ASTRA, remember to delete your unused variables to not allocate your memory.






### TASK: Call your network with needed parameters and get the parameters of the network to be used later.

# Define loss functions.



# Initialize the network.


# Get the parameters.



### TASK: Build your training loop with your favorite optimizer, and consider using scheduler.
### The function below can be used as a base, or can be changed.

def training_loop(net: nn.Module,
                  n_iter: int,
                  learning_rate: float):
    


# Call of the training loop.
net = training_loop(lpd, n_iter=10001, learning_rate=1e-3)



### TASK: Remember to evaluate the networks and see how those work.


