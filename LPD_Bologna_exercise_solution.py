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
        self.dual_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.dual_in_channels, 
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
        self.to(device)

    def forward(self,
                reconstruction: torch.Tensor,
                sinogram: torch.Tensor,
                h: torch.Tensor,
                forw_op,
                back_op) -> torch.Tensor:

        ### TASK: Implement forward operator for Learned Primal-Dual
        f_sinogram = forw_op(reconstruction)

        u = torch.cat([h, f_sinogram, sinogram], dim=1)
        
        h = h + self.dual_layers(u)

        rec = back_op(h)

        u = torch.cat([reconstruction.to(self.device), rec.to(self.device)], dim=1)
        
        u = self.primal_layers.to(self.device)(u)

        reconstruction = reconstruction + u
    
        return reconstruction, h

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
# amount_of_images = 20
num_of_ellipses = 50
# image_size = (128,128)
# validation_ellipses = torch.zeros(size=(amount_of_images, ) + image_size)
# for k in range(amount_of_images):
#     validation_ellipses[k,:,:] = random_ellipses(num_of_ellipses=num_of_ellipses, size=image_size)

test_images = torch.load('test_images.pt')
image_size = (test_images.shape[1], test_images.shape[2])
amount_of_images = test_images.shape[0]
print(amount_of_images)
print(image_size)
# Create a volume geometry of a size of variable "size".
vol_geom = astra.create_vol_geom(image_size)

### TASK: With ASTRA/other package, turn ellipse data to sinograms and add Gaussian noise to those.
### with 500 beams and 360° rotation with 90 angles. (This is sparse setting).

vol_geom = astra.create_vol_geom(image_size)
# Define variables for the sparse angle Fan-Beam projection.
num_of_lines = 512
start_angle = 0
end_angle = 2*np.pi
amount_of_angles = 90
angles = np.linspace(start_angle, end_angle, amount_of_angles)
SOD = 250
SDD = 260

# Create the Fan-Beam projction.
proj_geom = astra.create_proj_geom('fanflat', 1.0, num_of_lines, angles, SOD, SDD-SOD)

# Create a sinogram using the CPU or GPU.
proj_id_sparse = astra.create_projector('cuda', proj_geom, vol_geom)
validation_sinograms = torch.zeros(size=(amount_of_angles, ) + (amount_of_angles, num_of_lines))
for k in range(amount_of_images):
    validation_sinograms[k,:,:] = torch.as_tensor(astra.create_sino(test_images[k,:,:].numpy(), proj_id_sparse)[1])

# sparse_sinogram_id, sparse_sinogram = astra.create_sino(validation_ellipses, proj_id_sparse)

# Add x% of mean zero noise on every test image.
percentage = 0.02
mean = 0

test_noisy_sinograms = torch.zeros(size=(amount_of_images, ) + (amount_of_angles, num_of_lines))
test_noisy_sinogram_ids = []
test_recos = torch.zeros(size=(amount_of_images, ) + (image_size))
for k in range(amount_of_images):
    proj_id = astra.create_projector('cuda',proj_geom, vol_geom)

    test_sinogram = astra.create_sino(test_images[k,:,:].numpy(), proj_id)[1]

    test_sinogram = torch.as_tensor(astra.create_sino(test_images[k,:,:].numpy(), proj_id)[1])
    noise = torch.normal(mean=mean, std=torch.max(test_sinogram), size=test_sinogram.shape) * percentage
    test_noisy_sinogram_ids.append(astra.data2d.create('-sino', proj_geom, (test_sinogram + noise).numpy()))
    test_noisy_sinograms[k,:,:] = test_sinogram + noise
    # Define reconstruction id.
    rec_id = astra.data2d.create('-vol', vol_geom)

    # Create Filtered Back-Projection operator.
    FBP = astra.astra_dict('FBP_CUDA')

    # Create FBP reconstruction.
    cfg = FBP
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = test_noisy_sinogram_ids[k]
    cfg['option'] = { 'FilterType': 'Hann' }

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    test_recos[k,:,:] = torch.as_tensor(astra.data2d.get(rec_id))
    # test_recos[k,:,:] = np.maximum(0, test_recos[k,:,:])

    # Free memory, THIS IS EXTREMELY NECESSARY WITH ASTRA!
    astra.data2d.delete(test_noisy_sinogram_ids[k])
    astra.projector.delete(proj_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)

### TASK: Call your network with needed parameters and get the parameters of the network to be used later.

# Define loss functions.
train_loss = nn.MSELoss()
test_loss = nn.MSELoss()

proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
rec_id = astra.data2d.create('-vol', vol_geom)

LPD_operators = LPD_transform(proj_geom, vol_geom)

forw_op = LPDIterModule(LPD_operators, adjoint=False)
back_op = LPDIterModule(LPD_operators, adjoint=True)

# Initialize the network.
lpd = LPD(primal_in_channels=2,
          dual_in_channels=3,
          out_channels=1,
          n_iter=5,
          forw=forw_op,
          back=back_op,
          device=device)

lpd.load_state_dict(torch.load('/homedir01/asalline/Documents/BolognaSummerSchool/Saved networks/sparse_LPD.pth'))

lpd_parameters = list(lpd.parameters())

### TASK: Build your training loop with your favorite optimizer, and consider using scheduler.

def training_loop(net: nn.Module,
                  n_iter: int,
                  learning_rate: float):
    
    optimizer = torch.optim.Adam(lpd_parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)

    for k in range(n_iter):
        train_image = random_ellipses(num_of_ellipses=num_of_ellipses, size=image_size)

        proj_id = astra.create_projector('cuda',proj_geom, vol_geom)

        train_sinogram = astra.create_sino(train_image.numpy(), proj_id)[1]
        train_sinogram = torch.as_tensor(astra.create_sino(train_image.numpy(), proj_id)[1])
        noise = torch.normal(mean=mean, std=torch.max(train_sinogram), size=train_sinogram.shape) * percentage
        train_noisy_sinogram_id = astra.data2d.create('-sino', proj_geom, (train_sinogram + noise).numpy())
        train_noisy_sinogram = train_sinogram + noise
        # Define reconstruction id.
        rec_id = astra.data2d.create('-vol', vol_geom)

        # Create Filtered Back-Projection operator.
        FBP = astra.astra_dict('FBP_CUDA')

        # Create FBP reconstruction.
        cfg = FBP
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = train_noisy_sinogram_id
        cfg['option'] = { 'FilterType': 'Hann' }

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # test_recos[k,:,:] = torch.as_tensor(np.maximum(0, astra.data2d.get(rec_id)))
        train_reco = torch.as_tensor(astra.data2d.get(rec_id))
        # test_recos[k,:,:] = np.maximum(0, test_recos[k,:,:])

        # Free memory, THIS IS EXTREMELY NECESSARY WITH ASTRA!
        astra.data2d.delete(train_noisy_sinogram_id)
        astra.projector.delete(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)

        train_image = train_image[None,None,:,:].to(device)
        train_reco = train_reco[None,None,:,:].to(device)
        train_noisy_sinogram = train_noisy_sinogram[None,None,:,:].to(device)

        net.train()

        optimizer.zero_grad()

        outputs = net(train_reco, train_noisy_sinogram, vol_geom, proj_geom)

        loss = train_loss(outputs, train_image)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(lpd_parameters, max_norm=1.0, norm_type=2)

        optimizer.step()
        scheduler.step()

        if k % 5000 == 0:
            test_outputs = torch.zeros(size=(test_recos.shape[0], ) + (image_size))
            for i in range(test_images.shape[0]):
                test_outputs[i] = net(test_recos[i][None,None,:,:].to(device), 
                                      test_noisy_sinograms[i][None,None,:,:].to(device), 
                                      vol_geom, 
                                      proj_geom)
            testing_loss = test_loss(test_outputs.to(device), test_images.to(device)).item()

            # writer.add_scalar('Test Loss', testing_loss, k)
            # writer.add_scalar('PSNR', psnr(torch.max(test_images).item(), testing_loss), k)
            # writer.add_image('Testing ouput', test_outputs[0,:,:], k, dataformats='HW')
            # writer.add_image('Testing Ground truth', test_images[0,:,:], k, dataformats='HW')
            # writer.flush()
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(test_outputs[4,:,:].cpu().detach().numpy())
            plt.subplot(1,2,2)
            plt.imshow(test_images[4,:,:].cpu().detach().numpy())
            plt.show()

            # torch.save(net.state_dict(), '/homedir01/asalline/Documents/BolognaSummerSchool/Saved networks/' + 'sparse_LPD.pth')

    return net


net = training_loop(lpd, n_iter=10001, learning_rate=1e-3)


