import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
class ContinuousPiecewiseLinear(nn.Module):
    def __init__(self, t_win, num_pieces, device='cpu'):
        super(ContinuousPiecewiseLinear, self).__init__()
        self.num_pieces = num_pieces
        self.t_win = t_win
        self.t = torch.linspace(0, t_win, t_win).unsqueeze(0).to(device)
        self.device = device
        # Learnable parameters
        #self.slopes = slopes #nn.Parameter(torch.randn(num_pieces, 2, device=device) * 0.1)  # Small values
        self.base_intercept = torch.zeros(1, 2, device=device) # Start at 0
        self.segment_widths = torch.full((num_pieces,), t_win / num_pieces, device=device)  # Equal widths
        # we should start by dxy_pw
        # Fixed segment boundaries
        self.segment_boundaries = torch.linspace(0, t_win, num_pieces + 1, device=device).detach()

    def forward(self, slopes):
        t =  self.t
        #batch_size, t_steps = t.shape
        t = t.unsqueeze(-1)
        self.slopes = slopes   # sloops are velocities
        # Segment masks
        segment_masks = ((t >= self.segment_boundaries[:-1]) & (t < self.segment_boundaries[1:]))
        segment_masks = segment_masks.float()
        segment_masks[:,:,-1][0][-1] = torch.tensor(1).float().to(self.device)


        # Time relative to the start of each segment
        t_rel = t - self.segment_boundaries[:-1]
        t_rel = t_rel * segment_masks

        # Compute intercepts in a vectorized manner
        widths = self.segment_widths.unsqueeze(-1)
        slopes_scaled = self.slopes * widths
        intercepts = torch.cumsum(torch.cat([self.base_intercept, slopes_scaled[:-1]], dim=0), dim=0)

        # Compute piecewise values
        slopes = self.slopes.view(1, 1, self.num_pieces, 2)
        intercepts = intercepts.view(1, 1, self.num_pieces, 2)
        piecewise_values = t_rel.unsqueeze(-1) * slopes + intercepts

        # Combine segments
        trajectory = (segment_masks.unsqueeze(-1) * piecewise_values).sum(dim=2)

        return trajectory

    def regularization_loss(self, lam=0* 1e-3, mu=0.0, beta=0.0):
        # Smoothness regularization
        smoothness_loss = lam * torch.sum((self.slopes[1:] - self.slopes[:-1]) ** 2)

        # Width regularization
        #width_reg = mu * torch.sum((self.segment_widths[1:] - self.segment_widths[:-1]) ** 2)

        # L2 regularization
        l2_loss = beta * (torch.sum(self.slopes ** 2) + torch.sum(self.base_intercept ** 2))

        return smoothness_loss + l2_loss #+ width_reg




class ContinuousPiecewiseLinear_Dxy_pw(nn.Module):
    def __init__(self, t_win, num_pieces, device='cuda'):
        super(ContinuousPiecewiseLinear_Dxy_pw, self).__init__()
        self.num_pieces = num_pieces
        self.t_win = t_win
        t = torch.linspace(0, t_win, t_win + 1).unsqueeze(0).to(device)
        self.device = device
        # Learnable parameters
        #self.slopes = slopes #nn.Parameter(torch.randn(num_pieces, 2, device=device) * 0.1)  # Small values
        self.base_intercept = torch.zeros(1, 2, device=device) # Start at 0
        self.segment_widths = torch.full((num_pieces,), t_win / num_pieces, device=device)  # Equal widths
        # we should start by dxy_pw
        # Fixed segment boundaries
        self.segment_boundaries = torch.linspace(0, t_win, num_pieces + 1, device=device).detach()
        # Segment masks
        self.t = t.unsqueeze(-1)
        segment_masks = ((self.t >= self.segment_boundaries[:-1]) & (self.t < self.segment_boundaries[1:]))
        segment_masks[:, :, -1][0][-1] = torch.tensor(1).float().to(self.device)
        self.segment_masks = segment_masks.float()

    def forward(self, Dxy_pw):
        #batch_size, t_steps = t.shape
        # Time relative to the start of each segment
        t_rel = self.t - self.segment_boundaries[:-1]
        t_rel = t_rel * self.segment_masks

        # Compute intercepts in a vectorized manner

        widths = self.segment_widths.unsqueeze(-1)
        self.Dxy_pc = Dxy_pw

        slopes_scaled = Dxy_pw #self.slopes * widths
        self.slopes = Dxy_pw / widths  # sloops are velocities

        intercepts = torch.cumsum(torch.cat([self.base_intercept, slopes_scaled[:-1]], dim=0), dim=0)
        #v_dense = (self.segment_masks.unsqueeze(-1) * self.slopes).sum(dim=2)
        # Compute piecewise values
        slopes = self.slopes.view(1, 1, self.num_pieces, 2)
        intercepts = intercepts.view(1, 1, self.num_pieces, 2)
        piecewise_values = t_rel.unsqueeze(-1) * slopes + intercepts

        # Combine segments
        trajectory = (self.segment_masks.unsqueeze(-1) * piecewise_values).sum(dim=2)

        return trajectory

    def regularization_loss(self, lam=1e-1, mu=0.0, beta=0.0):
        # Smoothness regularization
        smoothness_loss = lam * torch.sum((self.Dxy_pc[1:] - self.Dxy_pc[:-1]) ** 2)
        l2_loss = beta * (torch.sum(self.Dxy_pc ** 2))


        # Width regularization
        #width_reg = mu * torch.sum((self.segment_widths[1:] - self.segment_widths[:-1]) ** 2)

        # L2 regularization
        #l2_loss = beta * (torch.sum(self.slopes ** 2) + torch.sum(self.base_intercept ** 2))

        return smoothness_loss + l2_loss #+ width_reg

def integrate_piecewise_displacement(Dxy_pw):
    """
    输入: Dxy_pw.shape = (N, 2)，每段的 Δx, Δy
    输出: positions.shape = (N+1, 2)，积分后的位置序列
    """
    # 累积求和并加上起点 (0,0)
    start = torch.zeros(1, 2, device=Dxy_pw.device, dtype=Dxy_pw.dtype)
    positions = torch.cat([start, torch.cumsum(Dxy_pw, dim=0)], dim=0)
    return positions

if __name__ == '__main__':
    # Initialize model
    t_win = 1
    num_pieces = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ContinuousPiecewiseLinear(t_win, num_pieces, device=device)

    # Generate free-fall trajectory: y = x^2
    real_pieces = 100
    t = torch.linspace(0, t_win, real_pieces).to(device)  # Shape: (1, 100)
    true_trajectory = torch.cat([10*torch.cos(t.unsqueeze(1) )-10 +2 * t.unsqueeze(1) , (10* torch.sin(t.unsqueeze(1))+ 2 * t.unsqueeze(1))], dim=-1).squeeze().T  # Shape: (2, 100)

    # Forward pass through the model
    reconstructed_trajectory = model(true_trajectory)

    # Compute regularization loss

    lr = 1e-1
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # Training loop
    num_epochs = 1500
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        reconstructed_trajectory = model(t).squeeze()

        # Compute loss
        loss = criterion(reconstructed_trajectory, true_trajectory).float() + model.regularization_loss()#+ model.lam * torch.sum((model.slopes[1:] - model.slopes[:-1]) ** 2)

        # Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
    # Plot the results
    plt.figure(figsize=(8, 6))
    true_trajectory_np = true_trajectory.squeeze().cpu().numpy()
    reconstructed_trajectory_np = reconstructed_trajectory.squeeze().cpu().detach().numpy()
    reg_loss = model.regularization_loss()
    print(f"Regularization Loss: {reg_loss.item()}")
    plt.plot(true_trajectory_np[:, 0], true_trajectory_np[:, 1], c= 'b', label="True Trajectory (y = x^2)")
    plt.plot(reconstructed_trajectory_np[:, 0], reconstructed_trajectory_np[:, 1], c = 'orange', label="Reconstructed Trajectory")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Piecewise Linear Reconstruction of Free-Fall Curve')
    plt.legend()
    plt.grid()
    plt.show()


