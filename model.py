# Kolmogorov-Arnold Networks
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for CUDA availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Calculate the grid step size
        grid_step = 2 / grid_size

        # Create the grid tensor
        grid_range = torch.arange(-spline_order, grid_size + spline_order + 1)
        grid_values = grid_range * grid_step - 1
        self.grid = grid_values.expand(in_features, -1).contiguous()

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.base_activation = nn.SiLU()

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the base weight tensor with Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        with torch.no_grad():
            # Generate random noise for initializing the spline weights
            noise_shape = (self.grid_size + 1, self.in_features, self.out_features)
            random_noise = (torch.rand(noise_shape) - 0.5) * 0.1 / self.grid_size

            # Compute the spline weight coefficients from the random noise
            grid_points = self.grid.T[self.spline_order : -self.spline_order]
            spline_coefficients = self.curve2coeff(grid_points, random_noise)

            # Copy the computed coefficients to the spline weight tensor
            self.spline_weight.data.copy_(spline_coefficients)

        # Initialize the spline scaler tensor with Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """

        # Expand the grid tensor to match the input tensor's dimensions
        expanded_grid = self.grid.unsqueeze(0).expand(
            x.size(0), *self.grid.size()
        )  # (batch_size, in_features, grid_size + 2 * spline_order + 1)

        # Add an extra dimension to the input tensor for broadcasting
        input_tensor_expanded = x.unsqueeze(-1)  # (batch_size, in_features, 1)

        # Convert tensor into the current device type
        expanded_grid = expanded_grid.to(device)
        input_tensor_expanded = input_tensor_expanded.to(device)

        # Initialize the bases tensor with boolean values
        bases = (
            (input_tensor_expanded >= expanded_grid[:, :, :-1])
            & (input_tensor_expanded < expanded_grid[:, :, 1:])
        ).to(
            x.dtype
        )  # (batch_size, in_features, grid_size + spline_order)

        # Compute the B-spline bases recursively
        for order in range(1, self.spline_order + 1):
            left_term = (
                (input_tensor_expanded - expanded_grid[:, :, : -order - 1])
                / (expanded_grid[:, :, order:-1] - expanded_grid[:, :, : -order - 1])
            ) * bases[:, :, :-1]

            right_term = (
                (expanded_grid[:, :, order + 1 :] - input_tensor_expanded)
                / (expanded_grid[:, :, order + 1 :] - expanded_grid[:, :, 1:-order])
            ) * bases[:, :, 1:]

            bases = left_term + right_term

        return bases.contiguous()

    def curve2coeff(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        # Compute the B-spline bases for the input tensor
        b_splines_bases = self.b_splines(
            input_tensor
        )  # (batch_size, input_dim, grid_size + spline_order)

        # Transpose the B-spline bases and output tensor for matrix multiplication
        transposed_bases = b_splines_bases.transpose(
            0, 1
        )  # (input_dim, batch_size, grid_size + spline_order)
        transposed_output = output_tensor.transpose(
            0, 1
        )  # (input_dim, batch_size, output_dim)

        # Convert tensor into the current device type
        transposed_bases = transposed_bases.to(device)
        transposed_output = transposed_output.to(device)

        # Solve the least-squares problem to find the coefficients
        coefficients_solution = torch.linalg.lstsq(
            transposed_bases, transposed_output
        ).solution
        # (input_dim, grid_size + spline_order, output_dim)

        # Permute the coefficients to match the expected shape
        coefficients = coefficients_solution.permute(
            2, 0, 1
        )  # (output_dim, input_dim, grid_size + spline_order)

        return coefficients.contiguous()

    def forward(self, x: torch.Tensor):

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output


class KAN(nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        layers_hidden,
        grid_size=5,
        spline_order=3,
    ):
        super().__init__()

        self.input_layer = KANLinear(
            input_features, layers_hidden[0], grid_size, spline_order
        )
        self.hidden_layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.hidden_layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size,
                    spline_order,
                )
            )
        self.output_layer = KANLinear(
            layers_hidden[-1], output_features, grid_size, spline_order
        )

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        return x
