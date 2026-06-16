import os

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.special import gamma

from bumblebee.ai.rnn import CursorRNN as RNN


class Predictor:
    def __init__(self, model_name: str = "bumblebee-c-v1.pth"):
        """
        Initialize the prediction model with a pre-trained RNN for generating intermediate path steps.

        Parameters:
            model_name (str): The filename of the pre-trained model to load. Defaults to "bumblebee-c-v1.pth".

        Attributes:
            MAX_COORDINATE (int): Maximum coordinate value supported (e.g., 4096 for 4K resolution).
            MIN_COORDINATE (int): Minimum coordinate value.
            INPUT_DIM (int): Number of input features (start and destination x, y coordinates).
            HIDDEN_DIM (int): Size of the hidden state in the RNN.
            STEPS (int): Number of intermediate steps predicted between the start and destination.
            OUTPUT_DIM (int): Number of output features (2 coordinates for each intermediate step).
            LSTM_LAYERS (int): The number of LSTM layers in the RNN.
            device (torch.device): The device (CUDA, MPS, or CPU) on which the model computations are performed.
            model (RNN): The RNN model instance used for predicting paths.
            model_path (str): The full file path to the pre-trained model file.

        The constructor loads the pre-trained model, maps it to the appropriate computation device,
        and sets the model to evaluation mode.
        """
        # Constants for normalization
        self.MAX_COORDINATE = 4096  # allows to use 4k display,taken from training notebook: `rnn-train.ipynb`
        self.MIN_COORDINATE = 0

        # Constant for RNN model
        self.INPUT_DIM = 4  # Start (x, y) and Destination (x, y) coordinates
        self.HIDDEN_DIM = 512
        self.STEPS = 39  # Number of intemediate steps between start and destination
        self.OUTPUT_DIM = self.STEPS * 2  # 2 coordinates (x, y) for each steps
        self.LSTM_LAYERS = 2

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )  # Check for best hardware to run the model on

        self.model = RNN(
            self.INPUT_DIM,
            self.HIDDEN_DIM,
            self.OUTPUT_DIM,
            num_layers=self.LSTM_LAYERS,
        ).to(self.device)

        self.model_path = os.path.join(os.path.dirname(__file__), "models", model_name)
        if self.device == "cuda":
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=torch.device("cpu"))
            )
        self.model.eval()

    def __normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates to [0,1] range for model input
        """
        return (data - self.MIN_COORDINATE) / (
            self.MAX_COORDINATE - self.MIN_COORDINATE
        )

    def __denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize coordinates from [0,1] range to original range
        """
        return data * (self.MAX_COORDINATE - self.MIN_COORDINATE) + self.MIN_COORDINATE

    def _calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two points in 2D space.
        Parameters:
            point1 (np.ndarray): The first point as [x1, y1].
            point2 (np.ndarray): The second point as [x2, y2].
        Returns:
            float: The Euclidean distance between point1 and point2.
        """
        return float(np.linalg.norm(point1 - point2))

    def __calculate_angle(
        self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray
    ) -> float:
        """
        Calculate the angle between three points in 2D space.
        Parameters:
            point1 (np.ndarray): The first point as [x1, y1].
            point2 (np.ndarray): The second point as [x2, y2].
            point3 (np.ndarray): The third point as [x3, y3].
        Returns:
            float: The angle in radians between the line segments formed by point1-point2 and point2-point3.
        """
        vector1 = point2 - point1
        vector2 = point3 - point2

        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = np.clip(
            cos_angle, -1.0, 1.0
        )  # Ensure the value is within the valid range for arccos
        angle = np.arccos(cos_angle)
        return angle

    def __clean_path(
        self, path: np.ndarray, min_distance=10.0, deviation_threshold=50.0
    ) -> np.ndarray:
        """
        Clean the predicted path by removing points that are too close to each other.
        Parameters:
            path (np.ndarray): The predicted path as an array of shape (STEPS, 2).
            min_distance (float): The minimum distance between consecutive points to keep them.
            devaition_threshold (float): The threshold for deviation to consider a point as part of the path.
        Returns:
            np.ndarray: The cleaned path with points that are sufficiently spaced apart.
        """
        destination = path[-1]
        start = path[0]

        cleaned_path = [start]  # Start with the first point

        for i in range(1, len(path) - 1):
            if self._calculate_distance(cleaned_path[-1], path[i]) >= min_distance:
                cleaned_path.append(path[i])
        cleaned_path.append(destination)  # Always add the last point

        undeviated_path = np.array([], dtype=np.float32)
        cleaned_path = np.array(cleaned_path)
        if len(cleaned_path) <= 2:
            return cleaned_path
        previous_point_distance = float("inf")
        for i in range(len(cleaned_path)):
            distance_to_dest = self._calculate_distance(cleaned_path[i], destination)
            if distance_to_dest > previous_point_distance:
                if distance_to_dest <= deviation_threshold:
                    undeviated_path = np.append(undeviated_path, cleaned_path[i])
                    previous_point_distance = distance_to_dest
            else:
                undeviated_path = np.append(undeviated_path, cleaned_path[i])
                previous_point_distance = distance_to_dest

        undeviated_path = undeviated_path.reshape(-1, 2)

        return undeviated_path

    def __calculate_speed_factor(self, progress: float) -> float:
        """
        Calculate the speed factor based on the progress of the path; uses gamma distribution.
        Parameters:
            progress (float): The progress of the path as a value between 0 and 1.
        Returns:
            float: The speed factor, which increases gradually and tapers off.
        """
        # Parameters for gamma distribution
        k = 1.8  # Shape parameter
        theta = 1.0  # Scale parameter

        # Precompute normalization factor over the scaled range [0, 10]
        x = np.linspace(0, 10, 1000)  # Scaled progress range
        y = (x ** (k - 1) * np.exp(-x / theta)) / (theta**k * gamma(k))
        max_val = np.max(y)  # Maximum value for normalization

        # Scale progress to fit gamma's natural range
        scaled_progress = progress * 10

        # Base gamma distribution
        speed_factor = (
            scaled_progress ** (k - 1) * np.exp(-scaled_progress / theta)
        ) / (theta**k * gamma(k))
        speed_factor = speed_factor / max_val  # Normalize

        # Define flat region and transitions
        flat_value = 1.0  # Base value for flat region
        smooth_width = 0.15  # Transition width

        # Precompute gamma curve values at key points for transitions
        base_y = y / max_val  # Normalized base curve
        x_0_05 = 0.05  # Start of lower transition
        y_0_05 = base_y[np.searchsorted(x / 5, x_0_05)]  # Value at x = 0.05
        y_1_0 = base_y[-1]  # Value at x = 1.0

        # Apply flat region and transitions
        if 0.20 <= progress <= 0.90:  # Flat region with noise
            noise = np.random.uniform(-0.05, 0.0)  # Noise between -0.05 and 0
            speed_factor = flat_value + noise

        elif (0.20 - smooth_width) < progress < 0.20:  # Lower transition (0.05 to 0.20)
            speed_factor = np.interp(
                progress, [0.20 - smooth_width, 0.20], [y_0_05, flat_value]
            )

        elif 0.90 < progress < (0.90 + smooth_width):  # Upper transition (0.90 to 1.0)
            speed_factor = np.interp(progress, [0.90, 1.0], [flat_value, y_1_0])

        return speed_factor

    def __smooth_path(self, path: np.ndarray, noise_factor=1.5) -> np.ndarray:
        """
        Smooth the predicted path by applying Gaussian noise to the coordinates.
        Parameters:
            path (np.ndarray): The predicted path as an array of shape (STEPS, 2).
            noise_factor (float): The standard deviation of the Gaussian noise to be added.
        Returns:
            np.ndarray: The smoothed path with Gaussian noise applied to the coordinates.
        """
        smoothed_x = gaussian_filter1d(
            path[:, 0], sigma=0.5
        )  # apply gaussian filter to x coordinates
        smoothed_y = gaussian_filter1d(
            path[:, 1], sigma=0.5
        )  # apply gaussian filter to y coordinates

        noise_x = np.random.normal(
            0, noise_factor, size=smoothed_x.shape
        )  # generate noise for x coordinates
        noise_y = np.random.normal(
            0, noise_factor, size=smoothed_y.shape
        )  # generate noise for y coordinates

        t = np.linspace(0, 1, len(smoothed_x))  # Normalized time variable
        noise_decay = np.exp(-4 * t)  # Exponential decay function

        # Here we are adding noise decay so that noise reduces as we move from start to end

        smoothed_x += noise_x * noise_decay
        smoothed_y += noise_y * noise_decay

        # Apply final smoothing to prevent abrupt jumps
        smoothed_x = gaussian_filter1d(smoothed_x, sigma=0.5)
        smoothed_y = gaussian_filter1d(smoothed_y, sigma=0.5)
        return np.column_stack((smoothed_x, smoothed_y))

    def __interpolate_path(self, path: np.ndarray, num_points=50) -> np.ndarray:
        """
        Interpolate the predicted path to have a fixed number of intermediate points for smoother transitions.
        Parameters:
            path (np.ndarray): The predicted path as an array of shape (STEPS, 2).
            num_points (int): The number of points to interpolate to.
        Returns:
            np.ndarray: The interpolated path with a fixed number of points.
        """
        try:
            x = path[:, 0]
            y = path[:, 1]
            t = np.linspace(0, 1, len(x))

            interp_func_x = interp1d(t, x, kind="linear")
            interp_func_y = interp1d(t, y, kind="linear")

            t_new = np.linspace(0, 1, num_points)
            x_new = interp_func_x(t_new)
            y_new = interp_func_y(t_new)

            return np.column_stack((x_new, y_new))
        except Exception as e:
            print(f"Interpolation failed: {e}")
            return path

    def __predict_path(self, input_arr: np.ndarray) -> np.ndarray:
        """
        Predict the intermediate path steps between the start and destination coordinates.

        Parameters:
            input_arr (np.ndarray): The input data containing start and destination coordinates. e.g [[start_x, start_y, dest_x, dest_y]]

        Returns:
            np.ndarray: The predicted intermediate steps of the path.

        This method uses the pre-trained RNN model to generate intermediate steps between the start
        and destination. The input coordinates are normalized before being fed into the model.
        """

        # Normalize the input array to be within the [0, 1] range for model compatibility
        input_normalized_arr = self.__normalize(input_arr)

        # Convert the normalized array to a PyTorch tensor, send it to the appropriate device, and add a dimension for batch processing (unsqueeze simulates a batch of size 1)
        input_tensor = torch.tensor(input_normalized_arr).to(self.device).unsqueeze(1)

        # Clean up intermediate arrays to free memory
        del input_arr
        del input_normalized_arr

        with torch.no_grad():
            # Pass the input tensor through the model to get the output
            output = self.model(input_tensor)

        # Clean up the input tensor to free memory
        del input_tensor

        # Reshape the output to match the expected dimensions and move it to CPU
        output = output[0].view([self.STEPS, 2]).cpu().numpy()

        # Denormalize the output to convert it back to the original coordinate range
        output = self.__denormalize(output)

        return output

    def __add_speed_factor(self, path: np.ndarray) -> np.ndarray:
        """
        Add speed factor to the predicted path to simulate variable speed along the path.
        Parameters:
            path (np.ndarray): The predicted path as an array of shape (STEPS, 2).
        Returns:
            np.ndarray: The path with speed factor applied to simulate variable speed, with shape (STEPS, 3).
        """
        path_with_speed = np.zeros((len(path), 3), dtype=np.float64)
        for i in range(len(path)):
            progress = i / (len(path) - 1)
            angle = (
                self.__calculate_angle(path[i - 1], path[i], path[i + 1])
                if i > 0 and i < len(path) - 1
                else 0
            )

            angle_speed_factor = 1 - (angle / np.pi)
            progress_speed_factor = self.__calculate_speed_factor(progress)
            speed_factor = (0.60 * angle_speed_factor) + (
                0.4 * progress_speed_factor
            )  # 60% weight to angle and 40% to progress
            path_with_speed[i] = np.array([path[i][0], path[i][1], speed_factor])

        return path_with_speed

    def predict(self, start: np.ndarray, dest: np.ndarray) -> np.ndarray:
        """
        Predict the path from start to destination coordinates.

        Parameters:
            start (np.ndarray): The starting coordinates as [x, y].
            dest (np.ndarray): The destination coordinates as [x, y].

        Returns:
            np.ndarray: The predicted path with speed factor as an array of shape (STEPS, 3).
        """
        # Combine start and destination coordinates into a single input array
        input_arr = np.concatenate([start, dest]).reshape(
            1, 4
        )  # the input must be in [[x1,y1, x2, y2]] format, required by model

        # Predict the intermediate path using the model
        path = self.__predict_path(input_arr)
        path = np.vstack([start, path, dest])
        path = self.__clean_path(path)

        path = self.__smooth_path(path)
        interpolated_path = np.array([], dtype=np.float32)

        # Interpolate the path to ensure smooth transitions, make sure that there are no jumps, if distance between two points is more than 20 pixels, we interpolate
        for i in range(len(path) - 1):
            distance_with_previous_point = int(
                self._calculate_distance(path[i], path[i + 1]) if i > 0 else 0
            )
            if distance_with_previous_point >= 40:
                current_num_points = int(distance_with_previous_point / 20)
                current_interpolated_path = self.__interpolate_path(
                    path[i : i + 2], num_points=current_num_points
                )
                interpolated_path = (
                    np.vstack([interpolated_path, current_interpolated_path])
                    if interpolated_path.size
                    else current_interpolated_path
                )
            else:
                interpolated_path = (
                    np.vstack([interpolated_path, path[i]])
                    if interpolated_path.size
                    else path[i]
                )

        path = interpolated_path
        if not np.array_equal(path[-1], dest):
            path = np.vstack([path, dest])
        if not np.array_equal(path[0], start):
            path = np.vstack([start, path])

        path = self.__add_speed_factor(path)

        return path
