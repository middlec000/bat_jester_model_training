import numpy as np


def custom_print(
    data, max_list_length: int = 10, indent: int = 0, indent_str: str = "  "
):
    """
    Prints data in a pretty format, truncating lists if they exceed max_list_length.

    Args:
        data: The data to print.
        max_list_length: The maximum length of lists to display.
        indent: The current indentation level.
        indent_str: The string used for indentation.
    """
    if isinstance(data, list):
        if len(data) > max_list_length:
            print(f"[{', '.join(map(repr, data[:max_list_length]))}, ...]")
        else:
            print(f"[{', '.join(map(repr, data))}]")
    elif isinstance(data, dict):
        print("{")
        for key, value in data.items():
            print(f"{indent_str * (indent + 1)}{repr(key)}: ", end="")
            custom_print(value, max_list_length, indent + 1, indent_str)
        print(f"{indent_str * indent}" + "}")
    else:
        print(repr(data))


def segment_array(
    array_to_segment: np.array, window_size: int, window_overlap: float = 0.5
):
    """
    Segment a numpy array into chunks of specified size with a specified overlap.

    Parameters:
    array_to_segment (numpy.ndarray): Array to be segmented
    window_size (int): Size of each chunk/window
    window_overlap (float): Fraction of overlap between consecutive windows (default: 0.5, meaning 50%)

    Returns:
    numpy.ndarray: Array of segmented windows
    """
    # Input validation
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if not (0 <= window_overlap < 1):
        raise ValueError("window_overlap must be between 0 and 1")

    # Calculate the hop length (how much to shift for each window)
    hop_length = int(window_size * (1 - window_overlap))
    if hop_length <= 0:
        hop_length = 1  # Ensure at least one sample shift

    # Get array length
    n_samples = len(array_to_segment)

    # Initialize list to store segments
    segments = []

    # Calculate start positions for each window
    start_positions = range(0, n_samples, hop_length)

    for start_pos in start_positions:
        # Calculate end position
        end_pos = start_pos + window_size

        # If we've reached beyond the array, this is the last segment
        if end_pos > n_samples:
            # Extract the remaining samples
            segment = array_to_segment[start_pos:]

            # Pad to ensure it's of size window_size
            padding_length = window_size - len(segment)
            if padding_length > 0:
                segment = np.pad(
                    segment, (0, padding_length), "constant", constant_values=0
                )

            segments.append(segment)
            break
        else:
            # Extract the full segment
            segment = array_to_segment[start_pos:end_pos]
            segments.append(segment)

    # Convert list of segments to numpy array
    return np.array(segments)
