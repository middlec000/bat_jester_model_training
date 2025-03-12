def custom_print(data, max_list_length=10, indent=0, indent_str="  "):
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
