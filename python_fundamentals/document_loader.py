def load_document(file_path):
    """
    Read a text file and return its content.

    Args:
        file_path: Path to the document.

    Returns:
        str: Contents of the file.
        None: If the file does not exist.
    """
    try:    
        with open (file_path, "r", encoding="utf-8") as file:
            contents = file.read()
            return contents
        
    except FileNotFoundError as error:
        print("Document could not be found:", error) 
        return None