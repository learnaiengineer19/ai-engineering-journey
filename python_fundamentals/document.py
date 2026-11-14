# --------------------------------------------------
# 1. DEFINE A CLASS
# --------------------------------------------------


class Document:
    """
    Represent a text document and calculate its metadata.

    Attributes:
        filename: Name of the document.
        content: Text stored in the document.
    """

    def __init__(self, filename: str, content: str) -> None:
        self.filename = filename
        self.content = content

    def word_count(self) -> int:
        """Return the number of words in the document."""
        return len(self.content.split())

    def character_count(self) -> int:
        """Return the total number of characters."""
        return len(self.content)

    def get_metadata(self) -> dict[str, str | int]:
        """Return document filename and text statistics."""
        metadata = {
            "filename": self.filename,
            "word_count": self.word_count(),
            "character_count": self.character_count()
        }
        return metadata