from src.utils.helpers import VietnameseTextProcessor, Helpers

# Initialize singletons
processor = VietnameseTextProcessor()
helpers = Helpers()

__all__ = ["processor", "helpers"]