from abc import ABC, abstractmethod


class Window(ABC):
    """
    Complementary Class to reduce size of classes using:

    * customtkinter.CTk
    * customtkinter.InputDialog
    * customtkinter.TopLevel
    """

    _valid_window_constructor_arguments: set = {"title", "icon", "width", "height"}

    def __init__(self, title: str | None, icon: str | None, width: int = 480, height: int = 480):
        """
        Contains the default values shared when creating a customtkinter window.

        :param title: used for the window title.
        :type title: str | None
        :param width: window width in pixels.
        :type width: int
        :param height: window height in pixels.
        :type height: int
        """

        self.window_title = title
        self.icon: str | None = icon
        self.width = width
        self.height = height

    @abstractmethod
    def _setup(self):
        """
        Abstract method to set up the window.
        """
        pass

    @abstractmethod
    def _ui_elements(self):
        """
        Abstract method to set up the window elements.
        """
        pass

    @staticmethod
    def center_to_display(
        screen_width: int,
        screen_height: int,
        width: int,
        height: int,
        scale_factor: float,
    ) -> str:
        """
        Centers the window to the main display.
        :param screen_width: Screen width in pixels.
        :type screen_width: int
        :param screen_height: Screen height in pixels.
        :type screen_height: int
        :param width: Window width in pixels.
        :type width: int
        :param height: Window height in pixels.
        :type height: int
        :param scale_factor: Multiplier of scale.
        :type scale_factor: float
        :return: String with size and displacement.
        """

        x = int(((screen_width / 2) - (width / 2)) * scale_factor)
        y = int(((screen_height / 2) - (height / 2)) * scale_factor)
        return f"{width}x{height}+{x}+{y}"

    # TODO: If we use TopLevels or InputDialogs, center it to parent.
    @staticmethod
    def center_to_parent():
        pass
