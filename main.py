from typing import Literal, Optional, Tuple, Type, Union

import customtkinter
from customtkinter import filedialog
from customtkinter.windows.widgets.utility import pop_from_dict_by_set
from PIL import Image, ImageFile

from window import Window


class App(customtkinter.CTk, Window):
    """
    UI for Reconstruction 3D App.
    """

    def __init__(self, fg_color: Optional[Union[str, Tuple[str, str]]] = None, **kwargs):
        """
        Initialize a customtkinter main app window receiving a Dictionary of params for
        customtkinter.CTk and Window.

        :param fg_color: Argument of customtkinter.CTk.
         Foreground color for main app. Can be a Color name: "red", or an HEX Color: "#AAAAAA".
        :type fg_color: Union[str, Tuple[str, str]]
        :param kwargs: Receives the required parameters for customtkinter.CTk and Window.
        """

        customtkinter.CTk.__init__(
            self,
            fg_color,
            **pop_from_dict_by_set(kwargs, self._valid_tk_constructor_arguments),
        )
        Window.__init__(
            self,
            **pop_from_dict_by_set(kwargs, self._valid_window_constructor_arguments),
        )
        self.__top_levels: dict[str, Type[customtkinter.CTkToplevel]] = kwargs.get("top_levels", {})
        self.__top_level: customtkinter.CTkToplevel | None = None
        self.selected_images: dict[str, ImageFile.ImageFile] = {}
        self.appearance: Literal["light", "dark", "system"] | str = kwargs.get("appearance", "system")
        self.color_theme: Literal["blue", "green", "dark-blue"] | str = kwargs.get("color_theme", "dark-blue")

        self._setup()
        self._ui_elements()

    # Main Window Setup

    def _setup(self):
        customtkinter.set_appearance_mode(self.appearance)
        customtkinter.set_default_color_theme(self.color_theme)
        self.geometry(
            Window.center_to_display(
                self.winfo_screenwidth(),
                self.winfo_screenheight(),
                self.width,
                self.height,
                self._get_window_scaling(),
            )
        )
        self.title(self.window_title)
        self.iconbitmap(default=self.icon)
        self.update()

    # Objects and Layout
    # TODO: Add a Carousel of Images to display them.
    # TODO: Add a button to start the 3D Reconstruction.
    # TODO: Add a Progress Top Level Window to show the progress of the 3D Reconstruction.
    def _ui_elements(self):
        """
        Creates the elements to displays and produces the layout.
        """

        self.button = customtkinter.CTkButton(self, text="Seleccionar Archivos", command=self.__select_images)
        self.button.pack(side="top", padx=20, pady=20)

    # Event Listeners

    def __open_top_level(self, top_level_key: str):
        """
        Opens a Top Level from the availables in the application.
        If there is one active, focus current.
        :param top_level_key: Selected Top Level name.
        """

        if self.__top_level is None or not self.__top_level.winfo_exists():
            if top_level_key in self.__top_levels:
                self.__top_level = self.__top_levels[top_level_key](self)

                self.__top_level.focus()
            else:
                valid_keys = ", ".join(self.__top_levels.keys())
                raise ValueError(f"Top level window '{top_level_key}' was not found. Valid options are: {valid_keys}")
        else:
            self.__top_level.focus()

    def __select_images(self):
        """
        Selects a series of Images to a Dictionary of paths and images.
        """

        images = filedialog.askopenfilenames(
            filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")],
            initialdir="~",
            parent=self,
            title="Selecciona las imágenes a procesar",
        )

        if images:
            self.selected_images.clear()

            for image in images:
                self.selected_images[image] = Image.open(image)


def main():
    app_top_levels: dict[str, Type[customtkinter.CTkToplevel]] = {}

    app_window_kwargs = {
        "title": "Reconstrucción 3D",
        "top_levels": app_top_levels,
        "width": 600,
        "height": 800,
        "appearance": "system",
        "color_theme": "themes/metal.json",
        "icon": "static/default.ico",
    }

    app = App(fg_color=None, **app_window_kwargs)
    app.mainloop()


if __name__ == "__main__":
    print("Running main.py")
    main()
