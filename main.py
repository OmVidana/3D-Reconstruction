import os
from tkinter import messagebox
from typing import Literal, Optional, Tuple, Union

import customtkinter
from customtkinter import filedialog
from customtkinter.windows.widgets.utility import pop_from_dict_by_set

from carousel import Carousel
from reconstruct_toplevel import ToplevelReconstruction
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
        self.__reconstruct_toplevel: ToplevelReconstruction | None = None
        self.appearance: Literal["light", "dark", "system"] | str = kwargs.get("appearance", "system")
        self.color_theme: Literal["blue", "green", "dark-blue"] | str = kwargs.get("color_theme", "dark-blue")

        self._setup_window()
        self._ui_elements_window()
        self.update()

    # Main Window Setup

    def _setup_window(self):
        customtkinter.set_appearance_mode(self.appearance)
        customtkinter.set_default_color_theme(self.color_theme)
        self.title(self.window_title)
        self.iconbitmap(default=self.icon)
        self.resizable(True, False)
        self.protocol("WM_DELETE_WINDOW", self.__on_close_operation)

        self.geometry(
            Window.center_to_display(
                self.winfo_screenwidth(),
                self.winfo_screenheight(),
                self.width,
                self.height,
                self._get_window_scaling(),
            )
        )

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=3)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=2)

    # Objects and Layout
    def _ui_elements_window(self):
        """
        Creates the elements to displays and produces the layout.
        """

        self.choose_button = customtkinter.CTkButton(self, text="Seleccionar Imágenes", command=self.__select_images)
        self.choose_button.grid(row=0, column=0, padx=(32, 32), pady=(32, 16), sticky="nsew")
        self.carousel = Carousel(self, label_text="Imágenes Seleccionadas", label_fg_color="transparent")
        self.carousel.grid(row=1, column=0, padx=(32, 32), pady=(16, 16), sticky="nsew")
        self.clean_button = customtkinter.CTkButton(self, text="Limpiar Seleccion", command=self.carousel.clear)
        self.clean_button.grid(row=2, column=0, padx=(32, 32), pady=(16, 16), sticky="nsew")
        self.reconstruct_button = customtkinter.CTkButton(
            self, text="Comenzar a Reconstruir en 3D", command=self.__start_reconstruction
        )
        self.reconstruct_button.grid(row=3, column=0, padx=(32, 32), pady=(16, 32), sticky="nsew")

    # Event Listeners

    def __select_images(self):
        """
        Selects a series of Images to a Dictionary of paths and images.
        """

        image_paths = filedialog.askopenfilenames(
            filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")],
            initialdir="~",
            parent=self,
            title="Selecciona las imágenes a procesar",
        )

        if image_paths:
            for image_path in image_paths:
                if image_path not in self.carousel.images_paths:
                    self.carousel.add_image(image_path)

    def __start_reconstruction(self):
        """
        Starts the 3D Reconstruction process.
        """
        if not self.carousel.images_paths:
            messagebox.showwarning(
                "Sin Imágenes",
                "No hay imágenes seleccionadas para realizar la reconstrucción 3D.",
                parent=self,
            )
            return

        if len(self.carousel.images_paths) < 2:
            messagebox.showwarning(
                "Pocas Imágenes",
                "Se necesitan al menos dos imágenes para realizar la reconstrucción 3D.",
                parent=self,
            )
            return

        if self.__reconstruct_toplevel is None or not self.__reconstruct_toplevel.winfo_exists():
            top_level_kwargs = {
                "window_title": "Procesando Reconstrucción 3D",
                "icon": self.icon,
                "width": 200,
                "height": 120,
            }
            self.__reconstruct_toplevel = ToplevelReconstruction(self, **top_level_kwargs)
            self.bind(
                "<<ReconstructionComplete>>",
                lambda event: messagebox.showinfo("Completado", f"Archivo creado en: {event}"),
            )
            self.bind(
                "<<ReconstructionError>>",
                lambda event: messagebox.showerror("Error", f"Ocurrió un error: {event}"),
            )
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.obj")
            self.__reconstruct_toplevel.start_reconstruction(self.carousel.images_paths, output_path)
            self.__reconstruct_toplevel.focus()
        else:
            self.__reconstruct_toplevel.focus()

    def __on_close_operation(self):
        if self.__reconstruct_toplevel is not None and self.__reconstruct_toplevel.winfo_exists():
            messagebox.showwarning(
                "Error",
                "No puede cerrar la ventana mientras la reconstrucción esté en progreso.",
                parent=self,
            )
        else:
            self.quit()


def main():
    app_window_kwargs = {
        "window_title": "Reconstrucción 3D",
        "width": 600,
        "height": 800,
        "appearance": "system",
        "color_theme": "themes/metal.json",
        "icon": "static/default.ico",
    }

    app = App(fg_color=None, **app_window_kwargs)
    app.mainloop()


if __name__ == "__main__":
    main()
