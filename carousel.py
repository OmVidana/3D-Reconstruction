import os
import tkinter
from datetime import datetime
from typing import Any, Dict, Literal, Set, Tuple

import customtkinter
from PIL import Image
from PIL.ExifTags import TAGS


class Carousel(customtkinter.CTkScrollableFrame):
    def __init__(
            self,
            master: Any,
            width: int = 200,
            height: int = 200,
            corner_radius: int | str | None = None,
            border_width: int | str | None = None,
            bg_color: str | Tuple[str, str] = "transparent",
            fg_color: str | Tuple[str, str] | None = None,
            border_color: str | Tuple[str, str] | None = None,
            scrollbar_fg_color: str | Tuple[str, str] | None = None,
            scrollbar_button_color: str | Tuple[str, str] | None = None,
            scrollbar_button_hover_color: str | Tuple[str, str] | None = None,
            label_fg_color: str | Tuple[str, str] | None = None,
            label_text_color: str | Tuple[str, str] | None = None,
            label_text: str = "",
            label_font: tuple | customtkinter.CTkFont | None = None,
            label_anchor: str = "center",
            orientation: Literal["vertical"] | Literal["horizontal"] = "vertical",
            initial_images: Set[str] = None,
    ):
        super().__init__(
            master,
            width,
            height,
            corner_radius,
            border_width,
            bg_color,
            fg_color,
            border_color,
            scrollbar_fg_color,
            scrollbar_button_color,
            scrollbar_button_hover_color,
            label_fg_color,
            label_text_color,
            label_text,
            label_font,
            label_anchor,
            orientation,
        )

        self.images_paths: Set[str] = initial_images if initial_images is not None else set()
        self.image_widgets: Dict[str, customtkinter.CTkLabel] = {}
        self.no_images_label = customtkinter.CTkLabel(self, text="Aquí se mostrarán sus imágenes", anchor="center")

        if self.images_paths:
            for image in self.images_paths:
                self.add_image(image)
        else:
            self.no_images_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(0, weight=1)

    @staticmethod
    def get_image_date(image_path: str) -> float:
        """
        Obtains the date original of the image. If it is not available it uses the last modification.
        """

        try:
            image = Image.open(image_path)
            exif = image.getexif()
            date_time_original: float = 0.0
            for key, value in TAGS.items():
                if value == "ExifOffset":
                    date_time_original = datetime.strptime(exif.get_ifd(key)[36867], "%Y:%m:%d %H:%M:%S").timestamp()
                    break
            return date_time_original
        except Exception as e:
            print(f"Error al obtener los metadatos de {image_path}: {e}")

        try:
            modified_date = os.path.getmtime(image_path)
            return modified_date
        except Exception as e:
            print(f"Error al obtener la fecha de modificación de {image_path}: {e}")
            return float("inf")

    def add_image(self, image_path: str):
        """
        Add an image to the carousel while preserving its aspect ratio.
        """

        try:
            image = Image.open(image_path)
            image_ctk = customtkinter.CTkImage(light_image=image, dark_image=image, size=(288, 288))
            self.image_widgets[image_path] = customtkinter.CTkLabel(self, image=image_ctk, text="")
            self.images_paths.add(image_path)
            self.update_grid()

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    def remove_image(self, image_path: str):
        """
        Remove an image from the carousel.
        """
        if image_path in self.image_widgets:
            self.image_widgets[image_path].destroy()
            del self.image_widgets[image_path]
            self.images_paths.discard(image_path)
            self.update_grid()

    def clear(self):
        """
        Clear all images from the carousel.
        """
        for widget in self.image_widgets.values():
            widget.grid_forget()
        self.image_widgets.clear()
        self.images_paths.clear()
        self.update_grid()

    def update_grid(self):
        """
        Update the grid layout for images.
        """
        for widget in self.image_widgets.values():
            widget.grid_forget()

        if len(self.images_paths) == 0:
            self.no_images_label.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        else:
            self.no_images_label.grid_forget()

            row = 0
            for image_path in sorted(self.images_paths, key=self.get_image_date):
                self.image_widgets[image_path].grid(row=row, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")
                row += 1
                self.image_widgets[image_path].bind(
                    "<Button-3>", lambda e, path=image_path: self._show_context_menu(e, path)
                )

        self.update_idletasks()

    def _show_context_menu(self, event, image_path: str):
        """
        Show context menu for removing an image.
        """
        menu = tkinter.Menu(self, tearoff=False)
        menu.add_command(label="Eliminar", command=lambda: self.remove_image(image_path))
        menu.tk_popup(event.x_root, event.y_root)
