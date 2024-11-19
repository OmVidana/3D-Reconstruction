import os
import tkinter
from typing import Any, Dict, Literal, Set, Tuple

import customtkinter
from PIL import Image


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
        initial_images: Set[str] = set(),
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

        self.images_paths: Set[str] = initial_images
        self.image_widgets: Dict[str, customtkinter.CTkLabel] = {}
        self.no_images_label = customtkinter.CTkLabel(self, text="Aquí se mostrarán sus imágenes", anchor="center")

        if self.images_paths:
            for image in sorted(self.images_paths, key=lambda x: os.path.getctime(x)):
                self.add_image(image)
        else:
            self.no_images_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(0, weight=1)

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
            for image_path in sorted(self.images_paths, key=lambda x: os.path.getctime(x)):
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
