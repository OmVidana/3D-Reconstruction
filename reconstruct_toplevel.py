import threading
from tkinter import messagebox
from typing import Tuple

import customtkinter
from customtkinter.windows.widgets.utility import pop_from_dict_by_set

# from test import reconstruct_3d
from test2 import reconstruct_3d
# from test3 import reconstruct_3d
# from test4 import reconstruct_3d
# from reconstruction_3d import reconstruct_3d
# from reconstruction_3d import reconstruct_3d_mock as reconstruct_3d
from window import Window


class ToplevelReconstruction(customtkinter.CTkToplevel, Window):
    def __init__(self, *args, fg_color: str | Tuple[str, str] | None = None, **kwargs):
        customtkinter.CTkToplevel.__init__(
            self,
            *args,
            fg_color=fg_color,
            **pop_from_dict_by_set(kwargs, self._valid_tk_toplevel_arguments),
        )
        Window.__init__(
            self,
            **pop_from_dict_by_set(kwargs, self._valid_window_constructor_arguments),
        )

        self.thread = None

        self._setup_window()
        self._ui_elements_window()
        self.update()

    def _setup_window(self):
        self.title(self.window_title)
        self.iconbitmap(default=self.icon)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.__on_close_operation)

        self.geometry(
            Window.center_to_parent(
                self.master.winfo_width(),
                self.master.winfo_height(),
                self.master.winfo_x(),
                self.master.winfo_y(),
                self.width,
                self.height,
            )
        )

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

    def _ui_elements_window(self):
        self.progress_label = customtkinter.CTkLabel(
            self, text="Trabajando en la Reconstrucción 3D, espere unos instantes..."
        )
        self.progress_label.grid(row=0, column=0, padx=(8, 8), pady=(16, 24), sticky="nsew")
        self.close_button = customtkinter.CTkButton(self, text="Cancelar", command=self.__on_cancel_operation)
        self.close_button.grid(row=1, column=0, padx=(8, 8), pady=(16, 24), sticky="nsew")

    def start_reconstruction(self, image_paths, output_path):
        self.thread = threading.Thread(target=self.__run_reconstruction, args=(image_paths, output_path), daemon=True)
        self.thread.start()

    def __run_reconstruction(self, image_paths, output_path):
        try:
            reconstruct_3d(image_paths, output_path)
        except Exception as e:
            messagebox.showerror("Error en la reconstrucción", f"Se produjo un error: {str(e)}")
        finally:
            self.__finalize_reconstruction()

    def __on_cancel_operation(self):
        if self.thread and self.thread.is_alive():
            self.thread.join(2.0)
            self.destroy()
        else:
            self.destroy()

    def __finalize_reconstruction(self):
        self.progress_label.configure(text="Reconstrucción 3D completada.")
        self.close_button.configure(text="Salir")

    def __on_close_operation(self):
        if self.thread and self.thread.is_alive():
            messagebox.showwarning(
                "Error",
                "No puede cerrar la ventana mientras la reconstrucción esté en progreso.",
                parent=self,
            )
        else:
            self.destroy()
