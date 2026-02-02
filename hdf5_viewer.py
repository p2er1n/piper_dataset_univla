import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import h5py
import numpy as np
from PIL import Image, ImageTk


class HDF5Viewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HDF5 Viewer")
        self.geometry("1100x700")

        self._file = None
        self._item_to_path = {}
        self._img_ref = None

        self._build_ui()

    def _build_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        open_btn = ttk.Button(toolbar, text="Open HDF5", command=self._open_file)
        open_btn.pack(side=tk.LEFT, padx=6, pady=6)

        refresh_btn = ttk.Button(toolbar, text="Refresh", command=self._refresh_tree)
        refresh_btn.pack(side=tk.LEFT, padx=6, pady=6)

        self._path_var = tk.StringVar(value="No file opened")
        path_label = ttk.Label(toolbar, textvariable=self._path_var)
        path_label.pack(side=tk.LEFT, padx=12)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=2)

        self._tree = ttk.Treeview(left, show="tree")
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        yscroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._tree.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.configure(yscrollcommand=yscroll.set)

        right_top = ttk.LabelFrame(right, text="Details")
        right_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._details = tk.Text(right_top, wrap=tk.NONE, height=18)
        self._details.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._details.configure(state=tk.DISABLED)

        details_scroll = ttk.Scrollbar(right_top, orient=tk.VERTICAL, command=self._details.yview)
        details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._details.configure(yscrollcommand=details_scroll.set)

        right_bottom = ttk.LabelFrame(right, text="Preview")
        right_bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._preview_container = ttk.Frame(right_bottom)
        self._preview_container.pack(fill=tk.BOTH, expand=True)
        self._preview_container.rowconfigure(1, weight=1)
        self._preview_container.columnconfigure(0, weight=1)

        self._preview_label = ttk.Label(
            self._preview_container, text="Select a dataset to preview"
        )
        self._preview_label.grid(row=1, column=0, sticky="nsew")

        self._table = None
        self._table_yscroll = None
        self._table_xscroll = None
        self._table_controls = None
        self._table_rows_var = tk.IntVar(value=200)
        self._table_cols_var = tk.IntVar(value=20)
        self._table_row_page_var = tk.IntVar(value=1)
        self._table_col_page_var = tk.IntVar(value=1)
        self._table_status_var = tk.StringVar(value="")
        self._flatten_var = tk.BooleanVar(value=False)

        self._current_ds = None
        self._table_row_count = 0
        self._table_col_count = 0

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open HDF5 File",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_file(path)

    def _load_file(self, path):
        self._close_file()
        try:
            self._file = h5py.File(path, "r")
        except Exception as exc:
            messagebox.showerror("Open Failed", f"Failed to open file:\n{exc}")
            return
        self._path_var.set(path)
        self._refresh_tree()

    def _close_file(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        self._item_to_path.clear()
        self._tree.delete(*self._tree.get_children())
        self._clear_details()
        self._set_preview_text("Select a dataset to preview")

    def _refresh_tree(self):
        if self._file is None:
            return
        self._tree.delete(*self._tree.get_children())
        self._item_to_path.clear()
        root_id = self._tree.insert("", "end", text="/", open=True)
        self._item_to_path[root_id] = "/"
        self._populate_tree(root_id, self._file)

    def _populate_tree(self, parent_id, group):
        for key in group.keys():
            obj = group[key]
            label = key
            if isinstance(obj, h5py.Dataset):
                label = f"{key} (dataset)"
            else:
                label = f"{key} (group)"
            item_id = self._tree.insert(parent_id, "end", text=label, open=False)
            self._item_to_path[item_id] = obj.name
            if isinstance(obj, h5py.Group):
                self._populate_tree(item_id, obj)

    def _on_select(self, _event):
        if self._file is None:
            return
        selected = self._tree.selection()
        if not selected:
            return
        item_id = selected[0]
        path = self._item_to_path.get(item_id)
        if not path or path == "/":
            self._show_group(self._file)
            return
        obj = self._file.get(path)
        if isinstance(obj, h5py.Group):
            self._show_group(obj)
            self._set_preview_text("Select a dataset to preview")
        elif isinstance(obj, h5py.Dataset):
            self._show_dataset(obj)
        else:
            self._set_details_text(f"Unknown object at {path}")
            self._set_preview_text("No preview available")

    def _show_group(self, group):
        lines = []
        lines.append(f"Type: Group")
        lines.append(f"Path: {group.name}")
        lines.append("")
        lines.append(f"Keys: {len(group.keys())}")
        for key in group.keys():
            lines.append(f"  - {key}")
        lines.append("")
        lines.append("Attributes:")
        if len(group.attrs) == 0:
            lines.append("  (none)")
        else:
            for k, v in group.attrs.items():
                lines.append(f"  {k}: {self._format_value(v)}")
        self._set_details_text("\n".join(lines))

    def _show_dataset(self, ds):
        lines = []
        lines.append("Type: Dataset")
        lines.append(f"Path: {ds.name}")
        lines.append(f"Shape: {ds.shape}")
        lines.append(f"Dtype: {ds.dtype}")
        lines.append("")
        lines.append("Attributes:")
        if len(ds.attrs) == 0:
            lines.append("  (none)")
        else:
            for k, v in ds.attrs.items():
                lines.append(f"  {k}: {self._format_value(v)}")
        lines.append("")
        lines.append("Preview:")
        preview_text, preview_image, table_data = self._preview_dataset(ds)
        lines.append(preview_text)
        self._set_details_text("\n".join(lines))
        if preview_image is not None:
            self._set_preview_image(preview_image)
        elif table_data is not None:
            self._set_preview_table(table_data)
        else:
            self._set_preview_text("No image preview available")

    def _preview_dataset(self, ds):
        try:
            if ds.ndim == 0:
                value = ds[()]
                return self._format_value(value), None, None

            if ds.ndim >= 1 and not self._flatten_var.get():
                shape_table = self._shape_table_data(ds)
                return "(table preview: shape only)", None, shape_table

            table_meta = self._to_table_meta(ds)
            if table_meta is not None:
                return "(table preview: paged view)", None, table_meta
            return "(no table preview available)", None, None
        except Exception as exc:
            return f"(preview failed: {exc})", None, None

    def _is_image_like(self, ds):
        if ds.ndim == 2:
            if ds.shape[0] > 2048 or ds.shape[1] > 2048:
                return False
            return True
        if ds.ndim == 3:
            if ds.shape[2] not in (1, 3, 4):
                return False
            if ds.shape[0] > 2048 or ds.shape[1] > 2048:
                return False
            return True
        return True

    def _to_image(self, data):
        if not isinstance(data, np.ndarray):
            return None
        if data.ndim not in (2, 3):
            return None
        if data.dtype != np.uint8:
            data = self._normalize_to_uint8(data)
        try:
            img = Image.fromarray(data)
        except Exception:
            return None
        return img

    def _normalize_to_uint8(self, data):
        data = np.asarray(data, dtype=np.float32)
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        if max_val <= min_val:
            return np.zeros_like(data, dtype=np.uint8)
        scaled = (data - min_val) / (max_val - min_val)
        return (scaled * 255.0).astype(np.uint8)

    def _format_value(self, value):
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except Exception:
                return repr(value)
        if isinstance(value, np.ndarray):
            return np.array2string(value, threshold=100, edgeitems=3)
        if np.isscalar(value):
            return str(value)
        return str(value)

    def _set_details_text(self, text):
        self._details.configure(state=tk.NORMAL)
        self._details.delete("1.0", tk.END)
        self._details.insert(tk.END, text)
        self._details.configure(state=tk.DISABLED)

    def _clear_details(self):
        self._set_details_text("")

    def _set_preview_text(self, text):
        self._clear_table()
        self._preview_label.configure(text=text, image="")
        self._preview_label.grid(row=1, column=0, sticky="nsew")
        self._img_ref = None

    def _set_preview_image(self, img):
        self._clear_table()
        self._preview_label.grid(row=1, column=0, sticky="nsew")
        max_w = self._preview_label.winfo_width() or 600
        max_h = self._preview_label.winfo_height() or 400
        img = img.copy()
        img.thumbnail((max_w, max_h))
        tk_img = ImageTk.PhotoImage(img)
        self._preview_label.configure(image=tk_img, text="")
        self._img_ref = tk_img

    def _set_preview_table(self, table_data):
        self._preview_label.configure(text="", image="")
        self._preview_label.grid_forget()
        self._img_ref = None
        self._build_table()
        if table_data.get("mode") == "shape":
            self._current_ds = table_data.get("ds")
            self._table_row_count = 0
            self._table_col_count = 0
            self._table_row_page_var.set(1)
            self._table_col_page_var.set(1)
            self._render_shape_table(table_data["rows"])
            return

        self._current_ds = table_data["ds"]
        self._table_row_count = table_data["row_count"]
        self._table_col_count = table_data["col_count"]
        self._table_row_page_var.set(1)
        self._table_col_page_var.set(1)
        self._update_table_page()

    def _build_table(self):
        if self._table is not None:
            return
        self._table_controls = ttk.Frame(self._preview_container)
        self._table_controls.grid(row=0, column=0, sticky="ew")
        self._table_controls.columnconfigure(9, weight=1)

        ttk.Label(self._table_controls, text="Rows/page").grid(
            row=0, column=0, padx=(6, 2), pady=4, sticky="w"
        )
        rows_entry = ttk.Entry(
            self._table_controls, textvariable=self._table_rows_var, width=6
        )
        rows_entry.grid(row=0, column=1, padx=(0, 8), pady=4, sticky="w")

        ttk.Label(self._table_controls, text="Cols/page").grid(
            row=0, column=2, padx=(0, 2), pady=4, sticky="w"
        )
        cols_entry = ttk.Entry(
            self._table_controls, textvariable=self._table_cols_var, width=6
        )
        cols_entry.grid(row=0, column=3, padx=(0, 8), pady=4, sticky="w")

        ttk.Checkbutton(
            self._table_controls,
            text="Flatten",
            variable=self._flatten_var,
            command=self._refresh_current_dataset,
        ).grid(row=0, column=4, padx=(0, 8), pady=4, sticky="w")

        ttk.Button(self._table_controls, text="Row◀", command=self._row_prev).grid(
            row=0, column=5, padx=(0, 4), pady=4, sticky="w"
        )
        ttk.Button(self._table_controls, text="Row▶", command=self._row_next).grid(
            row=0, column=6, padx=(0, 8), pady=4, sticky="w"
        )
        ttk.Button(self._table_controls, text="Col◀", command=self._col_prev).grid(
            row=0, column=7, padx=(0, 4), pady=4, sticky="w"
        )
        ttk.Button(self._table_controls, text="Col▶", command=self._col_next).grid(
            row=0, column=8, padx=(0, 8), pady=4, sticky="w"
        )

        ttk.Label(self._table_controls, textvariable=self._table_status_var).grid(
            row=0, column=9, padx=(0, 6), pady=4, sticky="e"
        )

        self._table = ttk.Treeview(self._preview_container)
        self._table_yscroll = ttk.Scrollbar(
            self._preview_container, orient=tk.VERTICAL, command=self._table.yview
        )
        self._table_xscroll = ttk.Scrollbar(
            self._preview_container, orient=tk.HORIZONTAL, command=self._table.xview
        )
        self._table.configure(
            yscrollcommand=self._table_yscroll.set,
            xscrollcommand=self._table_xscroll.set,
        )
        self._table.grid(row=1, column=0, sticky="nsew")
        self._table_yscroll.grid(row=1, column=1, sticky="ns")
        self._table_xscroll.grid(row=2, column=0, sticky="ew")
        self._table.bind("<Double-1>", self._on_table_double_click)

    def _clear_table(self):
        if self._table is None:
            return
        self._table.destroy()
        self._table_yscroll.destroy()
        self._table_xscroll.destroy()
        self._table_controls.destroy()
        self._table = None
        self._table_yscroll = None
        self._table_xscroll = None
        self._table_controls = None

    def _to_table_meta(self, ds):
        if ds.ndim == 0:
            return None
        if ds.ndim == 1:
            row_count = ds.shape[0]
            col_count = 1
        else:
            row_count = ds.shape[0]
            col_count = int(np.prod(ds.shape[1:]))
        return {"mode": "paged", "ds": ds, "row_count": row_count, "col_count": col_count}

    def _shape_table_data(self, ds):
        rows = []
        item_shape = ds.shape[1:] if ds.ndim >= 1 else ()
        for i in range(ds.shape[0]):
            rows.append([i, item_shape])
        return {"mode": "shape", "ds": ds, "rows": rows}

    def _row_prev(self):
        if self._current_ds is None:
            return
        if self._table_row_page_var.get() > 1:
            self._table_row_page_var.set(self._table_row_page_var.get() - 1)
        self._update_table_page()

    def _row_next(self):
        if self._current_ds is None:
            return
        self._table_row_page_var.set(self._table_row_page_var.get() + 1)
        self._update_table_page()

    def _col_prev(self):
        if self._current_ds is None:
            return
        if self._table_col_page_var.get() > 1:
            self._table_col_page_var.set(self._table_col_page_var.get() - 1)
        self._update_table_page()

    def _col_next(self):
        if self._current_ds is None:
            return
        self._table_col_page_var.set(self._table_col_page_var.get() + 1)
        self._update_table_page()

    def _update_table_page(self):
        if self._current_ds is None:
            return
        if self._current_ds.ndim >= 2 and not self._flatten_var.get():
            shape_table = self._shape_table_data(self._current_ds)
            self._render_shape_table(shape_table["rows"])
            return
        rows_per_page = max(int(self._table_rows_var.get()), 1)
        cols_per_page = max(int(self._table_cols_var.get()), 1)
        row_page = max(int(self._table_row_page_var.get()), 1)
        col_page = max(int(self._table_col_page_var.get()), 1)

        max_row_page = max((self._table_row_count - 1) // rows_per_page + 1, 1)
        max_col_page = max((self._table_col_count - 1) // cols_per_page + 1, 1)

        if row_page > max_row_page:
            row_page = max_row_page
            self._table_row_page_var.set(row_page)
        if col_page > max_col_page:
            col_page = max_col_page
            self._table_col_page_var.set(col_page)

        row_start = (row_page - 1) * rows_per_page
        row_end = min(row_start + rows_per_page, self._table_row_count)
        col_start = (col_page - 1) * cols_per_page
        col_end = min(col_start + cols_per_page, self._table_col_count)

        self._table_status_var.set(
            f"Rows {row_start + 1}-{row_end}/{self._table_row_count}  "
            f"Cols {col_start + 1}-{col_end}/{self._table_col_count}"
        )

        if self._current_ds.ndim == 1:
            data = self._current_ds[row_start:row_end]
            columns = ["index", "value"]
            rows = []
            for idx, val in enumerate(data.tolist(), start=row_start):
                rows.append([idx, self._format_value(val)])
        else:
            data = self._current_ds[row_start:row_end]
            reshaped = np.asarray(data).reshape(data.shape[0], -1)
            slice_block = reshaped[:, col_start:col_end]
            columns = ["row"] + [f"c{i}" for i in range(col_start, col_end)]
            rows = []
            for r, row_vals in enumerate(slice_block.tolist(), start=row_start):
                rows.append([r] + [self._format_value(v) for v in row_vals])

        self._table.configure(columns=columns, show="headings")
        for col in columns:
            self._table.heading(col, text=col)
            self._table.column(col, width=120, anchor=tk.W)
        self._table.delete(*self._table.get_children())
        for row in rows:
            self._table.insert("", "end", values=row)

    def _render_shape_table(self, rows):
        self._table_status_var.set("Shape view (double-click to open)")
        columns = ["index", "shape"]
        self._table.configure(columns=columns, show="headings")
        for col in columns:
            self._table.heading(col, text=col)
            self._table.column(col, width=140, anchor=tk.W)
        self._table.delete(*self._table.get_children())
        for row in rows:
            self._table.insert("", "end", values=row)

    def _refresh_current_dataset(self):
        if self._current_ds is None:
            return
        preview_text, preview_image, table_data = self._preview_dataset(self._current_ds)
        if preview_image is not None:
            self._set_preview_image(preview_image)
        elif table_data is not None:
            self._set_preview_table(table_data)
        else:
            self._set_preview_text("No image preview available")

    def _on_table_double_click(self, _event):
        if self._current_ds is None:
            return
        if self._flatten_var.get():
            return
        selected = self._table.selection()
        if not selected:
            return
        values = self._table.item(selected[0], "values")
        if not values:
            return
        try:
            idx = int(values[0])
        except Exception:
            return
        self._open_drill_window(self._current_ds, prefix=(idx,))

    def _open_drill_window(self, ds, prefix=()):
        win = tk.Toplevel(self)
        prefix_text = ",".join(str(i) for i in prefix)
        title_suffix = f"[{prefix_text}]" if prefix_text else ""
        win.title(f"Dataset View: {ds.name} {title_suffix}".strip())
        win.geometry("1000x700")

        remaining_shape = ds.shape[len(prefix):]
        info = ttk.Label(
            win,
            text=f"Shape: {remaining_shape}  Dtype: {ds.dtype}  Prefix: {prefix_text or '(none)'}",
        )
        info.pack(side=tk.TOP, anchor="w", padx=8, pady=6)

        controls = ttk.Frame(win)
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        image_var = tk.BooleanVar(value=False)
        image_check = ttk.Checkbutton(
            controls, text="Interpret as image", variable=image_var
        )
        image_check.pack(side=tk.LEFT, padx=(0, 10))

        status_var = tk.StringVar(value="")
        status_label = ttk.Label(win, textvariable=status_var)
        status_label.pack(side=tk.TOP, anchor="w", padx=8, pady=(0, 6))

        body = ttk.Frame(win)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        preview_label = ttk.Label(body, text="")
        preview_label.grid(row=0, column=0, sticky="nsew")

        table = ttk.Treeview(body)
        yscroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=table.yview)
        xscroll = ttk.Scrollbar(body, orient=tk.HORIZONTAL, command=table.xview)
        table.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        def show_table(columns, rows):
            preview_label.grid_forget()
            table.configure(columns=columns, show="headings")
            for col in columns:
                table.heading(col, text=col)
                table.column(col, width=120, anchor=tk.W)
            table.delete(*table.get_children())
            for row in rows:
                table.insert("", "end", values=row)
            table.grid(row=0, column=0, sticky="nsew")
            yscroll.grid(row=0, column=1, sticky="ns")
            xscroll.grid(row=1, column=0, sticky="ew")

        def show_image(img):
            table.grid_forget()
            yscroll.grid_forget()
            xscroll.grid_forget()
            max_w = preview_label.winfo_width() or 800
            max_h = preview_label.winfo_height() or 500
            img = img.copy()
            img.thumbnail((max_w, max_h))
            tk_img = ImageTk.PhotoImage(img)
            preview_label.configure(image=tk_img, text="")
            preview_label.image = tk_img
            preview_label.grid(row=0, column=0, sticky="nsew")

        def load_current_view():
            status_var.set("")
            if len(remaining_shape) > 2:
                if image_var.get() and len(remaining_shape) in (2, 3):
                    slicer = prefix + (slice(None),) * (ds.ndim - len(prefix))
                    try:
                        data = ds[slicer]
                    except Exception as exc:
                        status_var.set(f"Load failed: {exc}")
                        return
                    array = np.asarray(data)
                    img = self._to_image(array)
                    if img is not None:
                        show_image(img)
                        return
                    status_var.set("Cannot interpret current data as an image.")
                rows = []
                for i in range(remaining_shape[0]):
                    rows.append([i, remaining_shape[1:]])
                show_table(["index", "shape"], rows)
                return

            slicer = prefix + (slice(None),) * (ds.ndim - len(prefix))
            try:
                data = ds[slicer]
            except Exception as exc:
                status_var.set(f"Load failed: {exc}")
                return

            array = np.asarray(data)
            if image_var.get():
                img = self._to_image(array)
                if img is not None:
                    show_image(img)
                    return
                status_var.set("Cannot interpret current data as an image.")

            if array.ndim == 0:
                show_table(["value"], [[self._format_value(array.item())]])
            elif array.ndim == 1:
                rows = [[i, self._format_value(v)] for i, v in enumerate(array.tolist())]
                show_table(["index", "value"], rows)
            else:
                columns = ["row"] + [f"c{i}" for i in range(array.shape[1])]
                rows = []
                for r in range(array.shape[0]):
                    rows.append(
                        [r] + [self._format_value(v) for v in array[r].tolist()]
                    )
                show_table(columns, rows)

        def on_double_click(_event):
            if len(remaining_shape) <= 2:
                return
            selected = table.selection()
            if not selected:
                return
            values = table.item(selected[0], "values")
            if not values:
                return
            try:
                idx = int(values[0])
            except Exception:
                return
            self._open_drill_window(ds, prefix=prefix + (idx,))

        def on_image_toggle(*_args):
            win.after_idle(load_current_view)

        image_var.trace_add("write", on_image_toggle)
        table.bind("<Double-1>", on_double_click)
        load_current_view()


def main():
    app = HDF5Viewer()
    app.mainloop()


if __name__ == "__main__":
    main()
