import tkinter as T
from tkinter import ttk
from tkinter import filedialog
import cv2
import numpy as np
import random
import PIL
from PIL import ImageTk
import threading

from preprocessing import denoise, sharpen
from cluster import cluster_image, recolour
from contours import draw_contours
from smooth import smooth_image

# TODO Try another gui. Qt?
# TODO change style


_als = (T.N, T.W, T.E, T.S)
_tor = (T.N, T.E, T.S)


def show_image(image, window_name):
    if image is None:
        return
    w, h, _ = image.shape
    image = cv2.resize(image, (h * 1000 // max(w, h), w * 1000 // max(w, h)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t_image = ImageTk.PhotoImage(image=PIL.Image.fromarray(image))

    # ugly? yeah
    window = T.Toplevel()
    window.title(window_name)
    label = ttk.Label(window, image=t_image)
    label.grid(column=0, row=0, sticky=_als)
    label.image = t_image


# caches output for previous input
class CachedCalculationFrame():
    def __init__(self, main):
        self.main = main
        self.cached_for_input = None
        self.cached_output = None
        self.updated = True

    def calc(self, input):
        pass

class LoadImageFrame(CachedCalculationFrame):
    def __init__(self, main, row):
        super().__init__(main)
        mainframe = ttk.Frame(main.root, padding=5, borderwidth=2, relief="groove")
        mainframe.grid(column=0, row=row, sticky=_als)

        self.load_image_button = ttk.Button(mainframe, text="load image", command=self.load_image)
        self.load_image_button.grid(column=1, row=1, sticky=_als)
        self.image_filename = None

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

    def load_image(self):
        self.image_filename = filedialog.askopenfilename(title="Choose image",
                filetypes=[("Images", (".bmp", ".dip", ".jpeg", ".jpg",
                ".jpe", ".jp2", ".png", ".webp", ".webpm", ".pbm", ".pgm",
                ".ppm", ".pxm", ".pnm", ".sr", ".ras", ".tiff", ".tif", ".hdr", ".pic"))])

    def calc(self, input):
        if self.image_filename is None or self.image_filename == "":
            return None
        assert(input is None)
        self.cached_output = cv2.imread(self.image_filename)
        return self.cached_output

class ResizeImageFrame(CachedCalculationFrame):
    def __init__(self, main, row):
        super().__init__(main)
        mainframe = ttk.Frame(main.root, padding=5, borderwidth=2, relief="groove")
        mainframe.grid(column=0, row=row, sticky=_als)

        ttk.Label(mainframe, text="width").grid(column=0, row=0, sticky=_als)
        self.width_pixels = T.StringVar()
        self.width_pixels_entry = ttk.Entry(mainframe, textvariable=self.width_pixels)
        self.width_pixels_entry.grid(column=0, row=1, sticky=_als)

        ttk.Label(mainframe, text="height").grid(column=1, row=0, sticky=_als)
        self.height_pixels = T.StringVar()
        self.height_pixels_entry = ttk.Entry(mainframe, textvariable=self.height_pixels)
        self.height_pixels_entry.grid(column=1, row=1, sticky=_als)

        ttk.Label(mainframe, text="ratio").grid(column=2, row=0, sticky=_als)
        self.ratio = T.StringVar(value="3.0")
        self.ratio_entry = ttk.Entry(mainframe, textvariable=self.ratio)
        self.ratio_entry.grid(column=2, row=1, sticky=_als)

        def clear_ratio(*args):
            #  self.ratio_entry.delete(0, T.END)
            self.updated = True

        self.ratio_entry.bind("<Return>", self.update_width_height_from_ratio)
        self.width_pixels.trace_add("write", clear_ratio)
        self.height_pixels.trace_add("write", clear_ratio)

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

    def update_width_height_from_ratio(self, *args):
        self.updated = True
        if self.cached_for_input is None:
            self.width_pixels_entry.delete(0, T.END)
            self.height_pixels_entry.delete(0, T.END)
        else:
            try:
                ratio = float(self.ratio.get())
            except ValueError:
                return
            h, w, _ = self.cached_for_input.shape
            nw = int(ratio * w)
            nh = int(ratio * h)
            self.width_pixels.set(str(nw))
            self.height_pixels.set(str(nh))

    def calc(self, input):
        if input is None:
            return None

        if not self.updated and self.cached_for_input is not None and np.array_equal(self.cached_for_input, input):
            return self.cached_output

        self.cached_for_input = input
        try:
            if self.ratio.get() != "":
                self.update_width_height_from_ratio()
            nw = int(self.width_pixels.get())
            nh = int(self.height_pixels.get())
        except ValueError:
            self.updated = True
            return None

        self.cached_output = cv2.resize(input, (nw, nh))
        self.updated = False
        return self.cached_output

class PreprocessingFrame(CachedCalculationFrame):
    def __init__(self, main, row):
        super().__init__(main)
        mainframe = ttk.Frame(main.root, padding=5, borderwidth=2, relief="groove")
        mainframe.grid(column=0, row=row, sticky=_als)

        self.denoise1_var = T.StringVar(value="denoise1: 3")
        self.denoise1_label = ttk.Label(mainframe, textvariable=self.denoise1_var)
        self.denoise1_label.grid(column=0, row=0, sticky=_als)
        self.denoise1 = T.IntVar(value=3)
        self.denoise1_scale = ttk.Scale(mainframe, variable=self.denoise1, from_=0, to=10, length=200,
                orient=T.HORIZONTAL, command=self.update_denoise1)
        self.denoise1_scale.grid(column=0, row=1, sticky=_als)

        self.denoise2_var = T.StringVar(value="denoise2: 3")
        self.denoise2_label = ttk.Label(mainframe, textvariable=self.denoise2_var)
        self.denoise2_label.grid(column=1, row=0, sticky=_als)
        self.denoise2 = T.IntVar(value=3)
        self.denoise2_scale = ttk.Scale(mainframe, variable=self.denoise2, from_=0, to=10, length=200,
                orient=T.HORIZONTAL, command=self.update_denoise2)
        self.denoise2_scale.grid(column=1, row=1, sticky=_als)

        self.sharpness_var = T.StringVar(value="sharpness: 0.0")
        self.sharpness_label = ttk.Label(mainframe, textvariable=self.sharpness_var)
        self.sharpness_label.grid(column=2, row=0, sticky=_als)
        self.sharpness = T.DoubleVar(value=0.0)
        self.sharpness_scale = ttk.Scale(mainframe, variable=self.sharpness, from_=-1.0, to=1.0, length=200, orient=T.HORIZONTAL, command=self.update_sharpness)
        self.sharpness_scale.grid(column=2, row=1, sticky=_als)
        self.sharpness_scale.bind("<Double-1>", self.reset_sharpness)

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

    def update_denoise1(self, value):
        self.updated = True
        self.denoise1_var.set("denoise1: {}".format(int(float(value))))

    def update_denoise2(self, value):
        self.updated = True
        self.denoise2_var.set("denoise2: {}".format(int(float(value))))

    def update_sharpness(self, value):
        self.updated = True
        self.sharpness_var.set("sharpness: {:.3f}".format(float(value)))

    def reset_sharpness(self, *args):
        self.updated = True
        self.sharpness_scale.set(0.0)

    def calc(self, input):
        if input is None:
            return None

        if not self.updated and self.cached_for_input is not None and np.array_equal(self.cached_for_input, input):
            return self.cached_output

        self.cached_for_input = input
        denoised_image = denoise(input, self.denoise1.get(), self.denoise2.get())
        sharpened_image = sharpen(denoised_image, kernel_size=(5, 5), sharpness=self.sharpness.get(), iterations=3)

        self.cached_output = sharpened_image
        self.updated = False
        return self.cached_output

class NumColorsFrame(CachedCalculationFrame):
    def __init__(self, main, row):
        super().__init__(main)
        mainframe = ttk.Frame(main.root, padding=5, borderwidth=2, relief="groove")
        mainframe.grid(column=0, row=row, sticky=_als)

        self.numcolors_var = T.StringVar(value="num colors: 8")
        self.numcolors_label = ttk.Label(mainframe, textvariable=self.numcolors_var)
        self.numcolors_label.grid(column=0, row=0, sticky=_als)
        self.numcolors = T.IntVar(value=8)
        self.numcolors_scale = ttk.Scale(mainframe, variable=self.numcolors, from_=0, to=64, length=200,
                orient=T.HORIZONTAL, command=self.update_numcolors)
        self.numcolors_scale.grid(column=0, row=1, sticky=_als)

        self.cached_output = (None, None, None)
        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

    def update_numcolors(self, value):
        self.updated = True
        self.numcolors_var.set("numcolors: {}".format(int(float(value))))

    def calc(self, input):
        if input is None:
            return None
        if not self.updated and self.cached_for_input is not None and np.array_equal(self.cached_for_input, input):
            return self.cached_output
        self.cached_for_input = input
        labels, centers = cluster_image(input, self.numcolors.get())

        self.cached_output = (input, labels, centers)
        self.updated = False
        return self.cached_output

class SmoothFrame(CachedCalculationFrame):
    def __init__(self, main, row):
        super().__init__(main)
        mainframe = ttk.Frame(main.root, padding=5, borderwidth=2, relief="groove")
        mainframe.grid(column=0, row=row, sticky=_als)

        self.smoothing_var = T.StringVar(value="smoothing: 3")
        self.smoothing_label = ttk.Label(mainframe, textvariable=self.smoothing_var)
        self.smoothing_label.grid(column=0, row=0, sticky=_als)
        self.smoothing = T.IntVar(value=3)
        self.smoothing_scale = ttk.Scale(mainframe, variable=self.smoothing, from_=0, to=10, length=200,
                orient=T.HORIZONTAL, command=self.update_smoothing)
        self.smoothing_scale.grid(column=0, row=1, sticky=_als)

        self.cached_output = (None, None, None)
        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

    def update_smoothing(self, value):
        self.updated = True
        self.smoothing_var.set("smoothing: {}".format(int(float(value))))

    def calc(self, input):
        if input is None:
            return None
        if not self.updated and self.cached_for_input is not None and all([np.array_equal(self.cached_for_input[i], input[i]) for i in range(3)]):
            return self.cached_output
        self.cached_for_input = input
        image, labels, centers = input
        smoothed_labels = smooth_image(labels, 3 + np.arange(self.smoothing.get()), 1)

        self.cached_output = (image, smoothed_labels, centers)
        self.updated = False
        return self.cached_output

class ContoursFrame(CachedCalculationFrame):
    def __init__(self, main, row):
        super().__init__(main)
        mainframe = ttk.Frame(main.root, padding=5, borderwidth=2, relief="groove")
        mainframe.grid(column=0, row=row, sticky=_als)

        self.minarea_var = T.StringVar(value="minarea: 50")
        self.minarea_label = ttk.Label(mainframe, textvariable=self.minarea_var)
        self.minarea_label.grid(column=0, row=0, sticky=_als)
        self.minarea = T.IntVar(value=50)
        self.minarea_scale = ttk.Scale(mainframe, variable=self.minarea, from_=0, to=500, length=200,
                orient=T.HORIZONTAL, command=self.update_minarea)
        self.minarea_scale.grid(column=0, row=1, sticky=_als)

        self.digitsize_var = T.StringVar(value="digitsize: 8")
        self.digitsize_label = ttk.Label(mainframe, textvariable=self.digitsize_var)
        self.digitsize_label.grid(column=1, row=0, sticky=_als)
        self.digitsize = T.IntVar(value=8)
        self.digitsize_scale = ttk.Scale(mainframe, variable=self.digitsize, from_=0, to=50, length=200,
                orient=T.HORIZONTAL, command=self.update_digitsize)
        self.digitsize_scale.grid(column=1, row=1, sticky=_als)

        self.gapclose_var = T.StringVar(value="gapclose: 5")
        self.gapclose_label = ttk.Label(mainframe, textvariable=self.gapclose_var)
        self.gapclose_label.grid(column=2, row=0, sticky=_als)
        self.gapclose = T.IntVar(value=5)
        self.gapclose_scale = ttk.Scale(mainframe, variable=self.gapclose, from_=0, to=10, length=200,
                orient=T.HORIZONTAL, command=self.update_gapclose)
        self.gapclose_scale.grid(column=2, row=1, sticky=_als)

        self.warning_label_var = T.StringVar("")
        self.warning_label = ttk.Label(mainframe, textvariable=self.warning_label_var)
        self.warning_label.grid(column=0, row=2, sticky=_als, columnspan=3)

        self.cached_output = (None, None)

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

    def update_minarea(self, value):
        self.updated = True
        self.minarea_var.set("minarea: {}".format(int(float(value))))

    def update_digitsize(self, value):
        self.updated = True
        self.digitsize_var.set("digitsize: {}".format(int(float(value))))

    def update_gapclose(self, value):
        self.updated = True
        self.gapclose_var.set("gapclose: {}".format(int(float(value))))

    def calc(self, input):
        if input is None:
            return None
        image, labels, centers = input
        if not self.updated and self.cached_for_input is not None and all([np.array_equal(self.cached_for_input[i], input[i]) for i in range(2)]):
            return self.cached_output
        self.cached_for_input = input

        output_contours, output_labels = draw_contours(centers[labels], labels,
                min_area=self.minarea.get(), min_radius=self.digitsize.get(), gaps_smooth=self.gapclose.get())
        output_image = recolour(image, output_labels)
        output_contours = cv2.cvtColor(output_contours, cv2.COLOR_GRAY2BGR)
        if np.min(output_labels):
            self.warning_label_var.set("Warning: not every pixel coloured")
        else:
            self.warning_label_var.set("")

        self.cached_output = (output_image, output_contours)
        self.updated = False
        return self.cached_output

class SaveFrame(CachedCalculationFrame):
    def __init__(self, main, row):
        super().__init__(main)
        mainframe = ttk.Frame(main.root, padding=5, borderwidth=2, relief="groove")
        mainframe.grid(column=0, row=row, sticky=(T.N, T.W, T.E, T.S))

        self.update_button = ttk.Button(mainframe, text="update", command=self.update)
        self.update_button.grid(column=0, row=0, sticky=(T.N, T.W, T.E, T.S))

        self.save_button = ttk.Button(mainframe, text="save", command=self.save)
        self.save_button.grid(column=1, row=0, sticky=(T.N, T.W, T.E, T.S))

        self.output_image, self.output_contours = None, None

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

    def update(self):
        self.main.calc()

    def save(self):
        out_filename = filedialog.asksaveasfilename(title="Save image")
        if out_filename is not None and out_filename != "" and self.output_contours is not None:
            cv2.imwrite(out_filename, self.output_contours)

    def calc(self, input):
        if input is None:
            return None
        self.output_image, self.output_contours = input
        return None


class PaintByNumbersUI():
    def __init__(self):
        self.root = T.Tk()
        self.root.title("Paint by numbers")
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        for i in range(6):
            self.root.rowconfigure(i, weight=1)

        self.pipeline = [
            LoadImageFrame(self, 0),
            ResizeImageFrame(self, 1),
            PreprocessingFrame(self, 2),
            NumColorsFrame(self, 3),
            SmoothFrame(self, 4),
            ContoursFrame(self, 5),
            SaveFrame(self, 6),
        ]
        self.p = ttk.Progressbar(self.root, orient=T.HORIZONTAL, length=200, mode='determinate', maximum=7)
        self.p.grid(column=0, row=7, sticky=(T.N, T.W, T.E, T.S))

        self.image_original = ttk.Button(self.root, image=None, command=self.update_original)
        self.image_original.grid(column=1, row=0, sticky=_tor)

        self.image_resized = ttk.Button(self.root, image=None, command=self.update_resized)
        self.image_resized.grid(column=1, row=1, sticky=_tor)

        self.image_preprocessed = ttk.Button(self.root, image=None, command=self.update_preprocessed)
        self.image_preprocessed.grid(column=1, row=2, sticky=_tor)

        self.image_numcolors = ttk.Button(self.root, image=None, command=self.update_numcolors)
        self.image_numcolors.grid(column=1, row=3, sticky=_tor)

        self.image_smooth = ttk.Button(self.root, image=None, command=self.update_smooth)
        self.image_smooth.grid(column=1, row=4, sticky=_tor)

        self.image_output = ttk.Button(self.root, image=None, command=self.update_output)
        self.image_output.grid(column=1, row=5, sticky=_tor)

        self.image_contour = ttk.Button(self.root, image=None, command=self.update_contour)
        self.image_contour.grid(column=1, row=6, sticky=_tor)

    def update_original(self, show=True):
        image = self.pipeline[0].cached_output
        if image is None:
            return
        if show:
            show_image(image, "original")
        image = cv2.cvtColor(cv2.resize(image, (50, 50)), cv2.COLOR_BGR2RGB)
        t_image = ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
        self.image_original.configure(image=t_image)
        self.image_original.image = t_image

    def update_resized(self, show=True):
        image = self.pipeline[1].cached_output
        if image is None:
            return
        if show:
            show_image(image, "resized")
        image = cv2.cvtColor(cv2.resize(image, (50, 50)), cv2.COLOR_BGR2RGB)
        t_image = ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
        self.image_resized.configure(image=t_image)
        self.image_resized.image = t_image

    def update_preprocessed(self, show=True):
        image = self.pipeline[2].cached_output
        if image is None:
            return
        if show:
            show_image(image, "denoised")
        image = cv2.cvtColor(cv2.resize(image, (50, 50)), cv2.COLOR_BGR2RGB)
        t_image = ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
        self.image_preprocessed.configure(image=t_image)
        self.image_preprocessed.image = t_image

    def update_numcolors(self, show=True):
        image, labels, centers = self.pipeline[3].cached_output
        if image is None:
            return
        image = centers[labels]
        if show:
            show_image(image, "clustered")
        image = cv2.cvtColor(cv2.resize(image, (50, 50)), cv2.COLOR_BGR2RGB)
        t_image = ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
        self.image_numcolors.configure(image=t_image)
        self.image_numcolors.image = t_image

    def update_smooth(self, show=True):
        image, labels, centers = self.pipeline[4].cached_output
        if image is None:
            return
        image = centers[labels]
        if show:
            show_image(image, "smoothed")
        image = cv2.cvtColor(cv2.resize(image, (50, 50)), cv2.COLOR_BGR2RGB)
        t_image = ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
        self.image_smooth.configure(image=t_image)
        self.image_smooth.image = t_image

    def update_output(self, show=True):
        image = self.pipeline[5].cached_output[0]
        if image is None:
            return
        if show:
            show_image(image, "output")
        image = cv2.cvtColor(cv2.resize(image, (50, 50)), cv2.COLOR_BGR2RGB)
        t_image = ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
        self.image_output.configure(image=t_image)
        self.image_output.image = t_image

    def update_contour(self, show=True):
        image = self.pipeline[5].cached_output[1]
        if image is None:
            return
        if show:
            show_image(image, "contour")
        image = cv2.cvtColor(cv2.resize(image, (50, 50)), cv2.COLOR_BGR2RGB)
        t_image = ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
        self.image_contour.configure(image=t_image)
        self.image_contour.image = t_image

    def calc(self):
        image = None
        self.p.setvar("0")
        for step in self.pipeline:
            image = step.calc(image)
            self.p.step()
            self.root.update()

        # extremely ugly code, like everything else in this file
        self.update_original(show=False)
        self.update_resized(show=False)
        self.update_preprocessed(show=False)
        self.update_numcolors(show=False)
        self.update_smooth(show=False)
        self.update_output(show=False)
        self.update_contour(show=False)
        return image


    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    random.seed(239)
    np.random.seed(239)
    PaintByNumbersUI().run()
