import os
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QTextEdit, QMessageBox, QSpinBox, QGroupBox,
    QFrame
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.backend import (
    load_xyz, load_cmyk_10nm, load_pink_interpolated,
    optimize_mix, ref2lab,
)


def lab_to_rgb(L, a, b):
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def finv(t):
        return t**3 if t > 6.0/29.0 else 3.0 * (6.0/29.0)**2 * (t - 4.0/29.0)

    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X = Xn * finv(fx)
    Y = Yn * finv(fy)
    Z = Zn * finv(fz)

    r_lin = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_lin = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def gamma(u):
        u = max(0.0, min(1.0, u))
        return 12.92 * u if u <= 0.0031308 else 1.055 * u**(1.0/2.4) - 0.055

    R = int(round(gamma(r_lin) * 255))
    G = int(round(gamma(g_lin) * 255))
    B = int(round(gamma(b_lin) * 255))
    return max(0, min(255, R)), max(0, min(255, G)), max(0, min(255, B))


class ColorSwatch(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(80, 50)
        self.setFrameShape(QFrame.Box)
        self.setStyleSheet("background-color: #808080;")

    def set_color(self, r, g, b):
        self.setStyleSheet(f"background-color: rgb({r},{g},{b});")


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_spectra(self, wavelengths, target, predicted):
        self.ax.clear()
        self.ax.plot(wavelengths, target, label="Target")
        self.ax.plot(wavelengths, predicted, label="Predicted")
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Reflectance")
        self.ax.legend()
        self.ax.grid(True)
        self.draw()


class MainWindow(QWidget):
    def __init__(self, xyz_path, cmyk_path, pink_path=None):
        super().__init__()
        self.setWindowTitle("SpectOptiBlend (Desktop)")

        self.xyz = load_xyz(xyz_path)

        self.wavelengths, self.cmyk = load_cmyk_10nm(cmyk_path)

        self.all_spectra = dict(self.cmyk)
        if pink_path and os.path.exists(pink_path):
            pink_wl, pink_sp = load_pink_interpolated(pink_path)
            if np.allclose(pink_wl, self.wavelengths):
                self.all_spectra.update(pink_sp)

        main_layout = QHBoxLayout(self)

        left = QVBoxLayout()
        main_layout.addLayout(left, 1)

        left.addWidget(QLabel("Target color (the color you want to make):"))
        self.target_combo = QComboBox()
        for name in self.all_spectra.keys():
            self.target_combo.addItem(name)
        idx = self.target_combo.findText("Purple")
        if idx >= 0:
            self.target_combo.setCurrentIndex(idx)
        left.addWidget(self.target_combo)

        left.addWidget(QLabel("Ingredient colors (the waste plastics you have):"))
        self.ingredients_list = QListWidget()
        self.ingredients_list.setSelectionMode(QListWidget.NoSelection)
        self._populate_ingredients()
        left.addWidget(self.ingredients_list)

        box = QGroupBox("Settings")
        box_layout = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Solver:"))
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["Nelder-Mead", "SLSQP", "L-BFGS-B"])
        row.addWidget(self.solver_combo)
        box_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Total grams:"))
        self.total_grams = QSpinBox()
        self.total_grams.setRange(1, 5000)
        self.total_grams.setValue(200)
        row.addWidget(self.total_grams)
        box_layout.addLayout(row)

        self.run_btn = QPushButton("Optimize")
        self.run_btn.clicked.connect(self.on_optimize)
        box_layout.addWidget(self.run_btn)

        left.addWidget(box)

        # Results
        left.addWidget(QLabel("Results:"))
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 10))
        left.addWidget(self.output, 1)

        # Right side: plot + color swatches
        right = QVBoxLayout()
        main_layout.addLayout(right, 2)

        right.addWidget(QLabel("Target vs Predicted Spectrum"))
        self.plot = PlotCanvas(self)
        right.addWidget(self.plot, 1)

        # Color comparison swatches
        swatch_group = QGroupBox("Color Comparison")
        swatch_layout = QHBoxLayout(swatch_group)

        target_col = QVBoxLayout()
        target_col.addWidget(QLabel("Target"))
        self.target_swatch = ColorSwatch()
        target_col.addWidget(self.target_swatch)
        self.target_lab_label = QLabel("L*=—  a*=—  b*=—")
        self.target_lab_label.setFont(QFont("Consolas", 9))
        target_col.addWidget(self.target_lab_label)
        swatch_layout.addLayout(target_col)

        pred_col = QVBoxLayout()
        pred_col.addWidget(QLabel("Predicted Mix"))
        self.predicted_swatch = ColorSwatch()
        pred_col.addWidget(self.predicted_swatch)
        self.predicted_lab_label = QLabel("L*=—  a*=—  b*=—")
        self.predicted_lab_label.setFont(QFont("Consolas", 9))
        pred_col.addWidget(self.predicted_lab_label)
        swatch_layout.addLayout(pred_col)

        right.addWidget(swatch_group)

    def _populate_ingredients(self):
        self.ingredients_list.clear()
        for name in self.all_spectra.keys():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.ingredients_list.addItem(item)

    def get_checked_ingredients(self):
        names = []
        for i in range(self.ingredients_list.count()):
            item = self.ingredients_list.item(i)
            if item.checkState() == Qt.Checked:
                names.append(item.text())
        return names

    def on_optimize(self):
        solver = self.solver_combo.currentText()

        target_name = self.target_combo.currentText()
        target = self.all_spectra[target_name]

        ingredient_names = [n for n in self.get_checked_ingredients() if n != target_name]

        if len(ingredient_names) < 2:
            QMessageBox.warning(self, "Not enough ingredients",
                                "Check at least 2 ingredient colors (excluding the target).")
            return

        ingredients = [self.all_spectra[n] for n in ingredient_names]

        result = optimize_mix(
            target_spectrum=target,
            ingredient_spectra=ingredients,
            xyz=self.xyz,
            wavelengths=self.wavelengths,
            solver=solver,
            mode="paper",
            weights_mode="notebook",
        )

        mix = np.array(result["mix"], dtype=float)
        predicted = np.array(result["predicted_spectrum"], dtype=float)
        total_g = float(self.total_grams.value())

        target_lab = ref2lab(target, self.xyz)
        pred_lab = ref2lab(predicted, self.xyz)

        lines = []
        lines.append(f"Target: {target_name}")
        lines.append(f"Solver: {solver}")
        if not result["success"]:
            lines.append(f"NOTE:   Solver reported: {result['message']}")
            lines.append("        (Results may still be usable — check Delta E.)")
        lines.append("")
        lines.append(f"Weighted RMS:  {result['rms']:.6f}")
        lines.append(f"Delta E 2000:  {result['deltaE2000']:.5f}")
        lines.append("")
        lines.append(f"  Target  Lab: L*={target_lab[0]:6.2f}  a*={target_lab[1]:6.2f}  b*={target_lab[2]:6.2f}")
        lines.append(f"  Predict Lab: L*={pred_lab[0]:6.2f}  a*={pred_lab[1]:6.2f}  b*={pred_lab[2]:6.2f}")
        lines.append("")
        lines.append(f"Mix  ({total_g:.0f} g total):")
        lines.append(f"{'Color':<15} {'Fraction':>10} {'Grams':>10}")
        lines.append("-" * 37)
        for name, frac in zip(ingredient_names, mix):
            g = total_g * frac
            lines.append(f"{name:<15} {frac:>10.4f} {g:>10.2f}")

        self.output.setPlainText("\n".join(lines))

        self.plot.plot_spectra(self.wavelengths, target, predicted)

        tr, tg, tb = lab_to_rgb(target_lab[0], target_lab[1], target_lab[2])
        pr, pg, pb = lab_to_rgb(pred_lab[0], pred_lab[1], pred_lab[2])

        self.target_swatch.set_color(tr, tg, tb)
        self.predicted_swatch.set_color(pr, pg, pb)

        self.target_lab_label.setText(f"L*={target_lab[0]:.2f}  a*={target_lab[1]:.2f}  b*={target_lab[2]:.2f}")
        self.predicted_lab_label.setText(f"L*={pred_lab[0]:.2f}  a*={pred_lab[1]:.2f}  b*={pred_lab[2]:.2f}")
