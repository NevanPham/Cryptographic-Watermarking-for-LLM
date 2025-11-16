from app_GUI import Ui_MainWindow
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QTextBrowser, QComboBox, QWidget, QLabel, QStackedWidget, QPlainTextEdit, QSlider, QMessageBox, QFileDialog, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np
from core import get_model, generate_unwatermarked, parse_final_output, get_paraphraser, evaluate_model
from src.watermark import ZeroBitWatermarker
import sys, os
import tqdm

def display_plot(widget: QWidget, fig: Figure):
    # clear layout widget
    layout = widget.layout()
    if layout:
        # remove old plots
        for i in reversed(range(layout.count())):
            old_widget = layout.itemAt(i).widget()
            if old_widget:
                old_widget.setParent(None)
    else:
        layout = QVBoxLayout()
        widget.setLayout(layout)

    canvas = FigureCanvas(fig)
    layout.addWidget(canvas)
    if widget.layout() is None:
        widget.setLayout(QVBoxLayout())
    matplotlib.rcParams.update({'font.size': 5})
    canvas.draw()
    QApplication.processEvents()
    

def plot_robustness_static(widget, df, output_dir, params, z_threshold):
    """Creates a box plot for a single, static parameter run."""
    plt = Figure(figsize=(12, 7))
    ax = plt.add_subplot(111)
    title_suffix = f"(Delta={params['delta']}, HC={params['hashing_context']}, ET={params['entropy_threshold']})"

    robustness_data = df[(df['type'] == 'watermarked')]
    sns.boxplot(x='perturbation', y='z_score', data=robustness_data)

    plt.title(f'Robustness to Perturbations {title_suffix}')
    plt.xlabel('Perturbation Type')
    plt.ylabel('Z-Score')
    plt.axhline(y=z_threshold, color='red', linestyle='--', label=f'Threshold (z={z_threshold})')
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'robustness_boxplot{title_suffix.replace(" ", "")}.png')
    plt.savefig(plot_path)
    print(f"📊 Static robustness plot saved.")
    display_plot(widget, plt)
    #plt.close()

def plot_completeness_vs_soundness(widget, df, output_dir, params, z_threshold):
    """
    Plots the full distribution and a zoomed-in view of z-scores for a specific parameter set.
    """
    title_suffix = f"(Delta={params['delta']}, HC={params['hashing_context']}, ET={params['entropy_threshold']})"
    file_suffix = f"_delta_{params['delta']}_hc_{params['hashing_context']}_et_{params['entropy_threshold']}"

    # Filter data for the specific plot
    wm_scores = df[
        (df['type'] == 'watermarked') &
        (df['perturbation'] == 'clean') &
        (df['delta'] == params['delta']) &
        (df['hashing_context'] == params['hashing_context']) &
        (df['entropy_threshold'] == params['entropy_threshold'])
    ]['z_score']
    unwm_scores = df[df['type'] == 'unwatermarked']['z_score']

    if wm_scores.empty:
        print(f"Warning: No 'clean' watermarked data found for parameters: {params}. Skipping this completeness plot.")
        return
    if unwm_scores.empty:
        print("Warning: No unwatermarked data found. Skipping all completeness plots.")
        return


    # --- 1. Generate the Full Distribution Plot (Improved) ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    q_low, q_high = wm_scores.quantile([0.01, 0.99])  

    # keep only the middle 98% of values
    wm_scores_filtered = wm_scores[(wm_scores >= q_low) & (wm_scores <= q_high)]

    sns.histplot(wm_scores_filtered, color="red", label='Watermarked (Completeness)', stat="density", bins=50, ax=ax1, alpha=0.7)
    ax1_twin = ax1.twinx()
    sns.histplot(unwm_scores, color="blue", label='Unwatermarked (Soundness)', ax=ax1_twin, alpha=0.5)

    ax1.set_title(f'Full Z-Score Distribution {title_suffix}')
    ax1.set_xlabel('Z-Score')
    ax1.set_ylabel('Watermarked Count')
    ax1_twin.set_ylabel('Unwatermarked Density')
    ax1.axvline(x=z_threshold, color='black', linestyle='--', label=f'Threshold (z={z_threshold})')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lines + lines2, labels + labels2, loc='upper right')

    x_min = min(unwm_scores.min(), wm_scores.min()) - 2
    x_max = max(wm_scores.quantile(0.99), unwm_scores.max()) + 5
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    full_plot_path = os.path.join(output_dir, f'completeness_dist{file_suffix}.png')
    fig1.savefig(full_plot_path)
    #plt.close(fig1)

    # --- 2. Generate the Zoomed-In Plot ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    x_zoom_range = (z_threshold - 5, z_threshold + 5)
    bins_for_zoom = max(10, int((x_zoom_range[1] - x_zoom_range[0]) * 4))

    sns.histplot(unwm_scores, color="blue", label='Unwatermarked (Soundness)', stat="count",
                    bins=bins_for_zoom, ax=ax2, alpha=0.7, edgecolor='black', binrange=x_zoom_range, kde=False)
    sns.histplot(wm_scores, color="red", label=f'Watermarked (Completeness)', stat="count",
                    bins=bins_for_zoom, ax=ax2, alpha=0.7, edgecolor='black', binrange=x_zoom_range, kde=False)

    ax2.set_title(f'Zoomed View @ Threshold {title_suffix}')
    ax2.set_xlabel('Z-Score')
    ax2.set_ylabel('Count')
    ax2.axvline(x=z_threshold, color='gray', linestyle='--', label=f'Threshold (z={z_threshold})')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_xlim(*x_zoom_range)

    zoomed_plot_path = os.path.join(output_dir, f'completeness_dist{file_suffix}_ZOOMED.png')
    fig2.savefig(zoomed_plot_path)
    print(f"📊 Completeness/Soundness plots saved for {title_suffix}")
    display_plot(widget, Figure(fig2))
    #plt.close(fig2)

def plot_robustness_sweep(widget, df, output_dir, sweep_var, z_threshold):
    """
    Plots the detection rate on PERTURBED watermarked text vs. the sweep parameter
    using a grouped bar chart for clearer comparison, with improved spacing and grid.
    """
    from matplotlib.figure import Figure
    import seaborn as sns

    # Create a Figure
    fig = Figure(figsize=(5, 3), dpi=60)
    ax = fig.add_subplot(111)

    # Exclude 'clean' texts to focus only on robustness against attacks
    robustness_df = df[(df['type'] == 'watermarked') & (df['perturbation'] != 'clean')].copy()

    if robustness_df.empty:
        print("No perturbed text data found to generate robustness plot.")
        fig.clf()
        return

    # Calculate detection success rate as a percentage
    robustness_df['Detection Rate (%)'] = (robustness_df['z_score'] > z_threshold) * 100

    # --- Create the Grouped Bar Chart with improved aesthetics ---
    sns.barplot(
        data=robustness_df,
        x=sweep_var,
        y='Detection Rate (%)',
        hue='perturbation',
        errorbar='sd',
        capsize=.08,
        palette='deep',
        edgecolor='black',
        linewidth=0.5,
        ax=ax  # specify axes explicitly
    )

    ax.set_title(f'Robustness vs. {sweep_var.replace("_", " ").title()}')
    ax.set_xlabel(f'{sweep_var.replace("_", " ").title()}')
    ax.set_ylabel(f'Detection Rate (%) @ Z > {z_threshold}')
    ax.set_ylim(0, 105)

    # Adjust legend position
    ax.legend(title='Attack Type', loc='lower left', bbox_to_anchor=(0.01, 0.01))

    # Add both X and Y grid lines
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)

    fig.tight_layout()

    # Save PNG
    plot_path = os.path.join(output_dir, f'robustness_sweep_vs_{sweep_var}.png')
    fig.savefig(plot_path)
    print(f"📊 Robustness sweep plot saved to {plot_path}")

    # Display in the QWidget
    display_plot(widget, fig)

    # Clear figure to free memory
    #fig.clf()


def analyse(eval_dir, z_threshold, widget1, widget2):
    results_path = os.path.join(eval_dir, 'analysis_results.json')
    if not os.path.exists(results_path):
        print(f"Error: Could not find 'analysis_results.json' in '{eval_dir}'")
        return

    print(f"Loading results from {results_path}")
    df = pd.read_json(results_path)

    df = df[df['z_score'].apply(lambda x: isinstance(x, (int, float)))]
    p_categories = ['clean', 'substitute 30 percent', 'delete start 20 percent', 'delete middle 20 percent', 'delete end 20 percent']
    if 'perturbation' in df.columns:
        df['perturbation'] = pd.Categorical(
            df['perturbation'],
            categories=[c for c in p_categories if c in df['perturbation'].unique()],
            ordered=True
        )
    else:
        print("Warning: 'perturbation' column not found in dataframe — skipping categorization.")


    # --- Identify the swept parameter ---
    sweep_var = None
    param_cols = [col for col in ['delta', 'hashing_context', 'entropy_threshold'] if col in df.columns]
    unique_counts = {col: df[col].nunique() for col in param_cols}
    for var, count in unique_counts.items():
        if count > 1:
            sweep_var = var
            break

    summary_lines = []

    # --- Branch logic based on whether a sweep was detected ---
    if sweep_var:
        print(f"\n--- Detected sweep of parameter: '{sweep_var}' ---")
        sweep_values = sorted(df[df[sweep_var].notna()][sweep_var].unique())

        # --- Generate Plots for the Sweep ---
        print(f"Generating plots for each value: {sweep_values}")
        for value in sweep_values:
            # Construct a params dict for the current value in the sweep
            static_params = {
                'delta': df['delta'].iloc[0] if 'delta' != sweep_var else value,
                'hashing_context': df['hashing_context'].iloc[0] if 'hashing_context' != sweep_var else value,
                'entropy_threshold': df['entropy_threshold'].iloc[0] if 'entropy_threshold' != sweep_var else value
            }
            plot_completeness_vs_soundness(widget1, df, eval_dir, static_params, z_threshold)

        plot_robustness_sweep(widget2, df, eval_dir, sweep_var, z_threshold)

        # --- Quantitative Analysis for the Sweep ---
        summary_lines.append("--- Quantitative Analysis Summary (Sweep) ---")
        unwatermarked_df = df[df['type'] == 'unwatermarked']
        false_positives = len(unwatermarked_df[unwatermarked_df['z_score'] > z_threshold])
        fp_rate = (false_positives / len(unwatermarked_df)) * 100 if len(unwatermarked_df) > 0 else 0
        summary_lines.append(f"\nSoundness @ Z-Threshold={z_threshold}:")
        summary_lines.append(f"  - False Positive Rate: {fp_rate:.2f}% ({false_positives}/{len(unwatermarked_df)})")

        for value in sweep_values:
            summary_lines.append(f"\nMetrics for {sweep_var.replace('_', ' ')} = {value} @ Z-Threshold={z_threshold}:")
            for p_type in df['perturbation'].cat.categories:
                subset = df[(df[sweep_var] == value) & (df['perturbation'] == p_type)]
                if subset.empty: continue
                total, detected = len(subset), len(subset[subset['z_score'] > z_threshold])
                detection_rate = (detected / total) * 100 if total > 0 else 0
                metric_name = "Completeness" if p_type == 'clean' else f"Robustness ({p_type.replace('_', ' ')})"
                summary_lines.append(f"  - {metric_name}: {detection_rate:.2f}% detected ({detected}/{total})")

    else: # Handle the static parameter case
        print("\n--- Detected a single parameter run ---")

        if not df[df['type'] == 'watermarked'].empty:
            # Extract the single set of parameters from the first watermarked entry
            static_params = df[df['type'] == 'watermarked'].iloc[0][param_cols].to_dict()

            # --- Generate Plots for the Static Run ---
            plot_completeness_vs_soundness(widget1, df, eval_dir, static_params, z_threshold)
            plot_robustness_static(widget2, df, eval_dir, static_params, z_threshold) # Use the box plot for static runs

            # --- Quantitative Analysis for the Static Run ---
            summary_lines.append("--- Quantitative Analysis Summary (Static) ---")
            unwatermarked_df = df[df['type'] == 'unwatermarked']
            false_positives = len(unwatermarked_df[unwatermarked_df['z_score'] > z_threshold])
            fp_rate = (false_positives / len(unwatermarked_df)) * 100 if len(unwatermarked_df) > 0 else 0
            summary_lines.append(f"\nSoundness @ Z-Threshold={z_threshold}:")
            summary_lines.append(f"  - False Positive Rate: {fp_rate:.2f}% ({false_positives}/{len(unwatermarked_df)})")

            summary_lines.append(f"\nMetrics for {static_params} @ Z-Threshold={z_threshold}:")
            for p_type in df['perturbation'].cat.categories:
                subset = df[df['perturbation'] == p_type]
                if subset.empty: continue
                total, detected = len(subset), len(subset[subset['z_score'] > z_threshold])
                detection_rate = (detected / total) * 100 if total > 0 else 0
                metric_name = "Completeness" if p_type == 'clean' else f"Robustness ({p_type.replace('_', ' ')})"
                summary_lines.append(f"  - {metric_name}: {detection_rate:.2f}% detected ({detected}/{total})")
        else:
            print("No watermarked data found to generate plots or summary.")

    # --- Print to console and save to file ---
    if summary_lines:
        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)
        summary_path = os.path.join(eval_dir, 'summary_analysis.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        print(f"\n📝 Quantitative summary saved to {summary_path}")


class GUIAppWindow(QMainWindow, Ui_MainWindow):
    #init app window object
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("AI GenTools")
        self.key_file_path = ""
        self.prompts_file_path = ""
        self.analyse_path = ""
        self.currentUser = ""

        #setup widgets to display matplotlib/seaborn graphs


    #init input controls

        #default page is generate
        self.generateButton.setChecked(True)

        #when click on each of the page buttons, trigger switch page methods
        self.generateButton.clicked.connect(self.switch_to_generate_page)
        self.detectButton.clicked.connect(self.switch_to_detect_page)
        self.evaluateButton.clicked.connect(self.switch_to_evaluate_page)

        # set placeholder text for prompt input and detect input
        self.promptBox.setPlaceholderText("Enter prompt here...")
        self.detectBox.setPlaceholderText("Paste content here...")

        # add model options
        self.genModelComboBox.addItems(["gpt2", "gpt-oss-20b", "gpt-oss-120b"])
        self.evalModelComboBox.addItems(["gpt2", "gpt-oss-20b", "gpt-oss-120b"])
        self.detectModelComboBox.addItems(["gpt2", "gpt-oss-20b", "gpt-oss-120b"])

        # close login/account windows until account button is clicked
        self.loginWidget.setVisible(False)
        self.accountWidget.setVisible(False)

        self.accountButton.clicked.connect(self.toggle_account_windows)

        # close advanced options windows, only open when button clicked
        self.detectAdvancedWidget.setVisible(False)
        self.genAdvancedWidget.setVisible(False)

        self.genAdvancedButton.clicked.connect(self.toggle_gen_advanced_window)
        self.detectAdvancedButton.clicked.connect(self.toggle_detect_advanced_window)

    # ~~ init all sliders to match label value with slider value ~~

    # ... generate page section ...
        
        # generate hashing context
        self.genHashingSlider.setMaximum(500)
        self.genHashingSlider.setValue(5)
        self.genHashingDisplayValue.setText(str(self.genHashingSlider.value()))
        self.genHashingSlider.valueChanged.connect(
            lambda v: self.genHashingDisplayValue.setText(str(v))
        )

        #generate max new tokens - steps between avaiable options
        self.token_steps = [512, 1024, 2048, 4096] # access from array for allowed options

        self.genMaxNewTokensSlider.setMinimum(0)
        self.genMaxNewTokensSlider.setMaximum(len(self.token_steps) - 1)
        self.genMaxNewTokensSlider.setSingleStep(1)
        self.genMaxNewTokensSlider.setValue(2)  # default to 2048 (index 2)
        self.genMaxDisplayValue.setText(str(self.token_steps[self.genMaxNewTokensSlider.value()]))
        self.genMaxNewTokensSlider.valueChanged.connect(
            lambda i: self.genMaxDisplayValue.setText(str(self.token_steps[i]))
        )
        # update max new tokens slider depending on model
        self.genModelComboBox.currentIndexChanged.connect(self.update_max_new_tokens_slider)
        self.update_max_new_tokens_slider()

        #generate delta - accommodates float
        self.genDeltaSlider.setMinimum(10)   # 1.0 * 10
        self.genDeltaSlider.setMaximum(50)   # 5.0 * 10
        self.genDeltaSlider.setSingleStep(1) # step = 0.1
        self.genDeltaSlider.setValue(25)     # 2.5 * 10

        self.genDeltaDisplayValue.setText("2.5")
        self.genDeltaSlider.valueChanged.connect(
            lambda v: self.genDeltaDisplayValue.setText(f"{v / 10:.1f}")
        )

        #generate entropy - also accommodates float
        self.genEntropySlider.setMinimum(10)   # 1.0 * 10
        self.genEntropySlider.setMaximum(60)   # 6.0 * 10
        self.genEntropySlider.setSingleStep(1)
        self.genEntropySlider.setValue(40)     # 4.0 * 10

        self.genEntropyDisplayValue.setText("4.0")
        self.genEntropySlider.valueChanged.connect(
            lambda v: self.genEntropyDisplayValue.setText(f"{v / 10:.1f}")
        )

    # ... detect page section ... GEMMA HI HERE EDIT HERE THIS SECTION HI 

        # detect hashing context 
        self.detectHashingSlider.setValue(5)
        self.detectHashingDisplayValue.setText(str(self.detectHashingSlider.value()))
        self.detectHashingSlider.valueChanged.connect(
            lambda v: self.detectHashingDisplayValue.setText(str(v))
        )

        # detect Z threshold
        self.detectZSlider.setMinimum(10)   # 1.0 * 10
        self.detectZSlider.setMaximum(50)   # 5.0 * 10
        self.detectZSlider.setSingleStep(1) # step = 0.1
        self.detectZSlider.setValue(40)     # 4 * 10
 
        self.detectZDisplayValue.setText("4")
        self.detectZSlider.valueChanged.connect(
            lambda v: self.detectZDisplayValue.setText(f"{v / 10:.1f}")
        )

        # entropy threshold 
        self.detectEntropySlider.setMinimum(10)   # 1.0 * 10
        self.detectEntropySlider.setMaximum(60)   # 6.0 * 10
        self.detectEntropySlider.setSingleStep(1)
        self.detectEntropySlider.setValue(40)     # 4.0 * 10
 
        self.detectEntropyDisplayValue.setText("4.0")
        self.detectEntropySlider.valueChanged.connect(
            lambda v: self.detectEntropyDisplayValue.setText(f"{v / 10:.1f}")
        )

    # ... evaluate page section ...

        # eval delta - with float
        self.evalDeltaSlider.setMinimum(10)   # 1.0 * 10
        self.evalDeltaSlider.setMaximum(50)   # 5.0 * 10
        self.evalDeltaSlider.setSingleStep(1) # step = 0.1
        self.evalDeltaSlider.setValue(25)     # 2.5 * 10

        self.evalDeltaDisplayValue.setText("2.5")
        self.evalDeltaSlider.valueChanged.connect(
            lambda v: self.evalDeltaDisplayValue.setText(f"{v / 10:.1f}")
        )

        # eval Z threshold
        self.evalZSlider.setMinimum(10)   # 1.0 * 10
        self.evalZSlider.setMaximum(50)   # 5.0 * 10
        self.evalZSlider.setSingleStep(1) # step = 0.1
        self.evalZSlider.setValue(40)     # 4 * 10
 
        self.evalZDisplayValue.setText("4")
        self.evalZSlider.valueChanged.connect(
            lambda v: self.evalZDisplayValue.setText(f"{v / 10:.1f}")
        )

        # init check boxes
        self.entropy_checkboxes = [self.evalEnt1Check, self.evalEnt2Check, self.evalEnt3Check, self.evalEnt4Check, self.evalEnt5Check, self.evalEnt6Check]
        self.hashing_checkboxes = [self.evalHash3Check, self.evalHash4Check, self.evalHash5Check, self.evalHash6Check, self.evalHash7Check, self.evalHash8Check]

        # eval entropy threshold check box
        self.evalEnt4Check.setChecked(True)

        # eval hashing context 
        self.evalHash5Check.setChecked(True)

        # eval max new tokens
        self.evalMaxNewTokensSlider.setMinimum(0)
        self.evalMaxNewTokensSlider.setMaximum(len(self.token_steps) - 1)
        self.evalMaxNewTokensSlider.setSingleStep(1)
        self.evalMaxNewTokensSlider.setValue(2)  # default to 2048 (index 2 of token steps)
        self.evalMaxDisplayValue.setText(str(self.token_steps[self.evalMaxNewTokensSlider.value()]))
        self.evalMaxNewTokensSlider.valueChanged.connect(
            lambda i: self.evalMaxDisplayValue.setText(str(self.token_steps[i]))
        )
        # update max new tokens slider depending on model
        self.evalModelComboBox.currentIndexChanged.connect(self.update_max_new_tokens_slider)
        self.update_eval_max_new_tokens_slider()

    # ~~ buttons and functions ~~
        
    # ... generate page section ...

        # upload key file location button clicked
        self.generateUploadFileButton.clicked.connect(self.choose_key_file_location)

        # generate unwatermarked text when button is clicked
        self.genButton.clicked.connect(self.generate_text)

    # ... detect page section ...

        # set results text label to blank on run and hide user display
        self.label_12.setVisible(False)
        self.resultsDisplay.setText("")
        self.label_15.setVisible(False)
        self.userDisplay.setVisible(False)

        # upload key file location button clicked
        self.detectUploadFileButton.clicked.connect(self.choose_key_file_location)

        # check pasted content when button is clicked
        self.checkButton.clicked.connect(self.detect_watermark)

    # ... evaluate page section ...

        # upload prompt file location button clicked
        self.evalUploadFileButton.clicked.connect(self.choose_prompts_file_location)

        # download location button clicked
        self.evalDownloadFileButton.clicked.connect(self.choose_analyse_folder_location)

        # call evaluate function when evaluate button is pressed
        self.evalButton.clicked.connect(self.evaluate)

        # set chosen sweep value
        # default disable multiple entropy selections
        for cb in self.entropy_checkboxes:
            cb.setChecked(False)
            cb.setEnabled(True)
            cb.setAutoExclusive(True)
        self.evalEnt4Check.setChecked(True)

       # default disable multiple hashing selections
        for cb in self.hashing_checkboxes:
            cb.setChecked(False)
            cb.setEnabled(True)
            cb.setAutoExclusive(True)
        self.evalHash5Check.setChecked(True)

        self.evalSweepComboBox.currentTextChanged.connect(self.update_sweep_checkbox_behaviour)

        # download evaluate data as raw json

        # download evaluate data pngs


    #switch between pages from the buttons

    #generate page - main page
    def switch_to_generate_page(self):
        self.stackedWidget.setCurrentIndex(0)

        #detect page
    def switch_to_detect_page(self):
        self.stackedWidget.setCurrentIndex(1)

        #eval page
    def switch_to_evaluate_page(self):
        self.stackedWidget.setCurrentIndex(2)

    # checks if user is logged in or not, shows login or account page accordingly
    def toggle_account_windows(self):
        if self.currentUser != "":
            if self.accountWidget.isVisible():
                self.accountWidget.setVisible(False)
            else:
                self.accountWidget.setVisible(True)
        else:
            if self.loginWidget.isVisible():
                self.loginWidget.setVisible(False)
            else:
                self.loginWidget.setVisible(True)

    def toggle_gen_advanced_window(self):
        if self.genAdvancedWidget.isVisible():
            self.genAdvancedWidget.setVisible(False)
        else:
            self.genAdvancedWidget.setVisible(True)

    def toggle_detect_advanced_window(self):
        if self.detectAdvancedWidget.isVisible():
            self.detectAdvancedWidget.setVisible(False)
        else:
            self.detectAdvancedWidget.setVisible(True)

    # open file dialog when file buttons are pressed
    def choose_key_file_location(self):
        directory_path = QFileDialog.getExistingDirectory(self, "Select folder to save key file", "")
        if not directory_path:
            QMessageBox.information(self, "Cancelled", "No folder selected.")
            return
        self.key_file_path = directory_path + "/secret.key" # set location to save key file
        if self.key_file_path:
            folder_name = os.path.basename(directory_path)
            self.generateUploadFileButton.setText(str(folder_name))
            self.detectUploadFileButton.setText(str(folder_name))
        QApplication.processEvents()

    # open file dialog when prompts file location button is pressed on evaluate page
    def choose_prompts_file_location(self):
        file_path = QFileDialog.getOpenFileName(self, "Select prompts file", "","Text files (*.txt)")
        if not file_path:
            QMessageBox.information(self, "Cancelled", "No file selected.")
            return
        self.prompts_file_path = file_path[0] # set location of prompts file
        if self.prompts_file_path:
            print(file_path[0])
            file_name = os.path.basename(file_path[0])
            self.evalUploadFileButton.setText(str(file_name))
        QApplication.processEvents()

    # analyse folder location dialog
    def choose_analyse_folder_location(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select folder to save analysis results", "")
        if not folder_path:
            QMessageBox.information(self, "Cancelled", "No folder selected.")
            return
        self.analyse_path = folder_path + "/gui_app_evaluation_results" # set location to save  analysis results
        if self.analyse_path:
            print(self.analyse_path)
            folder_name = os.path.basename(folder_path)
            self.evalDownloadFileButton.setText(str(folder_name))
        QApplication.processEvents()

    def update_max_new_tokens_slider(self):
        # adjust max new tokens slider maximum valyue based on selected model - safe context sizes
        token_steps = self.token_steps
        model_name = self.genModelComboBox.currentText()

        # context limit sizes based on model
        context_limits = {
            "gpt2": 1024,         # 1024 max but mayeb switch back to 512 for local testing??
            "gpt-oss-20b": 4096,  
            "gpt-oss-120b": 4096, 
        }
        safe_limit = context_limits.get(model_name, max(token_steps))

        # find highest index in token_steps array that doesn't go over safe_limit for selected model
        max_index = 0
        for i, v in enumerate(token_steps):
            if v <= safe_limit:
                max_index = i

        self.genMaxNewTokensSlider.setMaximum(max_index)

        # if current slider value is out of range move it down
        if self.genMaxNewTokensSlider.value() > max_index:
            self.genMaxNewTokensSlider.setValue(max_index)

        # update displayed value
        self.genMaxDisplayValue.setText(str(token_steps[self.genMaxNewTokensSlider.value()]))

    def update_eval_max_new_tokens_slider(self):
        # adjust max new tokens slider maximum valyue based on selected model - safe context sizes
        token_steps = self.token_steps
        model_name = self.evalModelComboBox.currentText()

        # context limit sizes based on model
        context_limits = {
            "gpt2": 1024,         # 1024 max but mayeb switch back to 512 for local testing??
            "gpt-oss-20b": 4096,  
            "gpt-oss-120b": 4096, 
        }
        safe_limit = context_limits.get(model_name, max(token_steps))

        # find highest index in token_steps array that doesn't go over safe_limit for selected model
        max_index = 0
        for i, v in enumerate(token_steps):
            if v <= safe_limit:
                max_index = i

        self.evalMaxNewTokensSlider.setMaximum(max_index)

        # if current slider value is out of range move it down
        if self.evalMaxNewTokensSlider.value() > max_index:
            self.evalMaxNewTokensSlider.setValue(max_index)

        # update displayed value
        self.evalMaxDisplayValue.setText(str(token_steps[self.evalMaxNewTokensSlider.value()]))

    # eval page sweep value choice
    def update_sweep_checkbox_behaviour(self, selected_mode):
        if selected_mode == "None":
            # disable multiple entropy selections
            for cb in self.entropy_checkboxes:
                cb.setChecked(False)
                cb.setEnabled(True)
                cb.setAutoExclusive(True)
            self.evalEnt4Check.setChecked(True)

            # disable multiple hashing selections
            for cb in self.hashing_checkboxes:
                cb.setChecked(False)
                cb.setEnabled(True)
                cb.setAutoExclusive(True)
            self.evalHash5Check.setChecked(True)
        if selected_mode == "Entropy":
            # enable multiple entropy selections
            for cb in self.entropy_checkboxes:
                cb.setChecked(False)
                cb.setEnabled(True)
                cb.setAutoExclusive(False)
            self.evalEnt3Check.setChecked(True)
            self.evalEnt4Check.setChecked(True)

            # restrict hashing checkboxes to only one selected
            for cb in self.hashing_checkboxes:
                cb.setChecked(False)
                cb.setEnabled(True)
                cb.setAutoExclusive(True)
            self.evalHash5Check.setChecked(True)

        elif selected_mode == "Hashing Context":
            # enable multiple hashing selections
            for cb in self.hashing_checkboxes:
                cb.setChecked(False)
                cb.setEnabled(True)
                cb.setAutoExclusive(False)
            self.evalHash5Check.setChecked(True)
            self.evalHash6Check.setChecked(True)
            # restrict entropy checkboxes to only one selected
            for cb in self.entropy_checkboxes:
                cb.setChecked(False)
                cb.setEnabled(True)
                cb.setAutoExclusive(True)
            self.evalEnt4Check.setChecked(True)

    # ... generate page functions ...

    def generate_text(self):
        #init variables
        max_new_tokens = self.token_steps[self.genMaxNewTokensSlider.value()]

        prompt = str(self.promptBox.toPlainText().strip())
        # input validation - check there is a prompt entered otherwise warning mesasge
        if not prompt:
            QMessageBox.warning(self, "Missing Prompt", "Please enter a prompt before generating.")
            return

        # init other variables
        watermark = self.genWaterMarkCheckBox.isChecked()
        key_file = self.key_file_path
        gen_model = self.genModelComboBox.currentText()
        delta = self.genDeltaSlider.value() / 10
        entropy_threshold = self.genEntropySlider.value() / 10
        hashing_context = self.genHashingSlider.value()

        # begin process
        print(f"Loading model '{gen_model}'...")
        
        self.responseBox.setPlainText("⏳ Loading...") # loading placeholder message
        QApplication.processEvents()
        model = get_model(gen_model)

        #generate unwatermarked text
        if not watermark:
            final_text = generate_unwatermarked(model, prompt, max_new_tokens, gen_model)
            print("\n--- Generated Text ---")
            print(final_text)
            print("------------------------")
        #generate watermarked text
        else:
            watermarker = ZeroBitWatermarker(
                model=model, 
                delta=delta, 
                entropy_threshold=entropy_threshold, 
                hashing_context=hashing_context
            )
            secret_key = watermarker.keygen()
            
            print("\nEmbedding watermark...")
            # Unpack the text and the final parameters used
            raw_watermarked_text, final_params = watermarker.embed(secret_key, prompt, max_new_tokens)
            
            # Parse the raw output to get the final answer
            final_text = parse_final_output(raw_watermarked_text, gen_model)
            
            print("\n--- Final Watermarked Response ---")
            print(final_text)
            print("----------------------------------")
            print(f"(Final parameters used: {final_params})")
            
            with open(key_file, 'wb') as f:
                f.write(secret_key)
            print(f"Secret key saved to {key_file}")
        
        self.responseBox.setPlainText(f"Prompt: '{prompt}'\n'{final_text}'") # add final text to text display box
        QApplication.processEvents()

    # ... detect page functions ...
         
    def detect_watermark(self):
        self.label_12.setVisible(False)
        self.label_15.setVisible(False)
        self.userDisplay.setText("")
        self.userDisplay.setVisible(False)
        # init variables
        key_file = self.key_file_path 
        gen_model = self.detectModelComboBox.currentText()
        entropy_threshold = self.detectEntropySlider.value() / 10
        hashing_context = self.detectHashingSlider.value()
        z_threshold = 4
        detectedUser = "Ella Hunt" ## CHANGE THIS PLACEHOLDER LINE FOR MULTI USER, INCLUDE USER DETECTION ALSO
        print(f"Loading model '{gen_model}' for tokenizer...")
        model = get_model(gen_model)
        try:
            with open(key_file, 'rb') as f:
                secret_key = f.read()
            print(f"Loaded secret key from {key_file}")
            text_to_check = str(self.detectBox.toPlainText().strip())

        except FileNotFoundError as e:
            print(f"Error: Could not find file {e.filename}")
            return
            
        print("\nRunning detection algorithm...")

        pass_1_params = {
            'delta': 3.0,
            'hashing_context': hashing_context,
            'entropy_threshold': entropy_threshold
        }
        watermarker_pass1 = ZeroBitWatermarker(model=model, z_threshold=z_threshold, **pass_1_params)
        z_score, is_detected, block_count = watermarker_pass1.detect(secret_key, text_to_check)
        
        final_params = pass_1_params

        if block_count < 75 and not is_detected:
            print(f"Initial block count ({block_count}) is low. Running a more aggressive second pass...")
            
            pass_2_params = pass_1_params.copy()
            pass_2_params['entropy_threshold'] -= 2.0

            # Check if the new parameters are within the valid ranges
            if pass_2_params['entropy_threshold'] >= 1.0:
                watermarker_pass2 = ZeroBitWatermarker(model=model, z_threshold=z_threshold, **pass_2_params)
                # Overwrite the results with the second pass
                z_score, is_detected, block_count = watermarker_pass2.detect(secret_key, text_to_check)
                final_params = pass_2_params

        print("\n--- Detection Results ---")
        print(f"  Z-Score: {z_score:.4f}")
        print(f"  Threshold: {z_threshold}")
        print(f"  Detected: {'Yes' if is_detected else 'No'}")
        print(f"  Blocks Found: {block_count}")
        print(f"  Final Params Used: {final_params}")
        print("-------------------------")

        if is_detected:
            self.label_12.setVisible(True)
            self.resultsDisplay.setText(f"AI Watermark detected\nBlocks found: {block_count}")
            self.label_15.setVisible(True)
            self.userDisplay.setText(detectedUser)
            self.userDisplay.setVisible(True)
        else:
            self.label_12.setVisible(True)
            self.resultsDisplay.setText(f"AI Watermark not detected\nBlocks found: {block_count}")
            self.label_15.setVisible(False)
            self.userDisplay.setText("")
            self.userDisplay.setVisible(False)

    # ... evaluate page functions ...
    
    def evaluate(self):
        # init variables
        prompts = self.prompts_file_path
        max_new_tokens = self.token_steps[self.evalMaxNewTokensSlider.value()]
        eval_model = self.evalModelComboBox.currentText()
        delta = self.evalDeltaSlider.value() / 10
        # set list of entropy value or values
        entropy_thresholds = []
        if (self.evalEnt1Check.isChecked()):
            entropy_thresholds.append(1.0)
        if (self.evalEnt2Check.isChecked()):
            entropy_thresholds.append(2.0)
        if (self.evalEnt3Check.isChecked()):
            entropy_thresholds.append(3.0)
        if (self.evalEnt4Check.isChecked()):
            entropy_thresholds.append(4.0)
        if (self.evalEnt5Check.isChecked()):
            entropy_thresholds.append(5.0)
        if (self.evalEnt6Check.isChecked()):
            entropy_thresholds.append(6.0)
        entropy_threshold = entropy_thresholds[0]
        # set list of hashing context value or values
        hashing_contexts = []
        if (self.evalHash3Check.isChecked()):
            hashing_contexts.append(3)
        if (self.evalHash4Check.isChecked()):
            hashing_contexts.append(4)
        if (self.evalHash5Check.isChecked()):
            hashing_contexts.append(5)
        if (self.evalHash6Check.isChecked()):
            hashing_contexts.append(6)
        if (self.evalHash7Check.isChecked()):
            hashing_contexts.append(7)
        if (self.evalHash8Check.isChecked()):
            hashing_contexts.append(8)
        hashing_context = hashing_contexts[0]

        if len(hashing_contexts) == 1:
            hashing_contexts = []
        if len(entropy_thresholds) == 1:
            entropy_thresholds = []

        z_threshold = self.evalZSlider.value()
        directory = self.analyse_path
        deltas = []
        evaluate_model(prompts, max_new_tokens, eval_model, delta, entropy_threshold, hashing_context, z_threshold, directory, deltas, entropy_thresholds, hashing_contexts)
        analyse(directory, z_threshold, self.graph1, self.graph2)


app = QApplication(sys.argv)

window = GUIAppWindow()
window.show()
app.exec()
