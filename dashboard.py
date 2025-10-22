from doctest import testfile
import glob
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tempfile
import os
from ClassificationML.classification_model import *
from matplotlib.image import imread

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
except ImportError:
    TkinterDnD = None
    DND_FILES = None


def create_dashboard(data_out, figs, geo_map):
    """
    Create and display a mobility dashboard with track statistics.

    Args:
        data_out: DataFrame with statistics per modality
        figs: List of matplotlib figures with plots
        geo_map: Folium interactive map object
    """

    print("creating dashboard...")
    
    if TkinterDnD is not None:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    root.title("Mobility Dashboard - Track Statistics")
    root.geometry("1000x900")

    # Bring window to front on macOS
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

    # Create main container with scrollbar
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Header
    header = tk.Label(
        main_frame,
        text="Mobility Dashboard - Track Statistics Overview",
        font=("Arial", 20, "bold"),
        bg="#2c3e50",
        fg="white",
        pady=15
    )
    header.pack(fill=tk.X)

    # Create canvas with scrollbars for content
    canvas = tk.Canvas(main_frame)
    v_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    h_scrollbar = ttk.Scrollbar(main_frame, orient="horizontal", command=canvas.xview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

    # Statistics Table Section
    stats_frame = tk.LabelFrame(
        scrollable_frame,
        text="Track Statistics by Modality",
        font=("Arial", 14, "bold"),
        padx=10,
        pady=10
    )
    stats_frame.pack(fill=tk.X, padx=20, pady=10)

    # Create Treeview for statistics table
    columns = list(data_out.columns)
    tree = ttk.Treeview(stats_frame, columns=columns, show="headings", height=8)

    # Configure columns
    for col in columns:
        tree.heading(col, text=col.replace("_", " ").title())
        tree.column(col, width=90, anchor="center")

    # Insert data
    for idx, row in data_out.iterrows():
        values = []
        for col in columns:
            val = row[col]
            # Format timedelta objects
            if hasattr(val, 'total_seconds'):
                hours = val.total_seconds() / 3600
                values.append(f"{hours:.2f}h")
            elif isinstance(val, (int, float)):
                values.append(f"{val:.2f}")
            else:
                values.append(str(val))
        tree.insert("", tk.END, values=values)

    tree.pack(fill=tk.X, padx=5, pady=5)

    # Map Section
    map_frame = tk.LabelFrame(
        scrollable_frame,
        text="GPS Tracks Map - Interactive Visualization by Modality",
        font=("Arial", 14, "bold"),
        padx=10,
        pady=10
    )
    map_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Save folium map to HTML file
    map_html_path = os.path.join(os.getcwd(), 'map_visualization.html')
    geo_map.save(map_html_path)

    # Provide button to open map in browser (most reliable approach)
    info_label = tk.Label(
        map_frame,
        text="Interactive map with pan/zoom capabilities.\nClick button below to open in your browser.",
        font=("Arial", 11),
        justify='center'
    )
    info_label.pack(pady=20)

    def open_map():
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(map_html_path))

    open_button = tk.Button(
        map_frame,
        text="Open Interactive Map in Browser",
        command=open_map,
        font=("Arial", 13, "bold"),
        bg="#3498db",
        fg="white",
        padx=30,
        pady=15,
        cursor="hand2"
    )
    open_button.pack(pady=10)

    map_path_label = tk.Label(
        map_frame,
        text=f"Saved to: {map_html_path}",
        font=("Arial", 9),
        fg="gray"
    )
    map_path_label.pack(pady=5)

    # Plots Section
    plots_frame = tk.LabelFrame(
        scrollable_frame,
        text="Statistical Visualizations",
        font=("Arial", 14, "bold"),
        padx=10,
        pady=10
    )
    plots_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # Create grid for plots (2 columns)
    num_cols = 2
    for idx, fig in enumerate(figs):
        row = idx // num_cols
        col = idx % num_cols

        # Create frame for each plot
        plot_frame = tk.Frame(plots_frame, relief=tk.RIDGE, borderwidth=2)
        plot_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

        # Embed matplotlib figure
        canvas_plot = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Configure grid weights for responsive layout
    for i in range(num_cols):
        plots_frame.grid_columnconfigure(i, weight=1)

    # Plots for ML statistics
    ml_frame = tk.LabelFrame(
        scrollable_frame,
        text="Classification Model Test Results Confusion Matrix",
        font=("Arial", 14, "bold"),
        padx=10,
        pady=10
    )
    ml_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    # Load ML confusion matrix plot
    ml_fig = Figure(figsize=(8, 6))
    ml_ax = ml_fig.add_subplot(111) 
    ml_img_path = os.path.join('outputs', 'confusion_matrix_test.png')
    
    img = imread(ml_img_path)
    ml_ax.imshow(img)
    ml_ax.axis('off')
    canvas_ml = FigureCanvasTkAgg(ml_fig, master=ml_frame)
    canvas_ml.draw()
    canvas_ml.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add a text
    ml_text = tk.Text(ml_frame, height=4, font=("Arial", 15), background='white', fg='black', padx=10, pady=10)
    ml_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    ml_text.insert(tk.END, "Classes Labels: 0: Car; 1: Pedestrian; 2: Still")

    # make a drag and drop for a test folder or file to test the ML model on
    # Drag and Drop Section for ML Model Testing
    dnd_frame = tk.LabelFrame(
        scrollable_frame,
        text="Test ML Model - Drag & Drop Folder or File",
        font=("Arial", 14, "bold"),
        padx=10,
        pady=10
    )
    dnd_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    dnd_label = tk.Label(
        dnd_frame,
        text="Drag and drop a folder or file here to test the ML model.",
        font=("Arial", 12),
        bg="#ecf0f1",
        fg="#2c3e50",
        height=4
    )
    dnd_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    ml_fig_2 = Figure(figsize=(8, 6))
    ml_ax_2 = ml_fig_2.add_subplot(111)

    # Drag and drop support (works on macOS with TkinterDnD2)
    output_path = tk.StringVar(value=ml_img_path)
    if DND_FILES is None:
        dnd_label.config(text="Drag & drop requires TkinterDnD2 package.\nInstall with: pip install TkinterDnD2")
    else:
        try:
            dnd_label.drop_target_register(DND_FILES)

            def handle_drop(event):
                dropped_path = event.data.strip()
                
                dnd_label.config(text=f"Dropped: {dropped_path}\nTesting ML model...")
                new_path = test_model(dropped_path, 'models/classification_model.joblib')
                output_path.set(new_path)
                img = imread(new_path)
                ml_ax_2.imshow(img)
                ml_ax_2.axis('off')
                canvas_ml = FigureCanvasTkAgg(ml_fig_2, master=dnd_frame)
                canvas_ml.draw()
                canvas_ml.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            dnd_label.dnd_bind('<<Drop>>', handle_drop)
        except tk.TclError:
            dnd_label.config(text="Drag & drop unavailable: tkdnd extension not loaded.")

    # Pack canvas and scrollbars
    canvas.pack(side="left", fill="both", expand=True)
    v_scrollbar.pack(side="right", fill="y")
    h_scrollbar.pack(side="bottom", fill="x")

    # Trackpad and mouse wheel scrolling (macOS)
    def _on_mousewheel_vertical(event):
        canvas.yview_scroll(int(-1 * (event.delta)), "units")

    def _on_mousewheel_horizontal(event):
        canvas.xview_scroll(int(-1 * (event.delta)), "units")

    def _on_shift_mousewheel(event):
        canvas.xview_scroll(int(-1 * (event.delta)), "units")

    # Bind trackpad/mouse events
    canvas.bind_all("<MouseWheel>", _on_mousewheel_vertical)
    canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)
    # macOS trackpad horizontal scrolling
    canvas.bind_all("<Shift-MouseWheel>", _on_mousewheel_horizontal)

    # Start the GUI event loop
    root.mainloop()

    print("dashboard created, you can use it now.")
