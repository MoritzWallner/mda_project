import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tempfile
import os


def create_dashboard(data_out, figs, geo_map):
    """
    Create and display a mobility dashboard with track statistics.

    Args:
        data_out: DataFrame with statistics per modality
        figs: List of matplotlib figures with plots
        geo_map: Folium interactive map object
    """

    print("creating dashboard...")
    
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
