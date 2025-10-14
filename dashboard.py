import tkinter as tk


def create_dashboard():
    """Create and display a simple tkinter window with Hello World"""
    root = tk.Tk()
    root.title("Dashboard")
    root.geometry("400x300")

    # Bring window to front on macOS
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

    # Create a label with Hello World
    label = tk.Label(
        root,
        text="Hello World",
        font=("Arial", 24),
        fg="blue"
    )
    label.pack(expand=True)

    # Start the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    create_dashboard()
