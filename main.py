import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
from pathlib import Path

# Add the Interaction directory to the path
sys.path.append(str(Path(__file__).parent / "Interaction"))

from AskQuestion import RAGQuestionAnswerer

class RAGChatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Chat Assistant")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize RAG system
        self.rag = None
        self.is_processing = False
        
        self.setup_ui()
        self.initialize_rag_system()

    def setup_ui(self):
        """Set up the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="RAG Chat Assistant", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Chat display area
        chat_frame = ttk.LabelFrame(main_frame, text="Conversation", padding="10")
        chat_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            state=tk.DISABLED,
            font=("Arial", 10),
            bg="white"
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure text tags for styling
        self.chat_display.tag_configure("user", foreground="#0066cc", font=("Arial", 10, "bold"))
        self.chat_display.tag_configure("assistant", foreground="#009900")
        self.chat_display.tag_configure("info", foreground="#666666", font=("Arial", 9, "italic"))
        self.chat_display.tag_configure("sources", foreground="#cc6600", font=("Arial", 9))
        
        # Input area
        input_frame = ttk.LabelFrame(main_frame, text="Ask a Question", padding="10")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # Question input
        self.question_var = tk.StringVar()
        self.question_entry = ttk.Entry(
            input_frame, 
            textvariable=self.question_var, 
            font=("Arial", 11),
            width=50
        )
        self.question_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.question_entry.bind('<Return>', self.on_enter_pressed)
        self.question_entry.bind('<KeyPress>', self.on_key_press)
        
        # Send button
        self.send_button = ttk.Button(
            input_frame, 
            text="Send", 
            command=self.send_question,
            style="Accent.TButton"
        )
        self.send_button.grid(row=0, column=1)
        
        # Progress bar (hidden by default)
        self.progress_bar = ttk.Progressbar(
            input_frame, 
            mode='indeterminate',
            length=200
        )
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        self.progress_bar.grid_remove()  # Hide initially
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Top K setting
        ttk.Label(settings_frame, text="Context Chunks:").grid(row=0, column=0, padx=(0, 5))
        self.top_k_var = tk.IntVar(value=3)
        top_k_spinbox = ttk.Spinbox(
            settings_frame, 
            from_=1, 
            to=10, 
            textvariable=self.top_k_var,
            width=5
        )
        top_k_spinbox.grid(row=0, column=1, padx=(0, 10))
        
        # Include tables setting
        self.include_tables_var = tk.BooleanVar(value=True)
        tables_checkbox = ttk.Checkbutton(
            settings_frame, 
            text="Include Tables", 
            variable=self.include_tables_var
        )
        tables_checkbox.grid(row=0, column=2, padx=(0, 10))
        
        # Temperature setting
        ttk.Label(settings_frame, text="Temperature:").grid(row=0, column=3, padx=(0, 5))
        self.temperature_var = tk.DoubleVar(value=0.7)
        temp_spinbox = ttk.Spinbox(
            settings_frame, 
            from_=0.0, 
            to=1.0, 
            increment=0.1,
            textvariable=self.temperature_var,
            width=5,
            format="%.1f"
        )
        temp_spinbox.grid(row=0, column=4, padx=(0, 10))
        
        # Clear button
        clear_button = ttk.Button(
            settings_frame, 
            text="Clear Chat", 
            command=self.clear_chat
        )
        clear_button.grid(row=0, column=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Focus on input
        self.question_entry.focus_set()
        
        # Add welcome message
        self.add_message("Welcome to RAG Chat Assistant! Type your question and press Enter.", "info")

    def initialize_rag_system(self):
        """Initialize the RAG system in a separate thread"""
        def init_rag():
            try:
                self.status_var.set("Initializing RAG system...")
                self.rag = RAGQuestionAnswerer(
                    search_index_name="rag-test-index",
                    embedding_model="text-embedding-3-small",  # Updated to match your new embedder
                    default_top_k=3,
                    temperature=0.7
                )
                self.status_var.set("RAG system initialized successfully!")
                self.send_button.config(state=tk.NORMAL)
                
            except Exception as e:
                error_msg = f"Failed to initialize RAG system: {str(e)}"
                self.status_var.set(error_msg)
                messagebox.showerror("Initialization Error", error_msg)
        
        # Disable send button during initialization
        self.send_button.config(state=tk.DISABLED)
        
        # Start initialization in background thread
        threading.Thread(target=init_rag, daemon=True).start()

    def on_enter_pressed(self, event):
        """Handle Enter key press"""
        if not self.is_processing:
            self.send_question()
        return 'break'  # Prevent default behavior

    def on_key_press(self, event):
        """Handle key press events"""
        if event.keysym == 'Escape':
            self.question_entry.delete(0, tk.END)

    def send_question(self):
        """Send the question to the RAG system"""
        if self.is_processing or not self.rag:
            return
        
        question = self.question_var.get().strip()
        if not question:
            messagebox.showwarning("Empty Question", "Please enter a question.")
            return
        
        # Clear input
        self.question_var.set("")
        
        # Add user question to chat
        self.add_message(f"You: {question}", "user")
        
        # Start processing in background thread
        threading.Thread(target=self._process_question, args=(question,), daemon=True).start()

    def _process_question(self, question):
        """Process the question in a background thread"""
        self.is_processing = True
        
        # Update UI on main thread
        self.root.after(0, self._start_processing)
        
        try:
            # Get RAG response
            result = self.rag.ask_question(
                question=question,
                top_k=self.top_k_var.get(),
                include_tables=self.include_tables_var.get(),
                temperature=self.temperature_var.get()
            )
            
            # Update UI on main thread
            self.root.after(0, self._display_result, result)
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            self.root.after(0, self._display_error, error_msg)
        
        finally:
            self.is_processing = False
            self.root.after(0, self._stop_processing)

    def _start_processing(self):
        """Start processing UI updates"""
        self.send_button.config(state=tk.DISABLED)
        self.progress_bar.grid()
        self.progress_bar.start()
        self.status_var.set("Processing question...")

    def _stop_processing(self):
        """Stop processing UI updates"""
        self.send_button.config(state=tk.NORMAL)
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.status_var.set("Ready")
        self.question_entry.focus_set()

    def _display_result(self, result):
        """Display the RAG result"""
        # Add assistant response
        self.add_message(f"Assistant: {result['answer']}", "assistant")
        
        # Add metadata
        confidence = result.get('confidence', 'unknown')
        tokens = result.get('tokens_used', {})
        context_count = len(result.get('context_used', []))
        
        info_text = f"Confidence: {confidence} | Context chunks: {context_count} | Tokens: {tokens.get('total', 0)}"
        self.add_message(info_text, "info")
        
        # Add sources if available
        sources = result.get('sources', [])
        if sources:
            sources_text = "Sources: " + ", ".join([
                f"{source['document_title']} - {source['section_title']}" 
                for source in sources[:3]  # Show first 3 sources
            ])
            if len(sources) > 3:
                sources_text += f" (and {len(sources) - 3} more)"
            self.add_message(sources_text, "sources")
        
        self.add_message("", "info")  # Empty line for spacing

    def _display_error(self, error_msg):
        """Display an error message"""
        self.add_message(f"Error: {error_msg}", "info")

    def add_message(self, message, tag=""):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message + "\n", tag)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)  # Scroll to bottom

    def clear_chat(self):
        """Clear the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.add_message("Chat cleared. Ask your next question!", "info")


def main():
    """Main function to run the GUI"""
    try:
        root = tk.Tk()
        
        # Set the icon (optional)
        # root.iconbitmap("icon.ico")
        
        # Create the GUI
        app = RAGChatGUI(root)
        
        # Handle window closing
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start the GUI
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Application Error", f"Failed to start application: {str(e)}")


if __name__ == "__main__":
    main()