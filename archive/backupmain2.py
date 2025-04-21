import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import sqlite3
import requests
from dotenv import load_dotenv, set_key
from datetime import datetime
import time
import json
import re

# --- Constants ---
CHAT_DB_SOURCE = os.path.expanduser("~/Library/Messages/chat.db")
CHAT_DB_DEST = os.path.join(os.path.dirname(__file__), "chat.db")
CHATS_DIR = os.path.join(os.path.dirname(__file__), "saved_chats")

class Chat:
    def __init__(self, name="New Chat"):
        self.name = name
        self.history = []
        self.id = int(time.time() * 1000)  # Unique ID based on timestamp
        self.last_updated = datetime.now()

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        self.last_updated = datetime.now()

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "history": self.history,
            "last_updated": self.last_updated.isoformat()
        }

    @staticmethod
    def from_dict(data):
        chat = Chat(data["name"])
        chat.id = data["id"]
        chat.history = data["history"]
        chat.last_updated = datetime.fromisoformat(data["last_updated"])
        return chat

class IMessageDBApp:
    def __init__(self, root):
        self.root = root
        self.root.title("iMessageDB LLM Query Tool")
        
        # Set minimum window size and default size
        self.root.minsize(1200, 700)
        self.root.geometry("1400x800")
        
        # Initialize variables
        self.llm_provider = tk.StringVar(value="LM Studio")  # Default to LM Studio
        self.openai_api_key = tk.StringVar()
        self.lmstudio_url = tk.StringVar()
        self.models = []
        self.selected_model = tk.StringVar()
        self.db_status = tk.StringVar(value="No Chat.db Found")
        
        # Chat management
        self.chats = []
        self.current_chat = None
        self.chat_index_map = {}
        
        # Create chats directory if it doesn't exist
        os.makedirs(CHATS_DIR, exist_ok=True)
        
        # Load environment and create GUI
        self.load_env()
        self.setup_gui()
        
        # Load saved chats and create a new one if none exist
        self.load_saved_chats()
        if not self.chats:
            self.new_chat()
            
        self.update_db_status()
        
        # Auto-load models based on provider
        if self.llm_provider.get() == "OpenAI" and self.openai_api_key.get().strip():
            self.load_models()
        elif self.llm_provider.get() == "LM Studio" and self.lmstudio_url.get().strip():
            self.load_models()

    def setup_gui(self):
        # Configure grid weights for the root window
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main container with sidebar and content
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create sidebar frame
        self.sidebar = ttk.Frame(self.paned, width=250)
        self.sidebar.grid_propagate(False)  # Prevent sidebar from shrinking
        self.paned.add(self.sidebar, weight=1)
        
        # Create chat list
        self.chat_list_frame = ttk.Frame(self.sidebar)
        self.chat_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Buttons frame at top of sidebar
        buttons_frame = ttk.Frame(self.chat_list_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 5))
        
        # New Chat and Delete Chat buttons
        ttk.Button(buttons_frame, text="New Chat", command=self.new_chat).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.delete_chat_btn = ttk.Button(buttons_frame, text="Delete Chat", command=self.delete_selected_chat)
        self.delete_chat_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Chats listbox with scrollbar
        self.chats_frame = ttk.Frame(self.chat_list_frame)
        self.chats_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chats_scrollbar = ttk.Scrollbar(self.chats_frame)
        self.chats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.chats_listbox = tk.Listbox(self.chats_frame, yscrollcommand=self.chats_scrollbar.set,
                                      selectmode=tk.SINGLE, activestyle='none',
                                      bg='#2b2b2b', fg='white', selectbackground='#404040',
                                      font=('Helvetica', 11))
        self.chats_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.chats_scrollbar.config(command=self.chats_listbox.yview)
        
        # Bind chat selection
        self.chats_listbox.bind('<<ListboxSelect>>', self.on_chat_selected)
        
        # Create main content frame
        self.content = ttk.Frame(self.paned)
        self.content.grid_columnconfigure(0, weight=1)  # Make content frame expandable
        self.paned.add(self.content, weight=4)
        
        # Top controls frame
        frm_top = ttk.Frame(self.content)
        frm_top.pack(padx=10, pady=5, fill=tk.X)

        btn_copy_db = ttk.Button(frm_top, text="Load chat.db", command=self.load_db)
        btn_copy_db.pack(side=tk.LEFT)

        ttk.Label(frm_top, text="LLM Provider:").pack(side=tk.LEFT, padx=(10,0))
        current_provider = self.llm_provider.get()
        opt_provider = ttk.OptionMenu(frm_top, self.llm_provider, current_provider, "OpenAI", "LM Studio", command=self.on_provider_change)
        opt_provider.pack(side=tk.LEFT)

        self.frm_openai = ttk.Frame(frm_top)
        ttk.Label(self.frm_openai, text="OpenAI API Key:").pack(side=tk.LEFT)
        ttk.Entry(self.frm_openai, textvariable=self.openai_api_key, width=30, show="*").pack(side=tk.LEFT)
        
        self.frm_lmstudio = ttk.Frame(frm_top)
        ttk.Label(self.frm_lmstudio, text="LM Studio URL:").pack(side=tk.LEFT)
        ttk.Entry(self.frm_lmstudio, textvariable=self.lmstudio_url, width=30).pack(side=tk.LEFT)
        
        # Show/hide frames based on current provider
        if current_provider == "OpenAI":
            self.frm_openai.pack(side=tk.LEFT, padx=(10,0))
            self.frm_lmstudio.pack_forget()
        else:
            self.frm_lmstudio.pack(side=tk.LEFT, padx=(10,0))
            self.frm_openai.pack_forget()

        btn_load_models = ttk.Button(frm_top, text="Load Models", command=self.load_models)
        btn_load_models.pack(side=tk.LEFT, padx=(10,0))
        ttk.Label(frm_top, text="Model:").pack(side=tk.LEFT, padx=(10,0))
        self.opt_models = ttk.OptionMenu(frm_top, self.selected_model, "")
        self.opt_models.pack(side=tk.LEFT)

        btn_save_creds = ttk.Button(frm_top, text="Save Credentials", command=self.save_env)
        btn_save_creds.pack(side=tk.LEFT, padx=(10,0))

        # Query frame
        frm_query = ttk.Frame(self.content)
        frm_query.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frm_query, text="Ask a question:").pack(side=tk.LEFT)
        self.entry_query = ttk.Entry(frm_query, width=60)
        self.entry_query.pack(side=tk.LEFT, padx=(5,0), fill=tk.X, expand=True)
        self.entry_query.bind('<Return>', lambda event: self.ask_question())
        
        self.btn_ask = ttk.Button(frm_query, text="Ask", command=self.ask_question)
        self.btn_ask.pack(side=tk.LEFT, padx=(5,0))
        
        # DB status label with name for identification
        self.db_status_label = ttk.Label(frm_query, textvariable=self.db_status)
        self.db_status_label.pack(side=tk.LEFT, padx=(10,0))

        # Results text area
        self.txt_results = scrolledtext.ScrolledText(self.content, width=100, height=25, 
                                                   font=("Consolas", 11), bg='#1e1e1e', fg='white',
                                                   insertbackground='white')
        self.txt_results.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Configure text widget tags
        self.txt_results.tag_configure("user", foreground="#6495ED")
        self.txt_results.tag_configure("sql", foreground="#98FB98")
        self.txt_results.tag_configure("result", foreground="#FFFFFF")
        self.txt_results.tag_configure("error", foreground="#FF6B6B")
        self.txt_results.tag_configure("model_info", foreground="#FFD700")  # Gold color for model info
        self.txt_results.tag_configure("summary", foreground="#FFA500")  # Orange color for summaries

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling in the chat list"""
        if self.chat_canvas.winfo_containing(event.x_root, event.y_root) == self.chat_canvas:
            self.chat_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def update_chat_list(self):
        """Update the chat listbox with current chats"""
        self.chats_listbox.delete(0, tk.END)
        # Sort chats by last updated time
        sorted_chats = sorted(self.chats, key=lambda x: x.last_updated, reverse=True)
        
        # Store the mapping of listbox indices to chat objects
        self.chat_index_map = {i: chat for i, chat in enumerate(sorted_chats)}
        
        for chat in sorted_chats:
            self.chats_listbox.insert(tk.END, chat.name)
            if chat == self.current_chat:
                self.chats_listbox.selection_clear(0, tk.END)
                self.chats_listbox.selection_set(tk.END)

    def select_chat(self, chat):
        """Select a chat and display its contents"""
        self.current_chat = chat
        self.update_chat_list()  # Update visual selection
        self.display_conversation()

    def delete_selected_chat(self):
        """Delete the currently selected chat"""
        selection = self.chats_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a chat to delete")
            return
            
        index = selection[0]
        if index not in self.chat_index_map:
            return
            
        chat = self.chat_index_map[index]
        
        if messagebox.askyesno("Delete Chat", f"Are you sure you want to delete '{chat.name}'?"):
            # Remove chat file
            chat_file = os.path.join(CHATS_DIR, f"{chat.id}.json")
            try:
                if os.path.exists(chat_file):
                    os.remove(chat_file)
            except Exception as e:
                print(f"Error deleting chat file: {e}")
            
            # Remove from chats list
            self.chats.remove(chat)
            
            # If deleted current chat, switch to most recent or create new
            if chat == self.current_chat:
                if self.chats:
                    self.current_chat = max(self.chats, key=lambda x: x.last_updated)
                else:
                    self.new_chat()
            
            # Update display
            self.update_chat_list()
            self.display_conversation()

    def new_chat(self):
        """Create a new chat and make it current"""
        chat = Chat()
        self.chats.append(chat)
        self.current_chat = chat
        self.update_chat_list()
        self.display_conversation()
        self.save_chats()
        
    def save_chats(self):
        """Save all chats to disk"""
        for chat in self.chats:
            chat_file = os.path.join(CHATS_DIR, f"{chat.id}.json")
            with open(chat_file, 'w') as f:
                json.dump(chat.to_dict(), f, indent=2)

    def load_saved_chats(self):
        """Load saved chats from disk"""
        if not os.path.exists(CHATS_DIR):
            return
        
        for filename in os.listdir(CHATS_DIR):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(CHATS_DIR, filename), 'r') as f:
                        chat_data = json.load(f)
                        chat = Chat.from_dict(chat_data)
                        self.chats.append(chat)
                except Exception as e:
                    print(f"Error loading chat {filename}: {e}")
        
        # Set current chat to the most recent one if any chats were loaded
        if self.chats:
            self.current_chat = max(self.chats, key=lambda x: x.last_updated)
            self.update_chat_list()
            self.display_conversation()

    def summarize_chat(self):
        """Generate a summary/title for the current chat"""
        if not self.current_chat or not self.current_chat.history:
            return "New Chat"
            
        try:
            provider = self.llm_provider.get()
            if provider == "OpenAI":
                import openai
                client = openai.OpenAI(api_key=self.openai_api_key.get().strip())
                response = client.chat.completions.create(
                    model=self.selected_model.get(),
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Please provide a very brief (2-4 words) title summarizing the main topic of this chat."},
                        {"role": "user", "content": str(self.current_chat.history)}
                    ],
                    max_tokens=10,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            else:
                url = self.lmstudio_url.get().strip().rstrip('/')
                resp = requests.post(f"{url}/v1/chat/completions", json={
                    "model": self.selected_model.get(),
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. Please provide a very brief (2-4 words) title summarizing the main topic of this chat."},
                        {"role": "user", "content": str(self.current_chat.history)}
                    ],
                    "max_tokens": 10,
                    "temperature": 0.3
                })
                resp.raise_for_status()
                return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error generating chat summary: {e}")
            return "New Chat"

    def extract_clean_text_from_blob(self, blob):
        """Extract clean text from a message blob"""
        try:
            raw = blob.decode('utf-8', errors='ignore')
            if 'NSString' in raw:
                nsstring_index = raw.find('NSString')
                raw_after_nsstring = raw[nsstring_index:]
                plus_index = raw_after_nsstring.find('+')
                if plus_index != -1:
                    text_start = plus_index + 1
                    text_candidate = raw_after_nsstring[text_start:]
                    for cutoff in ['\\x02', '\\x0c', '\\n', 'NSDictionary', 'NSValue']:
                        cut_idx = text_candidate.find(cutoff)
                        if cut_idx != -1:
                            text_candidate = text_candidate[:cut_idx]
                    return text_candidate.strip()
            return ''.join(c for c in raw if 32 <= ord(c) <= 126).strip()
        except Exception as e:
            return f"[Error extracting text: {e}]"

    def fully_clean_text(self, text):
        """Clean up message text"""
        if not text:
            return ""
        # Remove all control characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        # Remove trailing weird patterns like iI, iIA, iI<, etc
        text = re.sub(r'(\s*iI[a-zA-Z0-9]*\s*$)', '', text)
        # Strip extra spaces
        return text.strip()

    def format_message_results(self, rows):
        """Format message results in a readable way"""
        if not rows:
            return "No messages found."
            
        output = []
        for row in rows:
            date_val, contact, is_from_me, text, attributed_body = row
            
            # Convert Apple's weird date format
            if date_val:
                timestamp = int(date_val) / 1000000000 + datetime(2001, 1, 1).timestamp()
                message_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            else:
                message_date = "Unknown Date"

            # Try to get text from attributed body first
            if attributed_body:
                clean_text = self.extract_clean_text_from_blob(attributed_body)
            else:
                clean_text = text if text else ""

            # Clean up the text
            clean_text = self.fully_clean_text(clean_text)
            if not clean_text:
                clean_text = "[No message content]"

            # Format the message
            sender = "Me" if is_from_me == 1 else (contact if contact else "Unknown")
            output.append(f"[{message_date}] {sender}: {clean_text}")

        return "\n".join(output)

    def normalize_phone_number(self, number):
        """Normalize phone number for flexible matching"""
        # Remove any non-digit characters
        digits = re.sub(r'\D', '', number)
        # If it's a 10-digit number, add +1
        if len(digits) == 10:
            return f"+1{digits}"
        # If it's 11 digits and starts with 1, add +
        elif len(digits) == 11 and digits.startswith('1'):
            return f"+{digits}"
        return digits

    def needs_summarization(self, question):
        """Determine if the question requires summarization of results"""
        summarization_keywords = [
            'summarize', 'summary', 'summarise', 'summarisation',
            'what was i talking about',
            'what did we discuss',
            'what were we talking about',
            'give me an overview',
            'tell me about',
            'analyze',
            'analyse',
            'recap',
            'brief me',
            'highlight',
            'main points',
            'key points',
            'topics',
            'give me a summary'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in summarization_keywords)

    def summarize_results(self, results, question, provider, model):
        """Use LLM to summarize query results"""
        try:
            system_prompt = """You are a helpful assistant analyzing iMessage conversations. 
Given the messages below, provide a concise summary focusing on:
- Main topics discussed
- Key points or decisions made
- Notable patterns or themes
- Time period covered

Keep the summary clear and focused. If there are no messages, just say so."""

            if provider == "OpenAI":
                import openai
                client = openai.OpenAI(api_key=self.openai_api_key.get().strip())
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Question: {question}\n\nMessages:\n{results}"}
                    ],
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            else:
                url = self.lmstudio_url.get().strip().rstrip('/')
                resp = requests.post(f"{url}/v1/chat/completions", json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Question: {question}\n\nMessages:\n{results}"}
                    ],
                    "temperature": 0.3
                })
                resp.raise_for_status()
                return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error generating summary: {e}"

    def ask_question(self):
        if not self.current_chat:
            self.new_chat()
            
        self.btn_ask.config(state=tk.DISABLED, text="Processing...")
        self.root.update_idletasks()
        question = self.entry_query.get().strip()
        if not question:
            messagebox.showerror("Error", "Please enter a question.")
            self.btn_ask.config(state=tk.NORMAL, text="Ask")
            return
        if not os.path.exists(CHAT_DB_DEST):
            messagebox.showerror("Error", "chat.db not found in project directory. Please load it first.")
            self.btn_ask.config(state=tk.NORMAL, text="Ask")
            return
        provider = self.llm_provider.get()
        model = self.selected_model.get()
        if not model:
            messagebox.showerror("Error", "Please load and select a model first.")
            self.btn_ask.config(state=tk.NORMAL, text="Ask")
            return

        # Verify model is compatible with current provider
        if provider == "OpenAI" and not any(model == m for m in self.models):
            messagebox.showerror("Error", "Selected model is not compatible with OpenAI. Please load OpenAI models and select one.")
            self.clear_models()
            self.btn_ask.config(state=tk.NORMAL, text="Ask")
            return
        elif provider == "LM Studio" and not any(model == m for m in self.models):
            messagebox.showerror("Error", "Selected model is not compatible with LM Studio. Please load LM Studio models and select one.")
            self.clear_models()
            self.btn_ask.config(state=tk.NORMAL, text="Ask")
            return

        # Add model info to chat history
        model_info = f"Using {provider} - {model}"
        self.current_chat.add_message("model_info", model_info)
        
        # Add user message to chat history
        self.current_chat.add_message("user", question)
        
        # Check if this question needs summarization
        needs_summary = self.needs_summarization(question)
        
        # Filter chat history for LLM to only include user and assistant messages
        llm_messages = [
            msg for msg in self.current_chat.history 
            if msg["role"] in ["user", "assistant"]
        ]
        
        # Get SQL query from LLM
        try:
            import json
            if provider == "OpenAI":
                import openai
                client = openai.OpenAI(api_key=self.openai_api_key.get().strip())
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": """You are an expert at generating SQLite SQL queries for the iMessage chat.db. The schema includes:
- message table: Contains messages with text, handle_id (foreign key to handle.rowid), date, is_from_me, attributedBody (contains the actual message content)
- handle table: Contains contact info with id (phone number/email), rowid (primary key)

IMPORTANT QUERY RULES:
1. For ALL message content queries (including summary requests), ALWAYS include these exact fields in this order:
   message.date,
   handle.id AS contact,
   message.is_from_me,
   message.text,
   message.attributedBody

2. For date-based queries:
   - message.date is in nanoseconds since 2001-01-01
   - Use 978307200 as the epoch offset (seconds between 1970 and 2001)
   - For "last 24 hours" or recent time queries, use:
     WHERE datetime(message.date/1000000000 + 978307200, 'unixepoch') >= datetime('now', '-1 day')
   - For specific date ranges, use similar datetime() comparisons

3. For aggregate queries (ONLY for count/statistics, NOT for summaries):
   A. For simple counts (total messages, etc):
   {"sql": "SELECT COUNT(*) FROM message"}
   
   B. For yearly message counts:
   {"sql": "SELECT strftime('%Y', datetime(date/1000000000 + 978307200, 'unixepoch')) AS year, COUNT(*) AS message_count FROM message WHERE is_from_me = 1 GROUP BY year ORDER BY year ASC"}
   
   C. For sender statistics:
   {"sql": "SELECT handle.id AS contact, COUNT(*) AS message_count FROM message JOIN handle ON message.handle_id = handle.rowid GROUP BY handle.id ORDER BY message_count DESC"}

4. For message content:
   - Always include both text and attributedBody fields
   - Join with handle table to get contact information
   - Order by date DESC for most recent messages first
   - NEVER use GROUP_CONCAT for summaries - instead return individual messages

5. For summary requests:
   - Use the same query format as message content queries (#1)
   - The application will handle the summarization after getting the messages
   - Do NOT try to summarize in SQL - return the full messages

Only output the SQL query as a JSON object with a single 'sql' key. No explanation needed."""}
                    ] + llm_messages,
                    max_tokens=500,
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content.strip()
                try:
                    sql_json = json.loads(content)
                    sql = sql_json.get('sql', '').strip()
                except Exception:
                    sql = content
            else:
                url = self.lmstudio_url.get().strip().rstrip('/')
                resp = requests.post(f"{url}/v1/chat/completions", json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": """You are an expert at generating SQLite SQL queries for the iMessage chat.db. The schema includes:
- message table: Contains messages with text, handle_id (foreign key to handle.rowid), date, is_from_me, attributedBody (contains the actual message content)
- handle table: Contains contact info with id (phone number/email), rowid (primary key)

IMPORTANT QUERY RULES:
1. For ALL message content queries (including summary requests), ALWAYS include these exact fields in this order:
   message.date,
   handle.id AS contact,
   message.is_from_me,
   message.text,
   message.attributedBody

2. For date-based queries:
   - message.date is in nanoseconds since 2001-01-01
   - Use 978307200 as the epoch offset (seconds between 1970 and 2001)
   - For "last 24 hours" or recent time queries, use:
     WHERE datetime(message.date/1000000000 + 978307200, 'unixepoch') >= datetime('now', '-1 day')
   - For specific date ranges, use similar datetime() comparisons

3. For aggregate queries (ONLY for count/statistics, NOT for summaries):
   A. For simple counts (total messages, etc):
   {"sql": "SELECT COUNT(*) FROM message"}
   
   B. For yearly message counts:
   {"sql": "SELECT strftime('%Y', datetime(date/1000000000 + 978307200, 'unixepoch')) AS year, COUNT(*) AS message_count FROM message WHERE is_from_me = 1 GROUP BY year ORDER BY year ASC"}
   
   C. For sender statistics:
   {"sql": "SELECT handle.id AS contact, COUNT(*) AS message_count FROM message JOIN handle ON message.handle_id = handle.rowid GROUP BY handle.id ORDER BY message_count DESC"}

4. For message content:
   - Always include both text and attributedBody fields
   - Join with handle table to get contact information
   - Order by date DESC for most recent messages first
   - NEVER use GROUP_CONCAT for summaries - instead return individual messages

5. For summary requests:
   - Use the same query format as message content queries (#1)
   - The application will handle the summarization after getting the messages
   - Do NOT try to summarize in SQL - return the full messages

Only output the SQL query as a JSON object with a single 'sql' key. No explanation needed."""}
                    ] + llm_messages,
                    "max_tokens": 500,
                    "temperature": 0,
                    "structured_output": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string"}
                        },
                        "required": ["sql"]
                    }
                })
                resp.raise_for_status()
                content = resp.json()['choices'][0]['message']['content'].strip()
                try:
                    sql_json = json.loads(content)
                    sql = sql_json.get('sql', '').strip()
                except Exception:
                    sql = content
        except Exception as e:
            error_msg = f"Error calling LLM: {e}\n"
            self.current_chat.add_message("error", error_msg)
            self.display_conversation()
            self.btn_ask.config(state=tk.NORMAL, text="Ask")
            return

        # Clear the question entry
        self.entry_query.delete(0, tk.END)

        # Add assistant message to chat history
        self.current_chat.add_message("assistant", sql)

        # Run SQL query
        try:
            conn = sqlite3.connect(CHAT_DB_DEST)
            cursor = conn.cursor()
            
            # Check if this is a message query
            is_message_query = (
                'message.text' in sql.lower() or 
                'message.attributedbody' in sql.lower() or
                ('from message' in sql.lower() and 'join handle' in sql.lower())  # Message queries should join with handle
            )
            
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Handle different types of queries
            if len(columns) == 1 and 'count' in columns[0].lower():
                # This is a COUNT query
                result_text = f"Total: {rows[0][0]}"
            elif is_message_query and len(columns) >= 5:  # Message queries should have at least 5 columns
                result_text = self.format_message_results(rows)
                
                # If summarization is needed and we have messages, process them
                if needs_summary and result_text != "No messages found.":
                    summary = self.summarize_results(result_text, question, provider, model)
                    self.current_chat.add_message("summary", summary)
            else:
                # For all other queries, use tabular format
                result_text = self.format_results(columns, rows)
            
            self.current_chat.add_message("result", result_text)
        except Exception as e:
            error_msg = f"Error running SQL: {e}\n"
            self.current_chat.add_message("error", error_msg)
        finally:
            conn.close()

        # Update chat name if it's still "New Chat"
        if self.current_chat.name == "New Chat":
            self.current_chat.name = self.summarize_chat()
            self.update_chat_list()

        # Save chats and display conversation
        self.save_chats()
        self.display_conversation()
        self.btn_ask.config(state=tk.NORMAL, text="Ask")

    def display_conversation(self):
        """Display the current chat conversation"""
        if not self.current_chat:
            return
            
        self.txt_results.delete('1.0', tk.END)
        
        for msg in self.current_chat.history:
            if msg['role'] == 'model_info':
                self.txt_results.insert(tk.END, f"{msg['content']}\n", "model_info")
            elif msg['role'] == 'user':
                self.txt_results.insert(tk.END, f"User: {msg['content']}\n", "user")
            elif msg['role'] == 'assistant':
                self.txt_results.insert(tk.END, f"SQL: {msg['content']}\n", "sql")
            elif msg['role'] == 'result':
                self.txt_results.insert(tk.END, f"\n--- Query Results ---\n{msg['content']}\n", "result")
            elif msg['role'] == 'summary':
                self.txt_results.insert(tk.END, f"\n--- Summary ---\n{msg['content']}\n", "summary")
            elif msg['role'] == 'error':
                self.txt_results.insert(tk.END, f"Error: {msg['content']}\n", "error")
        
        self.txt_results.see(tk.END)

    def update_db_status(self):
        if os.path.exists(CHAT_DB_DEST):
            try:
                # Get file modification time
                mod_time = os.path.getmtime(CHAT_DB_DEST)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                
                # Verify it's a valid chat.db by checking a table
                conn = sqlite3.connect(CHAT_DB_DEST)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='message'")
                has_message_table = cursor.fetchone()[0] > 0
                conn.close()
                
                if has_message_table:
                    self.db_status.set(f"chat.db loaded (from {mod_time_str})")
                    lbl_db_status = self.root.nametowidget(str(self.db_status_label))
                    lbl_db_status.configure(foreground='#98FB98')  # Light green color
                else:
                    self.db_status.set("Invalid chat.db format")
                    lbl_db_status = self.root.nametowidget(str(self.db_status_label))
                    lbl_db_status.configure(foreground='#FF6B6B')  # Red color
            except Exception:
                self.db_status.set("Error reading chat.db")
                lbl_db_status = self.root.nametowidget(str(self.db_status_label))
                lbl_db_status.configure(foreground='#FF6B6B')  # Red color
        else:
            self.db_status.set("No Chat.db Found")
            lbl_db_status = self.root.nametowidget(str(self.db_status_label))
            lbl_db_status.configure(foreground='#FF6B6B')  # Red color

    def load_db(self):
        try:
            if os.path.exists(CHAT_DB_DEST):
                os.remove(CHAT_DB_DEST)
            shutil.copy2(CHAT_DB_SOURCE, CHAT_DB_DEST)
            self.update_db_status()
            messagebox.showinfo("Success", f"Loaded fresh copy of chat.db")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load chat.db: {e}")
            self.update_db_status()

    def clear_models(self):
        """Clear the models dropdown and reset selection"""
        self.models = []
        self.selected_model.set("")
        menu = self.opt_models['menu']
        menu.delete(0, 'end')

    def update_models_dropdown(self):
        """Update the models dropdown with current models list"""
        menu = self.opt_models['menu']
        menu.delete(0, 'end')
        for model in self.models:
            menu.add_command(label=model, command=tk._setit(self.selected_model, model))
        if self.models:
            self.selected_model.set(self.models[0])
        else:
            self.selected_model.set("")

    def load_models(self):
        """Load models without showing success popup"""
        provider = self.llm_provider.get()
        self.clear_models()
        
        if provider == "OpenAI":
            success, error = self.load_openai_models()
        else:
            success, error = self.load_lmstudio_models()
            
        if success:
            self.update_models_dropdown()
            if not self.models:
                messagebox.showwarning("Warning", "No compatible models found")
        else:
            messagebox.showerror("Error", error)

    def load_openai_models(self):
        """Load models from OpenAI"""
        api_key = self.openai_api_key.get().strip()
        if not api_key:
            return False, "OpenAI API key not set"
        try:
            import openai
            openai.api_key = api_key
            models = openai.models.list()
            # Filter for GPT models, excluding audio/vision models
            self.models = [
                m.id for m in models.data 
                if 'gpt' in m.id.lower() 
                and not any(x in m.id.lower() for x in ['audio', 'vision', 'preview'])
            ]
            # Set default model if available
            if 'gpt-4.1' in self.models:
                self.selected_model.set('gpt-4.1')
            return True, None
        except Exception as e:
            return False, f"Failed to load OpenAI models: {e}"

    def load_lmstudio_models(self):
        """Load models from LM Studio"""
        url = self.lmstudio_url.get().strip().rstrip('/')
        if not url:
            return False, "LM Studio URL not set"
        try:
            resp = requests.get(f"{url}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            self.models = [m['id'] for m in data.get('data', [])]
            # Set default model if available
            if 'meta-llama-3.1-8b-instruct' in self.models:
                self.selected_model.set('meta-llama-3.1-8b-instruct')
            return True, None
        except Exception as e:
            return False, f"Failed to load LM Studio models: {e}"

    def on_provider_change(self, *args):
        # Clear existing models first
        self.clear_models()
        
        if self.llm_provider.get() == "OpenAI":
            self.frm_lmstudio.pack_forget()
            self.frm_openai.pack(side=tk.LEFT, padx=(10,0))
            # Auto-load OpenAI models if API key exists
            if self.openai_api_key.get().strip():
                self.load_models()
        else:
            self.frm_openai.pack_forget()
            self.frm_lmstudio.pack(side=tk.LEFT, padx=(10,0))
            # Auto-load LM Studio models if URL exists
            if self.lmstudio_url.get().strip():
                self.load_models()

    def load_env(self):
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            api_key = os.getenv('OPENAI_API_KEY', '')
            lmstudio_url = os.getenv('LMSTUDIO_URL', '')
            last_provider = os.getenv('LAST_PROVIDER', 'LM Studio')
            
            if api_key:
                self.openai_api_key.set(api_key)
            if lmstudio_url:
                self.lmstudio_url.set(lmstudio_url)
            if last_provider:
                self.llm_provider.set(last_provider)

    def save_env(self):
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if not os.path.exists(env_path):
            with open(env_path, 'w') as f:
                f.write('')
        set_key(env_path, 'OPENAI_API_KEY', self.openai_api_key.get())
        set_key(env_path, 'LMSTUDIO_URL', self.lmstudio_url.get())
        
        # Save last used provider and model
        set_key(env_path, 'LAST_PROVIDER', self.llm_provider.get())
        if self.selected_model.get():
            set_key(env_path, 'LAST_MODEL', self.selected_model.get())
            
        messagebox.showinfo("Saved", f"Settings saved to {env_path}")

    def format_results(self, columns, rows):
        """Format query results in a tabular format"""
        if not rows:
            return "No results."
            
        # Use tabular format for all non-special queries
        output = '\t'.join(str(col) for col in columns) + '\n'
        for row in rows:
            output += '\t'.join(str(x) for x in row) + '\n'
        return output

    def on_chat_selected(self, event):
        """Handle chat selection from listbox"""
        selection = self.chats_listbox.curselection()
        if selection:
            index = selection[0]
            if index in self.chat_index_map:
                self.current_chat = self.chat_index_map[index]
                self.display_conversation()

    def load_last_model(self):
        """Try to load the last used model"""
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_path):
            last_provider = os.getenv('LAST_PROVIDER', 'LM Studio')
            last_model = os.getenv('LAST_MODEL', '')
            
            if last_provider and last_model:
                # Set the provider first
                self.llm_provider.set(last_provider)
                self.on_provider_change()
                
                # Try to load the models
                success = False
                if last_provider == "OpenAI" and self.openai_api_key.get():
                    success, _ = self.load_openai_models()
                elif last_provider == "LM Studio" and self.lmstudio_url.get():
                    success, _ = self.load_lmstudio_models()
                
                # If models loaded successfully and last model is available, select it
                if success and last_model in self.models:
                    self.selected_model.set(last_model)
                    self.update_models_dropdown()

def main():
    root = tk.Tk()
    
    # Set dock icon on macOS - try multiple methods
    icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
    if os.path.exists(icon_path):
        try:
            # Method 1: Try tk::mac::setAppIcon (macOS specific)
            root.tk.call('tk::mac::setAppIcon', os.path.abspath(icon_path))
        except Exception as e1:
            try:
                # Method 2: Try setting window icon
                img = tk.PhotoImage(file=icon_path)
                root.tk.call('wm', 'iconphoto', root._w, img)
            except Exception as e2:
                try:
                    # Method 3: Try basic iconphoto
                    root.iconphoto(True, tk.PhotoImage(file=icon_path))
                except Exception as e3:
                    try:
                        # Method 4: Try NSApplication dock icon (macOS specific)
                        from Foundation import NSImage
                        from AppKit import NSApplication
                        image = NSImage.alloc().initWithContentsOfFile_(os.path.abspath(icon_path))
                        NSApplication.sharedApplication().setApplicationIconImage_(image)
                    except Exception as e4:
                        print(f"Could not set icon: {e1}, {e2}, {e3}, {e4}")
    
    app = IMessageDBApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
