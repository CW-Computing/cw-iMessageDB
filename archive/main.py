import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import sqlite3
import requests
from dotenv import load_dotenv, set_key

# --- Constants ---
CHAT_DB_SOURCE = os.path.expanduser("~/Library/Messages/chat.db")
CHAT_DB_DEST = os.path.join(os.path.dirname(__file__), "chat.db")

class IMessageDBApp:
    def __init__(self, root):
        self.root = root
        self.root.title("iMessageDB LLM Query Tool")
        self.llm_provider = tk.StringVar(value="OpenAI")
        self.openai_api_key = tk.StringVar()
        self.lmstudio_url = tk.StringVar()
        self.models = []
        self.selected_model = tk.StringVar()
        self.chat_history = []
        self.load_env()
        self.setup_gui()

    def setup_gui(self):
        frm_top = tk.Frame(self.root)
        frm_top.pack(padx=10, pady=5, fill=tk.X)

        btn_copy_db = tk.Button(frm_top, text="Copy chat.db", command=self.copy_db)
        btn_copy_db.pack(side=tk.LEFT)

        tk.Label(frm_top, text="LLM Provider:").pack(side=tk.LEFT, padx=(10,0))
        opt_provider = tk.OptionMenu(frm_top, self.llm_provider, "OpenAI", "LM Studio", command=self.on_provider_change)
        opt_provider.pack(side=tk.LEFT)

        self.frm_openai = tk.Frame(frm_top)
        tk.Label(self.frm_openai, text="OpenAI API Key:").pack(side=tk.LEFT)
        tk.Entry(self.frm_openai, textvariable=self.openai_api_key, width=30, show="*").pack(side=tk.LEFT)
        self.frm_openai.pack(side=tk.LEFT, padx=(10,0))

        self.frm_lmstudio = tk.Frame(frm_top)
        tk.Label(self.frm_lmstudio, text="LM Studio URL:").pack(side=tk.LEFT)
        tk.Entry(self.frm_lmstudio, textvariable=self.lmstudio_url, width=30).pack(side=tk.LEFT)
        # Initially hide LM Studio fields
        self.frm_lmstudio.pack_forget()

        btn_load_models = tk.Button(frm_top, text="Load Models", command=self.load_models)
        btn_load_models.pack(side=tk.LEFT, padx=(10,0))
        tk.Label(frm_top, text="Model:").pack(side=tk.LEFT, padx=(10,0))
        self.opt_models = tk.OptionMenu(frm_top, self.selected_model, "")
        self.opt_models.pack(side=tk.LEFT)

        btn_save_creds = tk.Button(frm_top, text="Save Credentials", command=self.save_env)
        btn_save_creds.pack(side=tk.LEFT, padx=(10,0))

        frm_query = tk.Frame(self.root)
        frm_query.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frm_query, text="Ask a question:").pack(side=tk.LEFT)
        self.entry_query = tk.Entry(frm_query, width=60)
        self.entry_query.pack(side=tk.LEFT, padx=(5,0))
        self.btn_ask = tk.Button(frm_query, text="Ask", command=self.ask_question)
        self.btn_ask.pack(side=tk.LEFT, padx=(5,0))
        btn_new_chat = tk.Button(frm_query, text="New Chat", command=self.new_chat)
        btn_new_chat.pack(side=tk.LEFT, padx=(5,0))

        self.txt_results = scrolledtext.ScrolledText(self.root, width=100, height=25, font=("Consolas", 10))
        self.txt_results.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    def copy_db(self):
        try:
            shutil.copy2(CHAT_DB_SOURCE, CHAT_DB_DEST)
            messagebox.showinfo("Success", f"Copied chat.db to {CHAT_DB_DEST}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy chat.db: {e}")

    def on_provider_change(self, *args):
        if self.llm_provider.get() == "OpenAI":
            self.frm_lmstudio.pack_forget()
            self.frm_openai.pack(side=tk.LEFT, padx=(10,0))
        else:
            self.frm_openai.pack_forget()
            self.frm_lmstudio.pack(side=tk.LEFT, padx=(10,0))

    def load_models(self):
        provider = self.llm_provider.get()
        self.models = []
        if provider == "OpenAI":
            api_key = self.openai_api_key.get().strip()
            if not api_key:
                messagebox.showerror("Error", "Please enter your OpenAI API key.")
                return
            # List models from OpenAI
            try:
                import openai
                openai.api_key = api_key
                models = openai.models.list()
                self.models = [m.id for m in models.data if 'gpt' in m.id]
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load OpenAI models: {e}")
                return
        else:
            url = self.lmstudio_url.get().strip().rstrip('/')
            if not url:
                messagebox.showerror("Error", "Please enter the LM Studio endpoint URL.")
                return
            try:
                resp = requests.get(f"{url}/v1/models")
                resp.raise_for_status()
                data = resp.json()
                self.models = [m['id'] for m in data.get('data', [])]
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load LM Studio models: {e}")
                return
        # Update model dropdown
        menu = self.opt_models['menu']
        menu.delete(0, 'end')
        for model in self.models:
            menu.add_command(label=model, command=tk._setit(self.selected_model, model))
        if self.models:
            self.selected_model.set(self.models[0])
        else:
            self.selected_model.set("")

    def ask_question(self):
        self.btn_ask.config(state=tk.DISABLED, text="Processing...")
        self.root.update_idletasks()
        question = self.entry_query.get().strip()
        if not question:
            messagebox.showerror("Error", "Please enter a question.")
            self.btn_ask.config(state=tk.NORMAL, text="Ask")
            return
        if not os.path.exists(CHAT_DB_DEST):
            messagebox.showerror("Error", "chat.db not found in project directory. Please copy it first.")
            self.btn_ask.config(state=tk.NORMAL, text="Ask")
            return
        provider = self.llm_provider.get()
        model = self.selected_model.get()
        if not model:
            messagebox.showerror("Error", "Please select a model.")
            self.btn_ask.config(state=tk.NORMAL, text="Ask")
            return
        # --- Add user message to chat history (raw question only) ---
        self.chat_history.append({"role": "user", "content": question})
        # --- Compose prompt ---
        prompt = self.compose_prompt(question)
        sql = None
        try:
            import json
            if provider == "OpenAI":
                import openai
                client = openai.OpenAI(api_key=self.openai_api_key.get().strip())
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert at generating SQLite SQL queries for the iMessage chat.db. Only output the SQL query, no explanation. Return your answer as a JSON object with a single key 'sql', like this: {\"sql\": \"SELECT ...\"}"}
                    ] + self.chat_history,
                    max_tokens=500,
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                # Parse JSON response
                content = response.choices[0].message.content.strip()
                # Robustly extract the first JSON object in the response
                json_start = content.find('{')
                json_end = content.rfind('}')
                sql = ""
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str = content[json_start:json_end+1]
                    try:
                        sql_json = json.loads(json_str)
                        sql = sql_json.get('sql', '').strip()
                    except Exception:
                        sql = json_str  # fallback
                else:
                    sql = content  # fallback
            else:
                url = self.lmstudio_url.get().strip().rstrip('/')
                resp = requests.post(f"{url}/v1/chat/completions", json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are an expert at generating SQLite SQL queries for the iMessage chat.db. Only output the SQL query, no explanation. Return your answer as a JSON object with a single key 'sql', like this: {\"sql\": \"SELECT ...\"}"}
                    ] + self.chat_history,
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
                # Robustly extract the first JSON object in the response
                json_start = content.find('{')
                json_end = content.rfind('}')
                sql = ""
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str = content[json_start:json_end+1]
                    try:
                        sql_json = json.loads(json_str)
                        sql = sql_json.get('sql', '').strip()
                    except Exception:
                        sql = json_str  # fallback
                else:
                    sql = content  # fallback
        except Exception as e:
            self.txt_results.insert(tk.END, f"Error calling LLM: {e}\n")
            self.btn_ask.config(state=tk.NORMAL, text="Ask")
            return
        # --- Add assistant message to chat history (raw SQL only) ---
        self.chat_history.append({"role": "assistant", "content": sql})
        # --- Run SQL on chat.db ---
        result_text = None
        try:
            conn = sqlite3.connect(CHAT_DB_DEST)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            result_text = self.format_results(columns, rows)
            conn.close()
        except Exception as e:
            result_text = f"Error running SQL: {e}"
        # --- Add result to chat history ---
        self.chat_history.append({"role": "result", "content": result_text})
        # --- Display full conversation ---
        self.display_conversation()
        self.btn_ask.config(state=tk.NORMAL, text="Ask")
        return
def compose_prompt(self, question):
    # Load schema
    schema_path = os.path.join(os.path.dirname(__file__), "dbinfo.schema")
    example_path = os.path.join(os.path.dirname(__file__), "imessage-counts.sql")
    try:
        with open(schema_path, 'r') as f:
            schema = f.read()
    except Exception:
        schema = "[Could not load schema]"
    try:
        with open(example_path, 'r') as f:
            example_query = f.read()
    except Exception:
        example_query = "[Could not load example query]"
    system_prompt = (
        "You are an expert SQLite query generator. Only output a valid SQLite SQL query, with no explanation or commentary. "
        "The database is Apple's iMessage chat.db. Use only tables and columns that exist in this schema. "
        "Here is the schema:\n" + schema + "\n"
        "Here is an example query:\n" + example_query + "\n"
        "User will ask a question about the data. Generate a single SQL query that answers it. "
        "Return your answer as a JSON object with a single key 'sql', like this: {\"sql\": \"SELECT ...\"}"
    )
    prompt = f"{system_prompt}\nQuestion: {question}"
    return prompt

    def new_chat(self):
        self.txt_results.delete('1.0', tk.END)
        self.chat_history = []

    def display_conversation(self):
        self.txt_results.delete('1.0', tk.END)
        i = 0
        while i < len(self.chat_history):
            msg = self.chat_history[i]
            if msg['role'] == 'user':
                self.txt_results.insert(tk.END, f"User: {msg['content']}\n")
                if i+1 < len(self.chat_history) and self.chat_history[i+1]['role'] == 'assistant':
                    self.txt_results.insert(tk.END, f"SQL: {self.chat_history[i+1]['content']}\n")
                    if i+2 < len(self.chat_history) and self.chat_history[i+2]['role'] == 'result':
                        self.txt_results.insert(tk.END, f"Result:\n{self.chat_history[i+2]['content']}\n")
                        i += 3
                        continue
                    i += 2
                    continue
            i += 1

    def load_env(self):
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            api_key = os.getenv('OPENAI_API_KEY', '')
            lmstudio_url = os.getenv('LMSTUDIO_URL', '')
            if api_key:
                self.openai_api_key.set(api_key)
            if lmstudio_url:
                self.lmstudio_url.set(lmstudio_url)

    def save_env(self):
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if not os.path.exists(env_path):
            with open(env_path, 'w') as f:
                f.write('')
        set_key(env_path, 'OPENAI_API_KEY', self.openai_api_key.get())
        set_key(env_path, 'LMSTUDIO_URL', self.lmstudio_url.get())
        messagebox.showinfo("Saved", f"Credentials saved to {env_path}")

    def format_results(self, columns, rows):
        if not rows:
            return "No results."
        output = '\t'.join(columns) + '\n'
        for row in rows:
            output += '\t'.join(str(x) for x in row) + '\n'
        return output


def main():
    root = tk.Tk()
    app = IMessageDBApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
