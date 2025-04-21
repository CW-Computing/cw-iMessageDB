<p align="center">
  <h1>iMessageDB LLM Query Tool 🚀</h1>
</p>
<div align="center">
  <img src="assets/icon.png" width="128" height="128" alt="iMessageDB Query Tool Icon">
</div>

> Ever wished you could have a conversation with your iMessage history? Now you can! Ask natural language questions about your messages and get instant insights.

![iMessageDB LLM Query Tool](/assets/mainapp.png)

## ✨ Features

Ask questions like:
- 💬 "What were we talking about last summer?"
- 📊 "Who sends me the most messages?"
- 🗓️ "Show me all conversations about dinner plans from March"
- 🔍 "Find all messages containing links from the past month"
- 📈 "How many messages did I send in 2023?"
- 🤔 "Summarize my conversation with John from yesterday"
- 🌟 "What were the main topics we discussed last week?"
- 📱 "Show me all messages about planning our vacation"
- 🕒 "When was the last time we talked about the project?"
- 📝 "Give me a summary of all work-related discussions"

## 📁 Project Structure

```
imessagedb-llm/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── .env                # Credentials file (not in git)
├── .gitignore         # Git ignore file
├── README.md          # This file
├── assets/            # Application assets
│   └── mainapp.png    # Application screenshot
├── saved_chats/       # Persistent chat storage
└── archive/           # Reference implementations
    ├── main2.py       # Alternative version
    ├── mainv3.py      # Previous version
    └── sqlcontext/    # SQL query examples and context
```

## 🚀 Getting Started

### Prerequisites
- macOS (access to iMessage database)
- Python 3.8+
- OpenAI API key or LM Studio setup
- Required Python packages (installed via requirements.txt)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/imessagedb-llm.git
   cd imessagedb-llm
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Configure credentials
   The app will automatically create and manage the `.env` file for you:
   1. Launch the app: `python main.py`
   2. Enter your credentials in the GUI
   3. Click "Save Credentials" button
   
   Alternatively, you can manually create the `.env` file (optional):
   ```ini
   # OpenAI Configuration
   OPENAI_API_KEY=your_api_key_here
   
   # LM Studio Configuration
   LMSTUDIO_URL=http://localhost:1234/v1
   
   # Last Used Configuration
   LAST_PROVIDER=OpenAI
   LAST_MODEL=gpt-4
   ```

4. Run the application
   ```bash
   python main.py
   ```

### Credential Management

The application uses a `.env` file for secure credential storage:

- 🔄 `.env` is automatically created when you click "Save Credentials"
- ✅ `.env` is automatically added to `.gitignore`
- 🔒 Credentials are never committed to version control
- 💾 Credentials persist between sessions
- 🔑 Supports both OpenAI API keys and LM Studio URLs
- 🛡️ Manual creation of `.env` is optional

First time setup:
1. Launch the application
2. Enter your OpenAI API key or LM Studio URL
3. Click "Save Credentials"
4. The app will automatically create and manage the `.env` file

Example `.gitignore`:
```
# Credentials and environment
.env
*.env

# Database
chat.db

# Saved chats
saved_chats/

# Python
__pycache__/
*.py[cod]
*$py.class
```

## 🛠️ Technical Overview

### Architecture

The iMessageDB LLM Query Tool is a sophisticated application that combines:
- Local iMessage database access
- Large Language Model (LLM) integration
- SQLite query generation
- Real-time message processing
- Dynamic chat management

### Core Components

1. **Database Integration**
   - Safely copies and interacts with Apple's `chat.db` (SQLite database)
   - Located at `~/Library/Messages/chat.db`
   - Handles complex message schema including attributed body content
   - Real-time database status monitoring and validation

2. **LLM Integration**
   - Dual provider support:
     - OpenAI API (GPT-4, GPT-3.5)
     - LM Studio (local models)
   - Automatic model loading and validation
   - Provider-specific optimizations
   - Credential management and persistence

3. **Message Processing**
   - Advanced binary blob decoding for `attributedBody` content
   - UTF-8 text extraction with error handling
   - Regular expression-based text cleaning
   - Timestamp conversion (nanoseconds since 2001 to human-readable format)
   - Contact information normalization

4. **Query Generation**
   - Natural language to SQL translation
   - Complex date handling (epoch offset calculations)
   - Intelligent query type detection:
     - Message content queries
     - Aggregate statistics
     - Time-based queries
     - Summary requests

5. **Chat Management**
   - Persistent chat history using JSON storage
   - Automatic chat naming using LLM
   - Real-time updates and synchronization
   - Chat import/export capabilities

### Key Technologies

- **Python 3.x**
  - Core application language
  - SQLite3 database interaction
  - Binary data processing

- **Tkinter**
  - Custom-themed GUI components
  - Responsive layout management
  - Real-time updates

- **SQLite**
  - Complex JOIN operations
  - Date/time calculations
  - Efficient query execution

- **API Integration**
  - RESTful API communication
  - JSON response parsing
  - Error handling and retry logic

### Data Processing

1. **Message Extraction**
```python
# Example of how messages are processed
def extract_clean_text_from_blob(blob):
    """
    Extracts clean text from binary attributedBody
    Handles NSString encoding and cleanup
    """
    raw = blob.decode('utf-8', errors='ignore')
    if 'NSString' in raw:
        # Complex NSString parsing logic
        # Handles various iOS message formats
```

2. **Date Handling**
```sql
-- Example of date conversion in queries
datetime(message.date/1000000000 + 978307200, 'unixepoch')
-- Converts Apple's nanoseconds since 2001 to readable dates
```

3. **Query Generation**
```sql
-- Example of a complex message query
SELECT 
    message.date,
    handle.id AS contact,
    message.is_from_me,
    message.text,
    message.attributedBody
FROM message
LEFT JOIN handle ON message.handle_id = handle.rowid
WHERE datetime(message.date/1000000000 + 978307200, 'unixepoch') 
    BETWEEN datetime('now', '-7 days') AND datetime('now')
ORDER BY message.date DESC
```

## 🔧 Advanced Usage

### Custom Query Examples

1. **Complex Date Ranges**
   ```sql
   -- Messages from summer months across years
   WHERE strftime('%m', datetime(date/1000000000 + 978307200, 'unixepoch')) 
   BETWEEN '06' AND '08'
   ```

2. **Contact Analysis**
   ```sql
   -- Most active conversation times
   SELECT 
       strftime('%H', datetime(date/1000000000 + 978307200, 'unixepoch')) as hour,
       COUNT(*) as message_count
   FROM message
   GROUP BY hour
   ORDER BY message_count DESC
   ```

### Customization
- Custom color schemes in GUI
- Configurable message formatting
- Adjustable summarization parameters
- Extensible provider system

## 📚 Technical Documentation

### Database Schema
- `message` table: Core message content
- `handle` table: Contact information
- `chat` table: Conversation grouping
- `message_attachment_join`: Media attachments
- `attachment` table: Attachment metadata

### Message Processing Pipeline
1. Query Generation
2. Database Interaction
3. Binary Data Extraction
4. Text Cleaning
5. Formatting
6. (Optional) Summarization

### LLM Integration
- System prompts for query generation
- Context management
- Response parsing
- Error handling

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Apple's Messages app and database structure
- OpenAI and LM Studio for LLM capabilities
- SQLite for robust database management
- The Python community for excellent libraries

---

Made with ❤️ by Charles Wade