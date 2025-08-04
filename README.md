# Sprinkle

**AI-powered bash command generator that converts natural language descriptions into executable shell commands**

## Abstract

Sprinkle is a Python-based command-line tool that leverages Google's Gemini AI to translate natural language descriptions into syntactically correct bash commands. It supports both simple command generation and template-based substitution for complex multi-step operations. The tool can either output commands for review or execute them directly, with an optional interactive editor for command refinement.

## Installation

### From PyPI (Recommended)

```bash
# Install with pip
pip install sprinkle

# Or install with pipx (recommended for CLI tools)
pipx install sprinkle
```

### Prerequisites

- Python 3.12 or higher
- Google Gemini API key

### Setup

1. Set up your Google Gemini API key:
```bash
export GOOGLE_API_KEY="your-gemini-api-key-here"
```

### Development Installation

If you want to contribute or modify the code:

1. Clone the repository:
```bash
git clone <repository-url>
cd sprinkle
```

2. Install in development mode:
```bash
pip install -e .
```

## Usage

### Basic Command Generation

Convert natural language descriptions to bash commands:

```bash
# Generate and execute a command
sprinkle "list all files in current directory"

# Output command without executing (safe mode)
sprinkle -o "find all python files larger than 1kb"

# Verbose output to see the AI reasoning process
sprinkle -v -o "count lines in all python files"
```

### Template-Based Commands

Use `{{}}` syntax for complex multi-step operations:

```bash
# Generate compound commands with AI substitution
sprinkle -o "{{find all python files}} and {{count the lines}}"

# More complex example
sprinkle -o "{{create backup directory}} then {{copy all config files to backup}}"
```

### Interactive Editing

Use the editor mode to review and modify commands before execution:

```bash
# Open interactive editor before execution
sprinkle -e "compress all log files older than 30 days"
```

### Command Line Options

- `-h, --help`: Show help message and exit
- `-v, --verbose`: Enable verbose output to see AI processing steps
- `-o, --output`: Output commands to stdout instead of executing them
- `-e, --editor`: Open interactive editor to modify commands before execution

### Examples

#### File Operations
```bash
# List files with details
sprinkle -o "show detailed information about all files including hidden ones"
# Output: ls -la

# Find and process files
sprinkle -o "find all JSON files and validate their syntax"
# Output: find . -name "*.json" -exec python -m json.tool {} \; > /dev/null

# Backup with timestamp
sprinkle -o "create a backup of the config directory with current timestamp"
# Output: cp -r config config_backup_$(date +%Y%m%d_%H%M%S)
```

#### System Administration
```bash
# Process management
sprinkle -o "kill all processes containing the word nginx"
# Output: pkill -f nginx

# Disk usage analysis
sprinkle -o "show disk usage of all directories sorted by size"
# Output: du -sh */ | sort -hr
```

#### Template-Based Complex Operations
```bash
# Multi-step file processing
sprinkle -o "{{find all log files older than 7 days}} and {{compress them with gzip}}"
# Output: find . -name "*.log" -mtime +7 and gzip

# Conditional operations
sprinkle -o "{{check if docker is running}} and {{start nginx container if it exists}}"
# Output: systemctl is-active docker and docker start nginx 2>/dev/null || echo "nginx container not found"
```

## Features

- **Natural Language Processing**: Convert plain English descriptions to bash commands
- **Template Substitution**: Use `{{}}` syntax for complex multi-part commands
- **Safety Mode**: Output commands for review before execution (`-o` flag)
- **Interactive Editing**: Modify generated commands before execution (`-e` flag)
- **Verbose Mode**: See AI reasoning and processing steps (`-v` flag)
- **Error Handling**: Automatic inclusion of appropriate error handling and safety flags
- **Command Chaining**: Intelligent use of `&&`, `;`, `|`, and `||` operators
- **POSIX Compatibility**: Generates portable commands when possible

## Safety Notes

- Always use the `-o` flag first to review generated commands before execution
- The tool can execute potentially destructive commands - exercise caution
- Set appropriate file permissions and run in appropriate directories
- Consider using the interactive editor (`-e`) for complex operations

## Requirements

The application requires the following Python packages:
- `langchain>=0.3.26`
- `langchain-google-genai>=1.0.0`
- `pydantic>=2.11.7`
- `textual>=5.2.0`

## Environment Variables

- `GOOGLE_API_KEY`: Required. Your Google Gemini API key for AI command generation.

## License

This project is licensed under the terms specified in the project configuration.