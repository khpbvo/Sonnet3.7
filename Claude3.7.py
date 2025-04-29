#!/usr/bin/env python3
"""
Claude Code Assistant CLI

This module provides an interactive command-line assistant for coding, context management,
and tool usage, powered by Anthropic Claude models.
"""
import argparse
import asyncio
import glob
import os
import sys
import tempfile
import difflib
from typing import Optional, Tuple

import anthropic
import anyio  # Import anyio library
import tiktoken  # Import tiktoken for token counting
from anthropic.types import MessageParam
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# Initialize console for rich text display
console = Console()


class ClaudeSonnetCodeAssistant:
    def get_token_toolbar(self):
        """Return the token counter string for the bottom toolbar."""
        percentage = (self.current_tokens / self.max_tokens) * 100
        color = "ansigreen"
        if percentage > 70:
            color = "ansiyellow"
        if percentage > 90:
            color = "ansired"
        return f"[{color}]Tokens: {self.current_tokens:,}/{self.max_tokens:,} ({percentage:.1f}%)[/{color}]"

    def __init__(
        self, api_key: Optional[str] = None, model: str = "claude-3-7-sonnet-20250219"
    ):
        """Initialize the Claude Code Assistant CLI."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set as ANTHROPIC_API_KEY environment variable"
            )

        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.current_dir = os.getcwd()
        self.context = []
        self.command_history = []

        # Initialize tiktoken encoder
        self.encoder = tiktoken.get_encoding(
            "cl100k_base"
        )  # Use get_encoding for encoding names
        self.max_tokens = 180000  # Max token threshold for auto-summarization
        self.current_tokens = 0  # Current token count

        # Diff settings
        self.max_diff_size = (
            50  # Maximum number of changed lines before auto-segmenting diffs
        )
        self.always_use_vim_style = (
            True  # Whether to always use the VIM-style diff viewer
        )

        # Command completions
        self.commands = [
            "code:read:",
            "code:read:files:",
            "code:changedir:",
            "code:listdir:",
            "code:generate:",
            "code:change:",
            "code:search:",
            "code:shell:",
            "help",
            "exit",
            "quit",
        ]

        # Create prompt session with history and bottom toolbar for token counter
        history_file = os.path.expanduser("~/.claude_code_assistant_history")
        self.session = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(self.commands, sentence=True),
            bottom_toolbar=self.get_token_toolbar,
        )

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encoder.encode(text))

    def update_token_count(self):
        """Recalculate the total token count from context."""
        self.current_tokens = 0
        for ctx in self.context:
            if ctx.get("type") == "file":
                self.current_tokens += self.count_tokens(ctx.get("content", ""))
                self.current_tokens += self.count_tokens(ctx.get("path", ""))
            else:
                # For other context items
                self.current_tokens += self.count_tokens(str(ctx))

        return self.current_tokens

    async def start(self):
        """Start the interactive CLI session."""
        console.print(
            Panel.fit(
                "[bold blue]Claude Sonnet Code Assistant[/bold blue]\n"
                f"[green]Using model: {self.model}[/green]\n"
                "Type [bold]help[/bold] for available commands or [bold]exit[/bold] to quit",
                title="Welcome",
                subtitle="v1.0",
            )
        )

        while True:
            try:
                # Get user input
                user_input = await self.session.prompt_async(
                    f"[{os.path.basename(self.current_dir)}] > "
                )
                user_input = user_input.strip()

                if not user_input:
                    continue

                # Handle exit commands
                if user_input.lower() in ["exit", "quit"]:
                    console.print("[yellow]Exiting Claude Code Assistant[/yellow]")
                    break

                # Process the command
                await self.process_command(user_input)

                # Check if we need to summarize
                if self.current_tokens > self.max_tokens:
                    await self.auto_summarize()

            except KeyboardInterrupt:
                console.print(
                    "\n[yellow]Interrupted. Press Ctrl+C again to exit.[/yellow]"
                )
                try:
                    await asyncio.sleep(1)
                except KeyboardInterrupt:
                    console.print("[yellow]Exiting Claude Code Assistant[/yellow]")
                    break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")

    async def process_command(self, command: str):
        """Process user commands."""
        # Add to command history
        self.command_history.append(command)

        # Help command
        if command.lower() == "help":
            self.show_help()
            return

        # Handle code: prefixed commands
        if command.startswith("code:"):
            parts = command.split(":", 2)

            if len(parts) < 2:
                console.print("[bold red]Invalid command format[/bold red]")
                return

            cmd_type = parts[1]

            # Process different command types
            if cmd_type == "read" and len(parts) > 2:
                # Check if it's the multi-file read command
                if parts[2].startswith("files:"):
                    file_list = parts[2][6:].strip()
                    await self.handle_read_multiple(file_list)
                else:
                    await self.handle_read(parts[2])
            elif cmd_type == "changedir" and len(parts) > 2:
                await self.handle_changedir(parts[2])
            elif cmd_type == "listdir" and len(parts) > 2:
                await self.handle_listdir(parts[2])
            elif cmd_type == "generate" and len(parts) > 2:
                file_prompt = parts[2].split(":", 1)
                if len(file_prompt) == 2:
                    await self.handle_generate(file_prompt[0], file_prompt[1])
                else:
                    console.print(
                        "[bold red]Invalid format. Use code:generate:file.py:prompt[/bold red]"
                    )
            elif cmd_type == "change" and len(parts) > 2:
                # Improved: allow colons in prompt
                first_colon = parts[2].find(":")
                if first_colon != -1:
                    file_path = parts[2][:first_colon]
                    prompt = parts[2][first_colon+1:]
                    await self.handle_change(file_path, prompt)
                else:
                    console.print(
                        "[bold red]Invalid format. Use code:change:file.py:prompt[/bold red]"
                    )
            elif cmd_type == "search" and len(parts) > 2:
                await self.handle_search(parts[2])
            elif cmd_type == "shell" and len(parts) > 2:
                await self.handle_shell(parts[2])
            else:
                console.print("[bold red]Unknown or invalid command[/bold red]")
        else:
            # Treat as a direct prompt to Claude
            await self.ask_claude(command)

    def show_help(self):
        help_text = """
        [bold]Available Commands:[/bold]
        
        [bold green]code:read:path/to/file[/bold green]
            Load a file as context for Claude
        
        [bold green]code:read:files:file1,file2,file3[/bold green]
            Load multiple files as context for Claude
        
        [bold green]code:changedir:/path/to/dir[/bold green]
            Change the current working directory
        
        [bold green]code:listdir:/path/to/dir[/bold green]
            List all files and directories in the specified path
        
        [bold green]code:generate:file.py:prompt[/bold green]
            Generate new code in the specified file based on the prompt
        
        [bold green]code:change:file.py:prompt[/bold green]
            Make changes to an existing file based on the prompt
        
        [bold green]code:search:pattern[/bold green]
            Search for files or code containing the pattern
        
        [bold green]code:shell:command[/bold green]
            Execute a shell command
        
        [bold green]help[/bold green]
            Show this help message
        
        [bold green]exit[/bold green] or [bold green]quit[/bold green]
            Exit the program
        
        You can also type any message to directly chat with Claude.
        
        [bold]File Editing:[/bold]
        File changes are now handled automatically by Claude's text-editor-tool. You do not need to review or apply diffs manually. Claude will request file operations and they will be executed directly.
        """
        console.print(Panel(help_text, title="Claude Code Assistant Help"))

    async def handle_read_multiple(self, file_list: str):
        """Read multiple files at once and add them to context."""
        # Split the comma-separated list of files
        files = [f.strip() for f in file_list.split(",")]

        if not files:
            console.print("[bold red]No files specified[/bold red]")
            return

        # Process each file
        for file_path in files:
            await self.handle_read(file_path)

    async def handle_read(self, file_path: str):
        try:
            resolved_path = await self.resolve_path(file_path)
            file_content = await self.read_file(resolved_path)
            if file_content:
                file_ext = os.path.splitext(file_path)[1].lstrip(".")
                syntax = Syntax(file_content, file_ext, line_numbers=True)
                console.print(Panel(syntax, title=f"File: {file_path}"))
                tokens_in_file = self.count_tokens(file_content)
                tokens_in_path = self.count_tokens(file_path)
                total_file_tokens = tokens_in_file + tokens_in_path
                console.print(f"[cyan]File contains {tokens_in_file:,} tokens[/cyan]")
                self.context.append(
                    {
                        "type": "file",
                        "path": file_path,
                        "content": file_content,
                        "tokens": total_file_tokens,
                    }
                )
                console.print(f"[green]Added {file_path} to context[/green]")
                self.current_tokens += total_file_tokens
        except Exception as e:
            console.print(f"[bold red]Error reading file:[/bold red] {str(e)}")

    async def handle_changedir(self, dir_path: str):
        """Change current working directory."""
        try:
            new_dir = await self.resolve_path(dir_path)
            if os.path.isdir(new_dir):
                os.chdir(new_dir)
                self.current_dir = os.getcwd()
                console.print(
                    f"[green]Changed directory to:[/green] {self.current_dir}"
                )
            else:
                console.print(
                    f"[bold red]Directory does not exist:[/bold red] {new_dir}"
                )
        except Exception as e:
            console.print(f"[bold red]Error changing directory:[/bold red] {str(e)}")

    async def handle_listdir(self, dir_path: str):
        """List files and directories in the specified path."""
        try:
            target_dir = await self.resolve_path(dir_path)
            if os.path.isdir(target_dir):
                items = os.listdir(target_dir)

                # Separate directories and files
                directories = []
                files = []

                for item in items:
                    full_path = os.path.join(target_dir, item)
                    if os.path.isdir(full_path):
                        directories.append(f"[bold blue]{item}/[/bold blue]")
                    else:
                        files.append(f"[green]{item}[/green]")

                # Sort and display
                directories.sort()
                files.sort()

                console.print(f"[bold]Contents of {target_dir}:[/bold]")
                for d in directories:
                    console.print(d)
                for f in files:
                    console.print(f)

                console.print(
                    f"[dim]{len(directories)} directories, {len(files)} files[/dim]"
                )
            else:
                console.print(
                    f"[bold red]Directory does not exist:[/bold red] {target_dir}"
                )
        except Exception as e:
            console.print(f"[bold red]Error listing directory:[/bold red] {str(e)}")

    async def handle_generate(self, file_path: str, prompt: str):
        """Request Claude to generate code for a file using the text-editor-tool."""
        # Just prompt Claude; let it use the tool-use protocol for file creation/edits
        await self.ask_claude(f"Please generate code for the file '{file_path}' with the following requirements: {prompt}. Use the text-editor-tool to create or edit the file as needed.")

    async def _handle_change_deprecated(self, file_path: str, prompt: str):
        """Deprecated implementation - kept for reference."""
        pass

    async def handle_editor_tool(self, tool_call):
        """Implements the Anthropic text_editor_20250124 commands."""
        try:
            params = tool_call.input if hasattr(tool_call, 'input') else tool_call.get('input', {})
            command = params.get('command')
            path = params.get('path')

            if not command:
                return "Error: No command specified."
            if not path:
                return "Error: No file path provided."

            console.print(f"[dim]Executing editor command: {command} on {path}[/dim]")

            abs_path = await self.resolve_path(path)

            # The rest of your handler implementation should work fine for the commands
            # view, str_replace, create, insert, append, undo_edit

            if command == 'view':
                with_line_numbers = params.get('with_line_numbers', False)
                if not os.path.exists(abs_path):
                    return f"Error: File {abs_path} does not exist."
                try:
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    if with_line_numbers:
                        return ''.join(f"{i+1}: {line}" for i, line in enumerate(lines))
                    return ''.join(lines)
                except UnicodeDecodeError:
                    # Try with a different encoding
                    with open(abs_path, 'r', encoding='latin-1') as f:
                        lines = f.readlines()
                    if with_line_numbers:
                        return ''.join(f"{i+1}: {line}" for i, line in enumerate(lines))
                    return ''.join(lines)

            elif command == 'str_replace':
                find = params.get('find', '')
                replace = params.get('replace', '')

                if not find:
                    return "Error: 'find' parameter is required for str_replace command."

                if not os.path.exists(abs_path):
                    return f"Error: File {abs_path} does not exist."

                try:
                    # Read the current content
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Create a backup before modifying
                    backup_dir = os.path.join(tempfile.gettempdir(), "claude_code_assistant_backups")
                    os.makedirs(backup_dir, exist_ok=True)
                    backup_path = os.path.join(backup_dir, f"{os.path.basename(abs_path)}.bak")
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    # Replace and write the new content
                    new_content = content.replace(find, replace)
                    with open(abs_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

                    # Return a helpful message indicating changes
                    occurrences = content.count(find)
                    return f"Replaced {occurrences} occurrence(s) of '{find}' with '{replace}' in {abs_path}. Backup created at {backup_path}."
                except Exception as e:
                    return f"Error replacing text: {str(e)}"

            elif command == 'create':
                text = params.get('text', '')

                try:
                    # Create directories if they don't exist
                    dir_path = os.path.dirname(abs_path)
                    if dir_path:
                        os.makedirs(dir_path, exist_ok=True)

                    # Write the file
                    with open(abs_path, 'w', encoding='utf-8') as f:
                        f.write(text)

                    return f"Created file {abs_path} with {len(text)} characters."
                except Exception as e:
                    return f"Error creating file: {str(e)}"

            elif command == 'insert':
                text = params.get('text', '')
                line = params.get('line', 1)
                col = params.get('col', 0)

                if not os.path.exists(abs_path):
                    return f"Error: File {abs_path} does not exist."

                try:
                    # Read the current content
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    # Create a backup before modifying
                    backup_dir = os.path.join(tempfile.gettempdir(), "claude_code_assistant_backups")
                    os.makedirs(backup_dir, exist_ok=True)
                    backup_path = os.path.join(backup_dir, f"{os.path.basename(abs_path)}.bak")
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)

                    # Insert the text at the specified position
                    if line > len(lines):
                        # If line is beyond the file, add empty lines as needed
                        lines.extend([''] * (line - len(lines)))
                        lines.append(text + '\n')
                    else:
                        # Adjust line number (1-based to 0-based)
                        line_idx = line - 1

                        # Ensure the line exists
                        if line_idx < 0:
                            line_idx = 0

                        # Get the original line
                        orig_line = lines[line_idx] if line_idx < len(lines) else ''

                        # Ensure the column is valid
                        col = max(0, min(col, len(orig_line)))

                        # Insert text
                        lines[line_idx] = orig_line[:col] + text + orig_line[col:]

                    # Write the modified content
                    with open(abs_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)

                    return f"Inserted {len(text)} characters at line {line}, column {col} in {abs_path}. Backup created at {backup_path}."
                except Exception as e:
                    return f"Error inserting text: {str(e)}"

            elif command == 'undo_edit':
                backup_path = params.get('backup_path')

                if not backup_path:
                    return "Error: 'backup_path' parameter is required for undo_edit command."

                backup_path_resolved = await self.resolve_path(backup_path)

                if not os.path.exists(backup_path_resolved):
                    return f"Error: Backup file {backup_path_resolved} does not exist."

                try:
                    # Restore from backup
                    with open(backup_path_resolved, 'r', encoding='utf-8') as src, open(abs_path, 'w', encoding='utf-8') as dst:
                        content = src.read()
                        dst.write(content)

                    return f"Restored {abs_path} from backup {backup_path_resolved}."
                except Exception as e:
                    return f"Error restoring from backup: {str(e)}"

            elif command == 'append':
                # New command to append text to the end of a file
                text = params.get('text', '')

                if not os.path.exists(abs_path):
                    return f"Error: File {abs_path} does not exist."

                try:
                    # Create a backup
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    backup_dir = os.path.join(tempfile.gettempdir(), "claude_code_assistant_backups")
                    os.makedirs(backup_dir, exist_ok=True)
                    backup_path = os.path.join(backup_dir, f"{os.path.basename(abs_path)}.bak")

                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    # Append the text
                    with open(abs_path, 'a', encoding='utf-8') as f:
                        f.write(text)

                    return f"Appended {len(text)} characters to {abs_path}. Backup created at {backup_path}."
                except Exception as e:
                    return f"Error appending text: {str(e)}"

            else:
                return f"Error: Unknown command '{command}'. Available commands: view, str_replace, create, insert, append, undo_edit."

        except Exception as e:
            return f"Error in text editor tool: {str(e)}"

    async def handle_search(self, query: str):
        """
        The function contains methods for handling search, shell commands, interacting with a
        text-editor-tool, file changes, generating summaries, and creating colored diffs.
        
        :param query: The `query` parameter in the `handle_search` method is used to search for files or
        code based on the input provided. The method splits the query into a search type and search query,
        then performs the search based on the type specified (either "file" or "code"). If no type
        :type query: str
        """
        """Search for files or code."""
        try:
            if ":" in query:
                search_type, search_query = query.split(":", 1)
                if search_type == "file":
                    await self.search_file(search_query)
                elif search_type == "code":
                    await self.search_code(search_query)
                else:
                    console.print(
                        f"[bold red]Unknown search type:[/bold red] {search_type}"
                    )
            else:
                # Default to searching both files and code
                console.print("[bold]Searching for files:[/bold]")
                await self.search_file(query)
                console.print("\n[bold]Searching in code:[/bold]")
                await self.search_code(query)
        except Exception as e:
            console.print(f"[bold red]Error during search:[/bold red] {str(e)}")

    async def handle_shell(self, cmd: str):
        """Execute a shell command."""
        try:
            console.print(f"[dim]Executing: {cmd}[/dim]")

            # Execute the command
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.current_dir,
            )

            stdout, stderr = await process.communicate()

            # Display output
            if stdout:
                stdout_str = stdout.decode()
                console.print(
                    Panel(stdout_str, title="Standard Output", border_style="green")
                )

            if stderr:
                stderr_str = stderr.decode()
                console.print(
                    Panel(stderr_str, title="Standard Error", border_style="red")
                )

            console.print(f"[dim]Command exited with code {process.returncode}[/dim]")

        except Exception as e:
            console.print(
                f"[bold red]Error executing shell command:[/bold red] {str(e)}"
            )

    async def ask_claude(self, prompt: str):
        """Send a direct prompt to Claude, handling tool_use blocks for the text-editor-tool."""
        try:
            # Prepare context based on loaded files
            context_str = ""
            for ctx in self.context[-5:]:  # Limit to most recent 5 files
                if ctx.get("type") == "file":
                    path = ctx.get("path", "")
                    content = ctx.get("content", "")
                    ext = os.path.splitext(path)[1].lstrip(".")
                    context_str += f"File: {path}\n```{ext}\n{content}\n```\n\n"

            # Create the full prompt
            full_prompt = (
                f"{context_str}Working directory: {self.current_dir}\n\n{prompt}"
            )
            # Display an animated thinking indicator
            thinking_styles = [
                "[bold blue]Thinking...[/bold blue]",
                "[bold green]Thinking...[/bold green]",
                "[bold yellow]Thinking...[/bold yellow]",
                "[bold magenta]Thinking...[/bold magenta]",
                "[bold cyan]Thinking...[/bold cyan]"
            ]
            stop_thinking = asyncio.Event()
            thinking_task = asyncio.create_task(self._animate_thinking(thinking_styles, stop_thinking))
            
            modified_files = set()
            
            # Use the proper tool name according to Anthropic docs: "text_editor_20250124"
            response = await self.client.messages.create(
                model=self.model,
                messages=[MessageParam(role="user", content=full_prompt)],
                max_tokens=4000,
                stream=False,  # Non-streaming
                tools=[{
                    "name": "text_editor_20250124",
                    "description": "A tool for viewing and editing files on disk.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "path": {"type": "string"},
                            "text": {"type": "string"},
                            "find": {"type": "string"},
                            "replace": {"type": "string"},
                            "line": {"type": "integer"},
                            "col": {"type": "integer"},
                            "backup_path": {"type": "string"},
                            "with_line_numbers": {"type": "boolean"}
                        },
                        "required": ["command", "path"]
                    }
                }]
            )
            
            stop_thinking.set()
            await thinking_task
            console.print("\r" + " " * 60 + "\r", end="")
            console.print("\n[bold red]Assistant:[/bold red] ", end="")
            
            full_response = ""
            
            # Handle any tool uses
            if hasattr(response, 'tool_uses') and response.tool_uses:
                for tool_use in response.tool_uses:
                    if tool_use.name == "text_editor_20250124":
                        cmd, path = tool_use.input.get('command', ''), tool_use.input.get('path', '')
                        console.print(f"\n[dim]Using text editor tool: {cmd} on {path}[/dim]")
                        result = await self.handle_editor_tool(tool_use)
                        console.print(f"[dim]{result.split(os.linesep)[0]}...[/dim]")
                        if cmd in ['str_replace','create','insert','append'] and path:
                            modified_files.add(path)
                        # send tool result back
                        follow = await self.client.messages.create(
                            model=self.model,
                            messages=[
                                {"role": "user", "content": [
                                    {"type": "text", "text": full_prompt},
                                    {"type": "tool_result", "tool_use_id": tool_use.id, "content": result}
                                ]}
                            ],
                            max_tokens=4000,
                            stream=False,
                            tools=[{
                                "name": "text_editor_20250124",
                                "description": "A tool for viewing and editing files on disk.",
                                "input_schema": {
                                    "type": "object",
                                    "properties": {
                                        "command": {"type": "string"},
                                        "path": {"type": "string"},
                                        "text": {"type": "string"},
                                        "find": {"type": "string"},
                                        "replace": {"type": "string"},
                                        "line": {"type": "integer"},
                                        "col": {"type": "integer"},
                                        "backup_path": {"type": "string"},
                                        "with_line_numbers": {"type": "boolean"}
                                    },
                                    "required": ["command", "path"]
                                }
                            }]
                        )
                        # nested uses
                        if hasattr(follow, 'tool_uses') and follow.tool_uses:
                            for nested in follow.tool_uses:
                                if nested.name == "text_editor_20250124":
                                    ncmd, npath = nested.input.get('command', ''), nested.input.get('path', '')
                                    console.print(f"\n[dim]Nested tool: {ncmd} on {npath}[/dim]")
                                    nres = await self.handle_editor_tool(nested)
                                    console.print(f"[dim]{nres.split(os.linesep)[0]}...[/dim]")
                                    if ncmd in ['str_replace', 'create', 'insert', 'append'] and npath:
                                        modified_files.add(npath)
                                    await self.client.messages.create(
                                        model=self.model,
                                        messages=[{"role": "user", "content": [
                                            {"type": "text", "text": full_prompt},
                                            {"type": "tool_result", "tool_use_id": nested.id, "content": nres}
                                        ]}],
                                        max_tokens=4000,
                                        stream=False
                                    )
                for block in getattr(follow, 'content', []):
                    txt = getattr(block, 'text', None)
                    if txt:
                        full_response += txt
                        console.print(txt, highlight=False)
            # fallback response
            if not full_response and hasattr(response, 'content'):
                for blk in response.content:
                    txt = getattr(blk, 'text', None)
                    if txt:
                        full_response += txt
                        console.print(txt, highlight=False)
            console.print()
            # update context
            if modified_files:
                console.print("[yellow]Updating context with modified files...[/yellow]")
                for fpath in modified_files:
                    abs = fpath and await self.resolve_path(fpath)
                    if abs and os.path.exists(abs):
                        cont = await self.read_file(abs)
                        # update or add
                        for i, ctx in enumerate(self.context):
                            if ctx.get('type') == 'file' and ctx.get('path') == fpath:
                                old = ctx.get('tokens', 0)
                                tok = self.count_tokens(cont) + self.count_tokens(fpath)
                                self.context[i] = {'type': 'file', 'path': fpath, 'content': cont, 'tokens': tok}
                                self.current_tokens = self.current_tokens - old + tok
                                console.print(f"[green]Updated {fpath}[/green]")
                                break
                        else:
                            tok = self.count_tokens(cont) + self.count_tokens(fpath)
                            self.context.append({'type': 'file', 'path': fpath, 'content': cont, 'tokens': tok})
                            self.current_tokens += tok
                            console.print(f"[green]Added {fpath}[/green]")
        except Exception as e:
            console.print(f"[bold red]Error asking Claude:[/bold red] {str(e)}")

    async def handle_change(self, file_path: str, prompt: str):
        """Request Claude to change a file using the text-editor-tool."""
        try:
            # First, ensure the file is in context by loading it
            resolved_path = await self.resolve_path(file_path)
            if not os.path.exists(resolved_path):
                console.print(f"[bold red]File does not exist:[/bold red] {resolved_path}")
                return
            
            # Read the file content
            file_content = await self.read_file(resolved_path)
            if not file_content:
                console.print(f"[bold red]Could not read file:[/bold red] {resolved_path}")
                return
                
            # Add to context if not already there
            file_in_context = False
            for ctx in self.context:
                if ctx.get("type") == "file" and ctx.get("path") == file_path:
                    file_in_context = True
                    break
                    
            if not file_in_context:
                tokens_in_file = self.count_tokens(file_content)
                tokens_in_path = self.count_tokens(file_path)
                total_file_tokens = tokens_in_file + tokens_in_path
                self.context.append({
                    "type": "file",
                    "path": file_path,
                    "content": file_content,
                    "tokens": total_file_tokens,
                })
                self.current_tokens += total_file_tokens
                console.print(f"[green]Added {file_path} to context[/green]")
            
            # Send specific instructions to Claude to modify the file
            clear_prompt = (
                f"I need you to complete the code in the file '{file_path}' according to these requirements: {prompt}\n\n"
                f"IMPORTANT: You MUST use the text-editor-tool to make these changes. Follow these steps:\n"
                f"1. First view the current content with the 'view' command\n"
                f"2. Analyze what changes are needed\n"
                f"3. Use the appropriate commands like 'str_replace', 'insert', or 'create' to implement the changes\n"
                f"4. After making changes, use 'view' again to confirm your changes were applied correctly\n"
                f"5. Summarize the changes you made"
            )
            
            await self.ask_claude(clear_prompt)
            
        except Exception as e:
            console.print(f"[bold red]Error handling change request:[/bold red] {str(e)}")

    async def auto_summarize(self):
        console.print(
            f"[yellow]Token limit ({self.max_tokens:,}) exceeded. Summarizing files...[/yellow]"
        )
        files_to_summarize = []
        for ctx in self.context:
            if ctx.get("type") == "file" and not ctx.get("is_summary", False):
                files_to_summarize.append(ctx)
        files_to_summarize.sort(key=lambda x: x.get("tokens", 0), reverse=True)
        for file_ctx in files_to_summarize:
            if self.current_tokens <= self.max_tokens * 0.7:
                break
            file_path = file_ctx.get("path", "")
            file_content = file_ctx.get("content", "")
            file_tokens = file_ctx.get("tokens", 0)
            console.print(
                f"[yellow]Summarizing {file_path} ({file_tokens:,} tokens)[/yellow]"
            )
            summary = await self.generate_file_summary(file_path, file_content)
            summary_tokens = self.count_tokens(summary)
            path_tokens = self.count_tokens(file_path)
            total_summary_tokens = summary_tokens + path_tokens
            for i, ctx in enumerate(self.context):
                if ctx is file_ctx:
                    self.current_tokens -= file_tokens
                    self.context[i]["content"] = summary
                    self.context[i]["tokens"] = total_summary_tokens
                    self.context[i]["is_summary"] = True
                    self.context[i]["original_tokens"] = file_tokens
                    self.current_tokens += total_summary_tokens
                    break
            console.print(
                f"[green]Summarized {file_path}: {file_tokens:,} → {total_summary_tokens:,} tokens ({(total_summary_tokens/file_tokens)*100:.1f}%)[/green]"
            )
            if self.current_tokens <= self.max_tokens * 0.7:
                break
        console.print(
            f"[green]Finished summarizing. Current token count: {self.current_tokens:,}[/green]"
        )

    async def generate_file_summary(self, file_path: str, content: str) -> str:
        """Generate a summary of a file using Claude."""
        try:
            file_ext = os.path.splitext(file_path)[1].lstrip(".")

            messages = [
                MessageParam(
                    role="user",
                    content=f"Please summarize the following {file_ext} file so I can understand its structure and functionality. Focus on key elements that would be important for understanding how to use or modify this code. Keep implementation details brief.\n\n```{file_ext}\n{content}\n```\n\nProvide a concise summary that preserves the most important information about this file.",
                )
            ]

            response = await self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
            )

            summary = response.content[0].text

            # Format as a code summary
            formatted_summary = f"# SUMMARY OF {file_path}\n\n{summary}\n\n# Original file was {self.count_tokens(content):,} tokens and has been summarized."

            return formatted_summary

        except Exception as e:
            console.print(f"[bold red]Error generating summary:[/bold red] {str(e)}")
            return f"# ERROR: Failed to summarize {file_path}\n\n{str(e)}"

    async def read_file(self, file_path: str) -> str:
        """Read a file asynchronously."""
        try:
            file_path = os.path.expanduser(file_path)
            file = await anyio.open_file(file_path, "r")
            async with file as f:
                return await f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                file = await anyio.open_file(file_path, "r", encoding="latin-1")
                async with file as f:
                    return await f.read()
            except Exception as e:
                console.print(
                    f"[bold red]Could not read file (binary file?):[/bold red] {str(e)}"
                )
                return ""
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

    async def search_file(self, pattern: str):
        """Search for files matching a pattern."""
        try:
            # Check if it's a glob pattern
            if any(char in pattern for char in "*?[]"):
                matches = glob.glob(
                    os.path.join(self.current_dir, "**", pattern), recursive=True
                )
            else:
                # Use find command for more complex searches
                cmd = f"find {self.current_dir} -type f -name '*{pattern}*' 2>/dev/null"
                proc = await asyncio.create_subprocess_shell(
                    cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()
                matches = stdout.decode().strip().split("\n")
                matches = [m for m in matches if m]  # Filter empty strings

            if matches:
                console.print(
                    f"[green]Found {len(matches)} file(s) matching '{pattern}':[/green]"
                )
                for match in matches:
                    rel_path = os.path.relpath(match, self.current_dir)
                    console.print(f"  {rel_path}")
            else:
                console.print(f"[yellow]No files found matching '{pattern}'[/yellow]")

        except Exception as e:
            console.print(f"[bold red]Error searching for files:[/bold red] {str(e)}")

    async def search_code(self, pattern: str):
        """Search for code containing a pattern."""
        try:
            # Use grep for code search
            cmd = f"grep -r --include='*.*' '{pattern}' {self.current_dir} 2>/dev/null"
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            results = stdout.decode().strip().split("\n")
            results = [r for r in results if r]  # Filter empty strings

            if results:
                console.print(
                    f"[green]Found {len(results)} match(es) for '{pattern}' in code:[/green]"
                )
                for result in results:
                    parts = result.split(":", 1)
                    if len(parts) >= 2:
                        file_path = parts[0]
                        content = parts[1]
                        rel_path = os.path.relpath(file_path, self.current_dir)
                        console.print(f"[bold]{rel_path}[/bold]: {content}")
            else:
                console.print(f"[yellow]No code matches found for '{pattern}'[/yellow]")

        except Exception as e:
            console.print(f"[bold red]Error searching code:[/bold red] {str(e)}")

    async def create_colored_diff(
        self, original: str, new: str, file_path: str, enhanced: bool = True
    ) -> Panel:
        """Create a colored unified diff with explicit line numbers."""
        try:
            # Create temporary files for diffing
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as old_file:
                old_file.write(original)
                old_file_path = old_file.name

            with tempfile.NamedTemporaryFile(mode="w", delete=False) as new_file:
                new_file.write(new)
                new_file_path = new_file.name

            try:
                # Use the 'diff' command to create a unified diff with more context lines and line numbers
                # -u shows unified diff format
                # -N treats absent files as empty
                # -p shows the function name for each change
                # -U3 shows 3 lines of unified context
                proc = await asyncio.create_subprocess_exec(
                    "diff",
                    "-u",
                    "-N",
                    "-p",
                    "-U3",
                    old_file_path,
                    new_file_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await proc.communicate()
                diff_output = stdout.decode()

                # Parse the diff to extract hunk information and add explicit line numbers
                diff_lines = diff_output.splitlines()
                enhanced_diff = []

                # Track current line numbers in both files
                old_line_num = 0
                new_line_num = 0
                in_hunk = False

                for line in diff_lines:
                    if line.startswith("---") or line.startswith("+++"):
                        # File headers
                        enhanced_diff.append(line)
                    elif line.startswith("@@"):
                        # Extract line numbers from hunk headers
                        # Format: @@ -old_start,old_count +new_start,new_count @@
                        in_hunk = True
                        parts = line.split(" ")
                        old_info = parts[1][1:]  # Remove the leading '-'
                        new_info = parts[2][1:]  # Remove the leading '+'

                        if "," in old_info:
                            old_start = int(old_info.split(",")[0])
                        else:
                            old_start = int(old_info)

                        if "," in new_info:
                            new_start = int(new_info.split(",")[0])
                        else:
                            new_start = int(new_info)

                        old_line_num = old_start
                        new_line_num = new_start

                        # Add a more prominent hunk header with better line number info
                        if enhanced:
                            enhanced_diff.append(f"[bold cyan]{line}[/bold cyan]")
                            enhanced_diff.append(f"[bold cyan]--- Lines {old_start}+ in original[/bold cyan]")
                            enhanced_diff.append(f"[bold cyan]+++ Lines {new_start}+ in new version[/bold cyan]")
                        else:
                            enhanced_diff.append(f"[bold cyan]{line}[/bold cyan]")
                    else:
                        # Content lines with explicit line numbers
                        if line.startswith("+"):
                            prefix = f"{new_line_num:4d}+ " if enhanced else ""
                            enhanced_diff.append(f"[green]{prefix}{line}[/green]")
                            new_line_num += 1
                        elif line.startswith("-"):
                            prefix = f"{old_line_num:4d}- " if enhanced else ""
                            enhanced_diff.append(f"[red]{prefix}{line}[/red]")
                            old_line_num += 1
                        else:
                            if in_hunk:
                                old_prefix = f"{old_line_num:4d}  " if enhanced else ""
                                new_prefix = f"{new_line_num:4d}  " if enhanced else ""
                                if enhanced:
                                    enhanced_diff.append(f"{old_prefix}{new_prefix} {line}")
                                else:
                                    enhanced_diff.append(line)
                                old_line_num += 1
                                new_line_num += 1
                            else:
                                enhanced_diff.append(line)

                diff_text = "\n".join(enhanced_diff)

                # Check if the changes are very large (more than 20 lines changed)
                plus_count = len([line for line in diff_lines if line.startswith("+")])
                minus_count = len([line for line in diff_lines if line.startswith("-")])
                total_changes = plus_count + minus_count

                if total_changes > 20:
                    diff_text += f"\n\n[bold yellow]Warning: This is a large change ({total_changes} lines). Consider breaking it into smaller edits for better review.[/bold yellow]"

                return Panel(diff_text, title=f"Diff for {file_path}", border_style="blue")

            finally:
                # Clean up temporary files
                os.unlink(old_file_path)
                os.unlink(new_file_path)

        except Exception as e:
            return Panel(f"[bold red]Error creating diff:[/bold red] {str(e)}", title=f"Diff Error for {file_path}", border_style="red")

    async def interactive_diff_review(
        self, original: str, new: str, file_path: str
    ) -> Tuple[bool, str]:
        """Provide an interactive VIM-like diff review experience.

        Returns:
            Tuple of (accepted, final_content)
        """
        try:
            # Create unified diff for display
            diff = await self.create_colored_diff(
                original, new, file_path, enhanced=True
            )
            console.print(diff)

            # Since a full VIM-like interface requires complex terminal handling,
            # we'll implement a simplified interactive review
            console.print(
                "\n[bold cyan]Interactive Diff Review (VIM-inspired)[/bold cyan]"
            )
            console.print(
                "Commands: a (accept all), r (reject all), s (skip to next diff chunk), v (view current chunk), q (quit)"
            )

            # Parse the original and new content into lines
            old_lines = original.splitlines()
            new_lines = new.splitlines()

            # Use difflib to get line-by-line diff

            differ = difflib.Differ()
            diff_lines = list(differ.compare(old_lines, new_lines))

            # Find diff chunks (continuous groups of changes)
            chunks = []
            current_chunk = []
            in_chunk = False

            for i, line in enumerate(diff_lines):
                if line.startswith("+ ") or line.startswith("- "):
                    if not in_chunk:
                        in_chunk = True
                        # Add some context lines before the chunk
                        start_idx = max(0, i - 3)
                        current_chunk = diff_lines[start_idx:i]
                    current_chunk.append(line)
                else:
                    if in_chunk:
                        # Add some context lines after the chunk
                        current_chunk.append(line)
                        if len(current_chunk) >= 3 or i == len(diff_lines) - 1:
                            chunks.append(current_chunk)
                            current_chunk = []
                            in_chunk = False

            # If there's a partial chunk left, add it
            if current_chunk:
                chunks.append(current_chunk)

            # If no chunks, it means no changes
            if not chunks:
                console.print("[yellow]No differences found.[/yellow]")
                return False, original

            # Review each chunk
            current_chunk_idx = 0

            while current_chunk_idx < len(chunks):
                chunk = chunks[current_chunk_idx]

                # Display the current chunk with line numbers
                console.print(
                    f"\n[bold]Chunk {current_chunk_idx+1}/{len(chunks)}:[/bold]"
                )
                for i, line in enumerate(chunk):
                    if line.startswith("+ "):
                        console.print(f"[green]{i+1:3d}: {line}[/green]")
                    elif line.startswith("- "):
                        console.print(f"[red]{i+1:3d}: {line}[/red]")
                    else:
                        console.print(f"{i+1:3d}: {line}")

                # Get command
                cmd = input("\nCommand (a/r/s/v/q): ").strip().lower()

                if cmd == "a":
                    # Accept all remaining changes
                    return True, new
                elif cmd == "r":
                    # Reject all changes
                    return False, original
                elif cmd == "s":
                    # Skip to next chunk
                    current_chunk_idx += 1
                elif cmd == "v":
                    # Just view the current chunk again
                    pass
                elif cmd == "q":
                    # Quit with current changes
                    break
                else:
                    console.print("[yellow]Unknown command. Try again.[/yellow]")

            # After review, ask for final confirmation
            console.print("\n[bold]Review completed.[/bold]")
            confirm = input("Apply all accepted changes? (y/n): ").strip().lower()
            if confirm == "y":
                return True, new
            else:
                return False, original

        except Exception as e:
            console.print(f"[bold red]Error in interactive diff:[/bold red] {str(e)}")
            # Fall back to non-interactive diff on error
            return await self.non_interactive_diff_confirm(original, new)

    async def non_interactive_diff_confirm(
        self, original: str, new: str
    ) -> Tuple[bool, str]:
        """Fall back to simple confirmation if interactive mode fails."""
        confirm = input("Apply these changes? (y/n): ").strip().lower()
        if confirm == "y":
            return True, new
        return False, original

    async def analyze_and_segment_changes(
        self, original: str, new: str, file_path: str
    ):
        """Analyze changes and recommend/implement segmented edits if changes are large."""
        # Parse files into lines
        old_lines = original.splitlines()
        new_lines = new.splitlines()

        # Use difflib to find unified diff

        diff = list(difflib.unified_diff(old_lines, new_lines, n=3))

        # Calculate change size
        plus_lines = len([line for line in diff if line.startswith("+")])
        minus_lines = len([line for line in diff if line.startswith("-")])
        total_changes = plus_lines + minus_lines

        # If changes are small, return the original and new as is
        if total_changes <= self.max_diff_size:
            return [(original, new, "All changes")]

        # For large changes, try to segment them
        console.print(
            "[yellow]Large changes detected. Attempting to break down into smaller segments...[/yellow]"
        )

        # Use sequence matcher to find matching blocks that can serve as segment boundaries
        s = difflib.SequenceMatcher(None, old_lines, new_lines)
        matches = s.get_matching_blocks()

        # Identify potential segment points (significant matches surrounded by changes)
        segment_points = []
        for i, j, n in matches:
            if n > 5:  # Only consider matches of 5+ lines as segment boundaries
                segment_points.append((i, j))

        # If we can't find good segment points, fall back to dividing evenly
        if len(segment_points) <= 1:
            console.print(
                "[yellow]Could not find natural segments. Creating segments by size...[/yellow]"
            )

            # Create maximum 5 segments
            target_segment_size = max(10, total_changes // 5)

            segments = []
            segment_old_lines = []
            segment_new_lines = []
            current_count = 0

            for i, old_line in enumerate(old_lines):
                # Try to find corresponding new line
                segment_old_lines.append(old_line)

                # Check if there's a matching line in new_lines
                if i < len(new_lines):
                    segment_new_lines.append(new_lines[i])
                    if old_line != new_lines[i]:
                        current_count += 1
                else:
                    # This line was removed
                    current_count += 1

                # If we've reached the target segment size, create a segment
                if current_count >= target_segment_size:
                    segments.append(
                        (
                            "\n".join(segment_old_lines),
                            "\n".join(segment_new_lines),
                            f"Lines {len(segments)*target_segment_size+1}-{(len(segments)+1)*target_segment_size}",
                        )
                    )
                    segment_old_lines = []
                    segment_new_lines = []
                    current_count = 0

            # Add any remaining segment
            if segment_old_lines or segment_new_lines:
                start_line = len(segments) * target_segment_size + 1
                end_line = start_line + len(segment_old_lines) - 1
                segments.append(
                    (
                        "\n".join(segment_old_lines),
                        "\n".join(segment_new_lines),
                        f"Lines {start_line}-{end_line}",
                    )
                )

            return segments

        # Create segments based on natural boundaries
        segments = []
        last_old_idx = 0
        last_new_idx = 0

        for i, (old_idx, new_idx) in enumerate(segment_points):
            if (
                old_idx > last_old_idx or new_idx > last_new_idx
            ):  # Only create segments if there's something to include
                # Create a segment from the last point to this one
                old_segment = "\n".join(old_lines[last_old_idx:old_idx])
                new_segment = "\n".join(new_lines[last_new_idx:new_idx])

                segment_name = f"Lines {last_old_idx+1}-{old_idx} (old) / {last_new_idx+1}-{new_idx} (new)"
                segments.append((old_segment, new_segment, segment_name))

                last_old_idx = old_idx
                last_new_idx = new_idx

        # Add the final segment if needed
        if last_old_idx < len(old_lines) or last_new_idx < len(new_lines):
            old_segment = "\n".join(old_lines[last_old_idx:])
            new_segment = "\n".join(new_lines[last_new_idx:])
            segment_name = f"Lines {last_old_idx+1}-{len(old_lines)} (old) / {last_new_idx+1}-{len(new_lines)} (new)"
            segments.append((old_segment, new_segment, segment_name))

        # If we ended up with only one segment, return the original and new as is
        if len(segments) <= 1:
            return [(original, new, "All changes")]

        return segments

    async def resolve_path(self, path: str) -> str:
        """Resolve a path to an absolute path."""
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.current_dir, path))

    async def _animate_thinking(self, styles, stop_event):
        """Display an animated thinking indicator like KITT from Knight Rider until stop_event is set."""
        # Use purple and orange colors for the beam
        colors = ["[bold magenta]", "[bold #FF8C00]"]  # Purple and orange
        position = 0
        direction = 1  # 1 for moving right, -1 for moving left
        max_position = 20  # Width of the animation
        dots = "●" * 3  # The beam size (3 dots)
        while not stop_event.is_set():
            spaces_before = " " * position
            spaces_after = " " * (max_position - position - len(dots))
            color = colors[(position // 2) % len(colors)]
            animation_text = f"\r{color}Thinking... {spaces_before}{dots}{spaces_after}[/]"
            console.print(animation_text, end="")
            sys.stdout.flush()
            position += direction
            if position > max_position - len(dots) or position < 0:
                direction *= -1
            await asyncio.sleep(0.09)
        # Clear the thinking indicator line when done
        console.print("\r" + " " * (max_position + 30) + "\r", end="")


# Add the main entry point so the program can start
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude Code Assistant CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-7-sonnet-20250219",
        help="Anthropic model to use (default: claude-3-7-sonnet-20250219)",
    )
    args = parser.parse_args()

    try:
        # Initialize the assistant with the specified model
        assistant = ClaudeSonnetCodeAssistant(model=args.model)
        # Run the assistant's start method in the event loop
        asyncio.run(assistant.start())
    except KeyboardInterrupt:
        print("\nExiting Claude Code Assistant")
    except Exception as e:
        print(f"Error: {str(e)}")
