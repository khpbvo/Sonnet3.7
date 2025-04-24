#!/usr/bin/env python3
import os
import sys
import asyncio
import subprocess
import glob
import re
import argparse
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

import anthropic
import anyio  # Import anyio library
import tiktoken  # Import tiktoken for token counting
from anthropic.types import MessageParam
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.progress import Progress
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import print as rprint
import unidiff
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

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
        
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-7-sonnet-20250219"):
        """Initialize the Claude Code Assistant CLI."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as ANTHROPIC_API_KEY environment variable")
        
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.current_dir = os.getcwd()
        self.context = []
        self.command_history = []
        
        # Initialize tiktoken encoder
        self.encoder = tiktoken.get_encoding("cl100k_base")  # Use get_encoding for encoding names
        self.max_tokens = 180000  # Max token threshold for auto-summarization
        self.current_tokens = 0  # Current token count
        
        # Command completions
        self.commands = [
            "code:read:", "code:read:files:", "code:changedir:", "code:listdir:", 
            "code:generate:", "code:change:", "code:search:", 
            "code:shell:", "help", "exit", "quit"
        ]
        
        # Create prompt session with history and bottom toolbar for token counter
        history_file = os.path.expanduser("~/.claude_code_assistant_history")
        self.session = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(self.commands, sentence=True),
            bottom_toolbar=self.get_token_toolbar
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
        console.print(Panel.fit(
            "[bold blue]Claude Sonnet Code Assistant[/bold blue]\n"
            f"[green]Using model: {self.model}[/green]\n"
            "Type [bold]help[/bold] for available commands or [bold]exit[/bold] to quit",
            title="Welcome", subtitle="v1.0"
        ))

        while True:
            try:
                # Get user input
                user_input = await self.session.prompt_async(f"[{os.path.basename(self.current_dir)}] > ")
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
                console.print("\n[yellow]Interrupted. Press Ctrl+C again to exit.[/yellow]")
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
                    console.print("[bold red]Invalid format. Use code:generate:file.py:prompt[/bold red]")
            elif cmd_type == "change" and len(parts) > 2:
                file_prompt = parts[2].split(":", 1)
                if len(file_prompt) == 2:
                    await self.handle_change(file_prompt[0], file_prompt[1])
                else:
                    console.print("[bold red]Invalid format. Use code:change:file.py:prompt[/bold red]")
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
        """Display help information."""
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
        
        [bold]Token Management:[/bold]
        Token usage is displayed at the bottom right.
        Files will be automatically summarized when total tokens exceed 180,000.
        """
        console.print(Panel(help_text, title="Claude Code Assistant Help"))

    async def handle_read_multiple(self, file_list: str):
        """Read multiple files at once and add them to context."""
        # Split the comma-separated list of files
        files = [f.strip() for f in file_list.split(',')]
        
        if not files:
            console.print("[bold red]No files specified[/bold red]")
            return
        
        # Process each file
        for file_path in files:
            await self.handle_read(file_path)

    async def handle_read(self, file_path: str):
        try:
            file_content = await self.read_file(self.resolve_path(file_path))
            if file_content:
                file_ext = os.path.splitext(file_path)[1].lstrip(".")
                syntax = Syntax(file_content, file_ext, line_numbers=True)
                console.print(Panel(syntax, title=f"File: {file_path}"))
                tokens_in_file = self.count_tokens(file_content)
                tokens_in_path = self.count_tokens(file_path)
                total_file_tokens = tokens_in_file + tokens_in_path
                console.print(f"[cyan]File contains {tokens_in_file:,} tokens[/cyan]")
                self.context.append({
                    "type": "file",
                    "path": file_path,
                    "content": file_content,
                    "tokens": total_file_tokens
                })
                console.print(f"[green]Added {file_path} to context[/green]")
                self.current_tokens += total_file_tokens
        except Exception as e:
            console.print(f"[bold red]Error reading file:[/bold red] {str(e)}")

    async def handle_changedir(self, dir_path: str):
        """Change current working directory."""
        try:
            new_dir = self.resolve_path(dir_path)
            if os.path.isdir(new_dir):
                os.chdir(new_dir)
                self.current_dir = os.getcwd()
                console.print(f"[green]Changed directory to:[/green] {self.current_dir}")
            else:
                console.print(f"[bold red]Directory does not exist:[/bold red] {new_dir}")
        except Exception as e:
            console.print(f"[bold red]Error changing directory:[/bold red] {str(e)}")

    async def handle_listdir(self, dir_path: str):
        """List files and directories in the specified path."""
        try:
            target_dir = self.resolve_path(dir_path)
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
                
                console.print(f"[dim]{len(directories)} directories, {len(files)} files[/dim]")
            else:
                console.print(f"[bold red]Directory does not exist:[/bold red] {target_dir}")
        except Exception as e:
            console.print(f"[bold red]Error listing directory:[/bold red] {str(e)}")

    async def handle_generate(self, file_path: str, prompt: str):
        try:
            # Check if file already exists
            full_path = self.resolve_path(file_path)
            original_content = ""
            
            if os.path.exists(full_path):
                original_content = await self.read_file(full_path)
                console.print(f"[yellow]File already exists. Will generate a diff for changes.[/yellow]")
            
            # Prepare prompt for Claude
            file_ext = os.path.splitext(file_path)[1].lstrip(".")
            messages = [
                MessageParam(
                    role="user", 
                    content=f"Generate code for a file named {file_path}. File type: {file_ext}.\n\nRequirements:\n{prompt}\n\nPlease provide only the code without any markdown formatting, explanations, or additional text."
                )
            ]
            
            # Get response from Claude
            with Progress(transient=True) as progress:
                task = progress.add_task("[cyan]Generating code...", total=None)
                
                new_content = ""
                stream = await self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=4000,
                    stream=True,
                )
                
                async for chunk in stream:
                    if chunk.delta.text:
                        new_content += chunk.delta.text
                        progress.update(task)
            
            # Show the diff
            diff = await self.create_colored_diff(original_content, new_content, file_path)
            console.print(diff)
            
            # Ask for confirmation before saving
            confirm = input("Save this code? (y/n): ").strip().lower()
            if confirm == 'y':
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(full_path)), exist_ok=True)
                
                # Write the file
                with open(full_path, 'w') as f:
                    f.write(new_content)
                console.print(f"[green]Saved code to {full_path}[/green]")
                
                # Count tokens for new file
                tokens_in_file = self.count_tokens(new_content)
                tokens_in_path = self.count_tokens(file_path)
                total_file_tokens = tokens_in_file + tokens_in_path
                
                # Add to context
                self.context.append({
                    "type": "file",
                    "path": file_path,
                    "content": new_content,
                    "tokens": total_file_tokens
                })
                self.current_tokens += total_file_tokens
            else:
                console.print("[yellow]Code generation cancelled[/yellow]")
                
        except Exception as e:
            console.print(f"[bold red]Error generating code:[/bold red] {str(e)}")

    async def handle_change(self, file_path: str, prompt: str):
        try:
            # Read the existing file
            full_path = self.resolve_path(file_path)
            if not os.path.exists(full_path):
                console.print(f"[bold red]File does not exist:[/bold red] {full_path}")
                return
            
            original_content = await self.read_file(full_path)
            
            # Calculate tokens to remove from context if file is already there
            old_tokens = 0
            for i, ctx in enumerate(self.context):
                if ctx.get("type") == "file" and ctx.get("path") == file_path:
                    old_tokens = ctx.get("tokens", 0)
                    break
            
            # Prepare context for Claude
            file_ext = os.path.splitext(file_path)[1].lstrip(".")
            messages = [
                MessageParam(
                    role="user", 
                    content=f"Here's the content of {file_path}:\n\n```{file_ext}\n{original_content}\n```\n\nPlease modify this code according to the following request:\n{prompt}\n\nProvide ONLY the complete updated code without any markdown formatting, explanations, or additional text."
                )
            ]
            
            # Get response from Claude
            with Progress(transient=True) as progress:
                task = progress.add_task("[cyan]Modifying code...", total=None)
                
                new_content = ""
                stream = await self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=4000,
                    stream=True,
                )
                
                # Process different types of events from the stream
                async for event in stream:
                    if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                        # Handle ContentBlockDeltaEvent
                        if event.delta.text:
                            new_content += event.delta.text
                            progress.update(task)
                    elif hasattr(event, 'type') and event.type == 'content_block_delta':
                        # Alternative way to handle content block delta
                        if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                            new_content += event.delta.text
                            progress.update(task)
                    elif hasattr(event, 'content_block') and hasattr(event.content_block, 'text'):
                        # Handle ContentBlockStartEvent
                        new_content += event.content_block.text
                        progress.update(task)
            
            # Clean up any potential markdown code blocks
            new_content = re.sub(r'```[\w]*\n', '', new_content)
            new_content = new_content.replace('```', '')
            
            # Show the diff
            diff = await self.create_colored_diff(original_content, new_content, file_path)
            console.print(diff)
            
            # Ask for confirmation before saving
            confirm = input("Apply these changes? (y/n): ").strip().lower()
            if confirm == 'y':
                with open(full_path, 'w') as f:
                    f.write(new_content)
                console.print(f"[green]Updated {full_path}[/green]")
                
                # Count tokens for updated file
                tokens_in_file = self.count_tokens(new_content)
                tokens_in_path = self.count_tokens(file_path)
                total_file_tokens = tokens_in_file + tokens_in_path
                
                # Update context
                for i, ctx in enumerate(self.context):
                    if ctx.get("type") == "file" and ctx.get("path") == file_path:
                        self.current_tokens -= old_tokens
                        self.context[i]["content"] = new_content
                        self.context[i]["tokens"] = total_file_tokens
                        self.current_tokens += total_file_tokens
                        break
                else:
                    # File wasn't in context before
                    self.context.append({
                        "type": "file",
                        "path": file_path,
                        "content": new_content,
                        "tokens": total_file_tokens
                    })
                    self.current_tokens += total_file_tokens
            else:
                console.print("[yellow]Code changes cancelled[/yellow]")
                
        except Exception as e:
            console.print(f"[bold red]Error modifying code:[/bold red] {str(e)}")

    async def handle_search(self, query: str):
        """Search for files or code."""
        try:
            if ":" in query:
                search_type, search_query = query.split(":", 1)
                if search_type == "file":
                    await self.search_file(search_query)
                elif search_type == "code":
                    await self.search_code(search_query)
                else:
                    console.print(f"[bold red]Unknown search type:[/bold red] {search_type}")
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
                cwd=self.current_dir
            )
            
            stdout, stderr = await process.communicate()
            
            # Display output
            if stdout:
                stdout_str = stdout.decode()
                console.print(Panel(stdout_str, title="Standard Output", border_style="green"))
            
            if stderr:
                stderr_str = stderr.decode()
                console.print(Panel(stderr_str, title="Standard Error", border_style="red"))
            
            console.print(f"[dim]Command exited with code {process.returncode}[/dim]")
            
        except Exception as e:
            console.print(f"[bold red]Error executing shell command:[/bold red] {str(e)}")

    async def ask_claude(self, prompt: str):
        """Send a direct prompt to Claude."""
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
            full_prompt = f"{context_str}Working directory: {self.current_dir}\n\n{prompt}"
            
            # Display an animated thinking indicator
            thinking_styles = ["[bold blue]Thinking...[/bold blue]", 
                              "[bold green]Thinking...[/bold green]", 
                              "[bold yellow]Thinking...[/bold yellow]", 
                              "[bold magenta]Thinking...[/bold magenta]", 
                              "[bold cyan]Thinking...[/bold cyan]"]
            
            # Create and start the thinking animation task
            stop_thinking = asyncio.Event()
            thinking_task = asyncio.create_task(self._animate_thinking(thinking_styles, stop_thinking))
            
            try:
                # Send to Claude
                response_text = ""
                stream = await self.client.messages.create(
                    model=self.model,
                    messages=[MessageParam(role="user", content=full_prompt)],
                    max_tokens=4000,
                    stream=True,
                )
                
                # Stop thinking animation and clear the line
                stop_thinking.set()
                await thinking_task
                console.print("\r" + " " * 60 + "\r", end="")
                
                # Print "Assistant:" label in red before starting the response
                console.print("\n[bold red]Assistant:[/bold red] ", end="")
                
                # Stream the response token by token with a small delay
                async for event in stream:
                    if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                        # Process ContentBlockDeltaEvent
                        chunk = event.delta.text
                        if chunk:
                            response_text += chunk
                            console.print(chunk, end="", highlight=False)
                            # Force flush to ensure immediate display
                            sys.stdout.flush()
                            # Add a small delay between tokens for better readability
                            await asyncio.sleep(0.01)
                    elif hasattr(event, 'type') and event.type == 'content_block_delta':
                        # Alternative way to handle content block delta
                        if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                            chunk = event.delta.text
                            if chunk:
                                response_text += chunk
                                console.print(chunk, end="", highlight=False)
                                sys.stdout.flush()
                                # Add a small delay between tokens for better readability
                                await asyncio.sleep(0.01)
                    elif hasattr(event, 'content_block') and hasattr(event.content_block, 'text'):
                        # Handle ContentBlockStartEvent
                        chunk = event.content_block.text
                        if chunk:
                            response_text += chunk
                            console.print(chunk, end="", highlight=False)
                            sys.stdout.flush()
                            # Add a small delay between tokens for better readability
                            await asyncio.sleep(0.01)
                
                console.print()  # newline after response
                
            except Exception as e:
                # Make sure to stop the thinking animation if there's an error
                stop_thinking.set()
                await thinking_task
                raise e
                
        except anthropic.APIStatusError as e:
            # Handle API-specific errors
            error_message = str(e)
            if "overloaded_error" in error_message:
                console.print("\r[bold red]Error: Claude API is currently overloaded. Please try again in a few moments.[/bold red]")
            else:
                console.print(f"\r[bold red]API Error communicating with Claude:[/bold red] {str(e)}")
        except Exception as e:
            console.print(f"\r[bold red]Error communicating with Claude:[/bold red] {str(e)}")
            
    async def _animate_thinking(self, styles, stop_event):
        """Display an animated thinking indicator like KITT from Knight Rider until stop_event is set."""
        # Clear any previous output and ensure we're on a fresh line
        console.print("\r", end="")
        
        # Use purple and orange colors for the beam
        colors = ["[bold purple]", "[bold #FF8C00]"]  # Purple and orange (dark orange)
        
        position = 0
        direction = 1  # 1 for moving right, -1 for moving left
        max_position = 25  # Maximum width of the animation
        dots = "●" * 3  # The beam size (3 dots)
        
        while not stop_event.is_set():
            # Create the spaces before and after the beam
            spaces_before = " " * position
            spaces_after = " " * (max_position - position - len(dots))
            
            # Alternate between purple and orange
            color = colors[int(position/2) % len(colors)]
            
            # Build the complete animation string on a single line
            animation_text = f"\r{color}Thinking... {spaces_before}{dots}{spaces_after}[/]"
            
            # Print without newline and flush
            console.print(animation_text, end="")
            sys.stdout.flush()
            
            # Update position for next iteration
            position += direction
            
            # Change direction when hitting the edges
            if position > max_position - len(dots) or position < 0:
                direction *= -1
            
            # Control speed
            await asyncio.sleep(0.09)
        
        # Clear the thinking indicator line when done
        console.print("\r" + " " * (max_position + 20) + "\r", end="")

    async def auto_summarize(self):
        console.print(f"[yellow]Token limit ({self.max_tokens:,}) exceeded. Summarizing files...[/yellow]")
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
            console.print(f"[yellow]Summarizing {file_path} ({file_tokens:,} tokens)[/yellow]")
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
            console.print(f"[green]Summarized {file_path}: {file_tokens:,} → {total_summary_tokens:,} tokens ({(total_summary_tokens/file_tokens)*100:.1f}%)[/green]")
            if self.current_tokens <= self.max_tokens * 0.7:
                break
        console.print(f"[green]Finished summarizing. Current token count: {self.current_tokens:,}[/green]")

    async def generate_file_summary(self, file_path: str, content: str) -> str:
        """Generate a summary of a file using Claude."""
        try:
            file_ext = os.path.splitext(file_path)[1].lstrip(".")
            
            messages = [
                MessageParam(
                    role="user", 
                    content=f"Please summarize the following {file_ext} file so I can understand its structure and functionality. Focus on key elements that would be important for understanding how to use or modify this code. Keep implementation details brief.\n\n```{file_ext}\n{content}\n```\n\nProvide a concise summary that preserves the most important information about this file."
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
            file = await anyio.open_file(file_path, 'r')
            async with file as f:
                return await f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                file = await anyio.open_file(file_path, 'r', encoding='latin-1')
                async with file as f:
                    return await f.read()
            except Exception as e:
                console.print(f"[bold red]Could not read file (binary file?):[/bold red] {str(e)}")
                return ""
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

    async def search_file(self, pattern: str):
        """Search for files matching a pattern."""
        try:
            # Check if it's a glob pattern
            if any(char in pattern for char in "*?[]"):
                matches = glob.glob(os.path.join(self.current_dir, "**", pattern), recursive=True)
            else:
                # Use find command for more complex searches
                cmd = f"find {self.current_dir} -type f -name '*{pattern}*' 2>/dev/null"
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()
                matches = stdout.decode().strip().split('\n')
                matches = [m for m in matches if m]  # Filter empty strings
            
            if matches:
                console.print(f"[green]Found {len(matches)} file(s) matching '{pattern}':[/green]")
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
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            results = stdout.decode().strip().split('\n')
            results = [r for r in results if r]  # Filter empty strings
            
            if results:
                console.print(f"[green]Found {len(results)} match(es) for '{pattern}' in code:[/green]")
                for result in results:
                    parts = result.split(':', 1)
                    if len(parts) >= 2:
                        file_path = parts[0]
                        content = parts[1]
                        rel_path = os.path.relpath(file_path, self.current_dir)
                        console.print(f"[bold]{rel_path}[/bold]: {content}")
            else:
                console.print(f"[yellow]No code matches found for '{pattern}'[/yellow]")
                
        except Exception as e:
            console.print(f"[bold red]Error searching code:[/bold red] {str(e)}")

    async def create_colored_diff(self, original: str, new: str, file_path: str) -> str:
        """Create a colored unified diff."""
        try:
            # Create temporary files for diffing
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as old_file:
                old_file.write(original)
                old_file_path = old_file.name
                
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as new_file:
                new_file.write(new)
                new_file_path = new_file.name
            
            try:
                # Use the 'diff' command to create a unified diff
                proc = await asyncio.create_subprocess_exec(
                    'diff', '-u', old_file_path, new_file_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await proc.communicate()
                diff_output = stdout.decode()
                
                # Skip unidiff parsing and just colorize the output directly
                diff_text = ""
                for line in diff_output.splitlines():
                    if line.startswith('+'):
                        diff_text += f"[green]{line}[/green]\n"
                    elif line.startswith('-'):
                        diff_text += f"[red]{line}[/red]\n"
                    elif line.startswith('@'):
                        diff_text += f"[cyan]{line}[/cyan]\n"
                    else:
                        diff_text += f"{line}\n"
                
                return Panel(diff_text, title=f"Diff for {file_path}", border_style="blue")
                
            finally:
                # Clean up temporary files
                os.unlink(old_file_path)
                os.unlink(new_file_path)
                
        except Exception as e:
            return f"[bold red]Error creating diff:[/bold red] {str(e)}"

    def resolve_path(self, path: str) -> str:
        """Resolve a path relative to the current directory."""
        if os.path.isabs(path):
            return os.path.normpath(path)
        else:
            return os.path.normpath(os.path.join(self.current_dir, path))

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Claude Sonnet Code Assistant")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default="claude-3-7-sonnet-20250219", help="Claude model to use")
    
    args = parser.parse_args()
    
    try:
        assistant = ClaudeSonnetCodeAssistant(api_key=args.api_key, model=args.model)
        await assistant.start()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())