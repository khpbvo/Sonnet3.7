#!/usr/bin/env python3
import argparse
import asyncio
import glob
import os
import re
import sys
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
from rich.progress import Progress
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
                file_prompt = parts[2].split(":", 1)
                if len(file_prompt) == 2:
                    await self.handle_change(file_prompt[0], file_prompt[1])
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

        [bold]Diff Operations:[/bold]
        When changing files, line numbers are shown and changes can be:
        - Broken into multiple smaller segments for easier review
        - Reviewed in an interactive VIM-like interface
        - Applied selectively (accept/reject specific change segments)
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
        try:
            # Check if file already exists
            full_path = self.resolve_path(file_path)
            original_content = ""

            if os.path.exists(full_path):
                original_content = await self.read_file(full_path)
                console.print(
                    "[yellow]File already exists. Will generate a diff for changes.[/yellow]"
                )

            # Prepare prompt for Claude
            file_ext = os.path.splitext(file_path)[1].lstrip(".")
            messages = [
                MessageParam(
                    role="user",
                    content=f"Generate code for a file named {file_path}. File type: {file_ext}.\n\nRequirements:\n{prompt}\n\nPlease provide only the code without any markdown formatting, explanations, or additional text.",
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

            # Clean up any potential markdown code blocks
            new_content = re.sub(r"```[\w]*\n", "", new_content)
            new_content = new_content.replace("```", "")

            # Determine if we need to show diffs (only if file already exists)
            if original_content:
                # Analyze and break down changes if needed
                segments = await self.analyze_and_segment_changes(
                    original_content, new_content, file_path
                )

                final_content = original_content
                if len(segments) == 1:
                    # Just one segment, show regular diff
                    diff = await self.create_colored_diff(
                        original_content, new_content, file_path, enhanced=True
                    )
                    console.print(diff)

                    # Ask if they want to use interactive diff viewer
                    use_interactive = (
                        input(
                            "Would you like to use the interactive diff viewer? (y/n): "
                        )
                        .strip()
                        .lower()
                        == "y"
                    )

                    if use_interactive:
                        accepted, final_content = await self.interactive_diff_review(
                            original_content, new_content, file_path
                        )
                    else:
                        # Regular confirmation
                        confirm = input("Apply these changes? (y/n): ").strip().lower()
                        accepted = confirm == "y"
                        final_content = new_content if accepted else original_content
                else:
                    # Multiple segments, review each one
                    console.print(
                        f"[yellow]Changes have been divided into {len(segments)} segments for easier review.[/yellow]"
                    )

                    # Start with the original content
                    working_content = original_content

                    # Process each segment
                    for i, (segment_old, segment_new, segment_name) in enumerate(
                        segments
                    ):
                        console.print(
                            f"[bold]Segment {i+1}/{len(segments)}: {segment_name}[/bold]"
                        )

                        # Show diff for this segment
                        segment_diff = await self.create_colored_diff(
                            segment_old,
                            segment_new,
                            f"{file_path} (Segment {i+1})",
                            enhanced=True,
                        )
                        console.print(segment_diff)

                        # Ask for confirmation
                        confirm = (
                            input(f"Apply changes for segment {i+1}? (y/n): ")
                            .strip()
                            .lower()
                        )
                        if confirm == "y":
                            # Apply this segment's changes
                            working_content = working_content.replace(
                                segment_old, segment_new
                            )
                            console.print(
                                f"[green]Applied changes for segment {i+1}[/green]"
                            )
                        else:
                            console.print(
                                f"[yellow]Skipped changes for segment {i+1}[/yellow]"
                            )

                    final_content = working_content
                    accepted = final_content != original_content
            else:
                # New file, just show the generated code
                file_ext = os.path.splitext(file_path)[1].lstrip(".")
                syntax = Syntax(new_content, file_ext, line_numbers=True)
                console.print(Panel(syntax, title=f"Generated code for: {file_path}"))

                # Ask for confirmation
                confirm = input("Save this code? (y/n): ").strip().lower()
                accepted = confirm == "y"
                final_content = new_content if accepted else ""

            # If accepted and has content, save the file
            if accepted and final_content:
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(full_path)), exist_ok=True)

                # Write the file
                with open(full_path, "w") as f:
                    f.write(final_content)
                console.print(f"[green]Saved code to {full_path}[/green]")

                # Count tokens for new file
                tokens_in_file = self.count_tokens(final_content)
                tokens_in_path = self.count_tokens(file_path)
                total_file_tokens = tokens_in_file + tokens_in_path

                # Add to context
                self.context.append(
                    {
                        "type": "file",
                        "path": file_path,
                        "content": final_content,
                        "tokens": total_file_tokens,
                    }
                )
                self.current_tokens += total_file_tokens
            else:
                console.print("[yellow]Code generation cancelled[/yellow]")

        except Exception as e:
            console.print(f"[bold red]Error generating code:[/bold red] {str(e)}")

    async def handle_change(self, file_path: str, prompt: str):
        try:
            # Read the existing file
            full_path = await self.resolve_path(file_path)
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
                    content=f"Here's the content of {file_path}:\n\n```{file_ext}\n{original_content}\n```\n\nPlease modify this code according to the following request:\n{prompt}\n\nProvide ONLY the complete updated code without any markdown formatting, explanations, or additional text.",
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
                    if hasattr(event, "delta") and hasattr(event.delta, "text"):
                        # Handle ContentBlockDeltaEvent
                        if event.delta.text:
                            new_content += event.delta.text
                            progress.update(task)
                    elif hasattr(event, "type") and event.type == "content_block_delta":
                        # Alternative way to handle content block delta
                        if hasattr(event, "delta") and hasattr(event.delta, "text"):
                            new_content += event.delta.text
                            progress.update(task)
                    elif hasattr(event, "content_block") and hasattr(
                        event.content_block, "text"
                    ):
                        # Handle ContentBlockStartEvent
                        new_content += event.content_block.text
                        progress.update(task)

            # Clean up any potential markdown code blocks
            new_content = re.sub(r"```[\w]*\n", "", new_content)
            new_content = new_content.replace("```", "")

            # Analyze and break down changes if needed
            segments = await self.analyze_and_segment_changes(
                original_content, new_content, file_path
            )

            final_content = original_content
            if len(segments) == 1:
                # Just one segment, show regular diff
                diff = await self.create_colored_diff(
                    original_content, new_content, file_path, enhanced=True
                )
                console.print(diff)

                # Automatically use the interactive diff viewer if always_use_vim_style is enabled
                if self.always_use_vim_style:
                    console.print(
                        "[cyan]Using VIM-style interactive diff reviewer for incremental patches...[/cyan]"
                    )
                    accepted, final_content = await self.interactive_diff_review(
                        original_content, new_content, file_path
                    )
                else:
                    # Ask if they want to use interactive diff viewer
                    use_interactive = (
                        input(
                            "Would you like to use the interactive diff viewer? (y/n): "
                        )
                        .strip()
                        .lower()
                        == "y"
                    )
                    if use_interactive:
                        accepted, final_content = await self.interactive_diff_review(
                            original_content, new_content, file_path
                        )
                    else:
                        # Regular confirmation
                        confirm = input("Apply these changes? (y/n): ").strip().lower()
                        accepted = confirm == "y"
                        final_content = new_content if accepted else original_content
            else:
                # Multiple segments, review each one
                console.print(
                    f"[yellow]Changes have been divided into {len(segments)} segments for easier review.[/yellow]"
                )

                # Start with the original content
                working_content = original_content

                # Process each segment
                for i, (segment_old, segment_new, segment_name) in enumerate(segments):
                    console.print(
                        f"[bold]Segment {i+1}/{len(segments)}: {segment_name}[/bold]"
                    )

                    # Show diff for this segment
                    segment_diff = await self.create_colored_diff(
                        segment_old,
                        segment_new,
                        f"{file_path} (Segment {i+1})",
                        enhanced=True,
                    )
                    console.print(segment_diff)

                    # Ask for confirmation
                    confirm = (
                        input(f"Apply changes for segment {i+1}? (y/n): ")
                        .strip()
                        .lower()
                    )
                    if confirm == "y":
                        # Apply this segment's changes
                        working_content = working_content.replace(
                            segment_old, segment_new
                        )
                        console.print(
                            f"[green]Applied changes for segment {i+1}[/green]"
                        )
                    else:
                        console.print(
                            f"[yellow]Skipped changes for segment {i+1}[/yellow]"
                        )

                final_content = working_content
                accepted = final_content != original_content

            # If changes were accepted, save the file
            if accepted:
                with open(full_path, "w") as f:
                    f.write(final_content)
                console.print(f"[green]Updated {full_path}[/green]")

                # Count tokens for updated file
                tokens_in_file = self.count_tokens(final_content)
                tokens_in_path = self.count_tokens(file_path)
                total_file_tokens = tokens_in_file + tokens_in_path

                # Update context
                for i, ctx in enumerate(self.context):
                    if ctx.get("type") == "file" and ctx.get("path") == file_path:
                        self.current_tokens -= old_tokens
                        self.context[i]["content"] = final_content
                        self.context[i]["tokens"] = total_file_tokens
                        self.current_tokens += total_file_tokens
                        break
                else:
                    # File wasn't in context before
                    self.context.append(
                        {
                            "type": "file",
                            "path": file_path,
                            "content": final_content,
                            "tokens": total_file_tokens,
                        }
                    )
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
            full_prompt = (
                f"{context_str}Working directory: {self.current_dir}\n\n{prompt}"
            )

            # Display an animated thinking indicator
            thinking_styles = [
                "[bold blue]Thinking...[/bold blue]",
                "[bold green]Thinking...[/bold green]",
                "[bold yellow]Thinking...[/bold yellow]",
                "[bold magenta]Thinking...[/bold magenta]",
                "[bold cyan]Thinking...[/bold cyan]",
            ]

            # Create and start the thinking animation task
            stop_thinking = asyncio.Event()
            thinking_task = asyncio.create_task(
                self._animate_thinking(thinking_styles, stop_thinking)
            )

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
                    if hasattr(event, "delta") and hasattr(event.delta, "text"):
                        # Process ContentBlockDeltaEvent
                        chunk = event.delta.text
                        if chunk:
                            response_text += chunk
                            console.print(chunk, end="", highlight=False)
                            # Force flush to ensure immediate display
                            sys.stdout.flush()
                            # Add a small delay between tokens for better readability
                            await asyncio.sleep(0.01)
                    elif hasattr(event, "type") and event.type == "content_block_delta":
                        # Alternative way to handle content block delta
                        if hasattr(event, "delta") and hasattr(event.delta, "text"):
                            chunk = event.delta.text
                            if chunk:
                                response_text += chunk
                                console.print(chunk, end="", highlight=False)
                                sys.stdout.flush()
                                # Add a small delay between tokens for better readability
                                await asyncio.sleep(0.01)
                    elif hasattr(event, "content_block") and hasattr(
                        event.content_block, "text"
                    ):
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
                console.print(
                    "\r[bold red]Error: Claude API is currently overloaded. Please try again in a few moments.[/bold red]"
                )
            else:
                console.print(
                    f"\r[bold red]API Error communicating with Claude:[/bold red] {str(e)}"
                )
        except Exception as e:
            console.print(
                f"\r[bold red]Error communicating with Claude:[/bold red] {str(e)}"
            )

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
            color = colors[int(position / 2) % len(colors)]

            # Build the complete animation string on a single line
            animation_text = (
                f"\r{color}Thinking... {spaces_before}{dots}{spaces_after}[/]"
            )

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
    ) -> str:
        """Create a colored unified diff with explicit line numbers."""
        try:
            # Create temporary files for diffing
            import os
            import tempfile

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
                            enhanced_diff.append(
                                f"[bold cyan]--- Lines {old_start}+ in original[/bold cyan]"
                            )
                            enhanced_diff.append(
                                f"[bold cyan]+++ Lines {new_start}+ in new version[/bold cyan]"
                            )
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
                                    enhanced_diff.append(
                                        f"{old_prefix}{new_prefix} {line}"
                                    )
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

                return Panel(
                    diff_text, title=f"Diff for {file_path}", border_style="blue"
                )

            finally:
                # Clean up temporary files
                os.unlink(old_file_path)
                os.unlink(new_file_path)

        except Exception as e:
            return f"[bold red]Error creating diff:[/bold red] {str(e)}"

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
            import difflib

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
            working_content = original
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
        return confirm == "y", new if confirm == "y" else original

    async def analyze_and_segment_changes(
        self, original: str, new: str, file_path: str
    ):
        """Analyze changes and recommend/implement segmented edits if changes are large."""
        # Parse files into lines
        old_lines = original.splitlines()
        new_lines = new.splitlines()

        # Use difflib to find unified diff
        import difflib

        diff = list(difflib.unified_diff(old_lines, new_lines, n=3))
        diff_text = "\n".join(diff)

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
