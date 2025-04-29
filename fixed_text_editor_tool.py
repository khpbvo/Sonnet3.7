"""
This is a corrected implementation for the text editor tool integration, based on Anthropic's
official documentation. Replace the relevant part of your Claude3.7.py file with this code.
"""

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
        
        try:
            modified_files = set()
            
            # Use the proper tool name according to Anthropic docs: "text_editor_20250124"
            response = await self.client.messages.create(
                model=self.model,
                messages=[MessageParam(role="user", content=full_prompt)],
                max_tokens=4000,
                stream=False,  # Changed to non-streaming for simplicity
                tools=[{
                    "name": "text_editor_20250124",  # Correct tool name as per Anthropic docs
                    "description": "A tool for viewing and editing files on disk."
                    # No schema needed - it's built into Claude
                }]
            )
            
            stop_thinking.set()
            await thinking_task
            console.print("\r" + " " * 60 + "\r", end="")
            console.print("\n[bold red]Assistant:[/bold red] ", end="")
            
            # Process the response
            full_response = ""
            
            # First handle any tool uses
            if hasattr(response, 'content') and hasattr(response, 'tool_uses') and response.tool_uses:
                for tool_use in response.tool_uses:
                    if tool_use.name == "text_editor_20250124":  # Check for correct tool name
                        tool_params = tool_use.input
                        tool_command = tool_params.get('command', '')
                        tool_path = tool_params.get('path', '')
                        
                        console.print(f"\n[dim]Using text editor tool: {tool_command} on {tool_path}[/dim]")
                        
                        # Execute the tool command
                        tool_result = await self.handle_editor_tool(tool_use)
                        console.print(f"[dim]{tool_result.split(os.linesep)[0]}...[/dim]")
                        
                        # Track modified files for context updates
                        if tool_command in ['str_replace', 'create', 'insert', 'append'] and tool_path:
                            modified_files.add(tool_path)
                        
                        # Continue the conversation with tool result
                        tool_response = await self.client.messages.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "user", 
                                    "content": [
                                        {"type": "text", "text": full_prompt},
                                        {"type": "tool_result", "tool_use_id": tool_use.id, "content": tool_result}
                                    ]
                                }
                            ],
                            max_tokens=4000,
                            stream=False,
                            tools=[{
                                "name": "text_editor_20250124",  # Correct tool name
                                "description": "A tool for viewing and editing files on disk."
                                # No schema needed
                            }]
                        )
                        
                        # Handle nested tool uses recursively
                        if hasattr(tool_response, 'tool_uses') and tool_response.tool_uses:
                            for nested_tool_use in tool_response.tool_uses:
                                if nested_tool_use.name == "text_editor_20250124":  # Check for correct tool name
                                    nested_tool_params = nested_tool_use.input
                                    nested_tool_command = nested_tool_params.get('command', '')
                                    nested_tool_path = nested_tool_params.get('path', '')
                                    
                                    console.print(f"\n[dim]Using text editor tool (nested): {nested_tool_command} on {nested_tool_path}[/dim]")
                                    
                                    # Execute the nested tool command
                                    nested_result = await self.handle_editor_tool(nested_tool_use)
                                    console.print(f"[dim]{nested_result.split(os.linesep)[0]}...[/dim]")
                                    
                                    # Track modified files
                                    if nested_tool_command in ['str_replace', 'create', 'insert', 'append'] and nested_tool_path:
                                        modified_files.add(nested_tool_path)
                                    
                                    # Continue the conversation with the nested tool result
                                    await self.client.messages.create(
                                        model=self.model,
                                        messages=[
                                            {
                                                "role": "user", 
                                                "content": [
                                                    {"type": "text", "text": full_prompt},
                                                    {"type": "tool_result", "tool_use_id": nested_tool_use.id, "content": nested_result}
                                                ]
                                            }
                                        ],
                                        max_tokens=4000,
                                        stream=False
                                    )
                        
                        # Add tool response to full response
                        if hasattr(tool_response, 'content') and tool_response.content:
                            for content_block in tool_response.content:
                                if hasattr(content_block, 'text'):
                                    text = content_block.text
                                    full_response += text
                                    console.print(text, highlight=False)
            
            # Add original response if there was no tool use or after tool use
            if hasattr(response, 'content') and response.content:
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        text = content_block.text
                        if not full_response:  # Only add if we haven't added from tool response
                            full_response += text
                            console.print(text, highlight=False)
            
            console.print()  # newline after response
            
            # Update context with any modified files
            if modified_files:
                console.print(f"[yellow]Updating context with modified files...[/yellow]")
                for file_path in modified_files:
                    try:
                        resolved_path = await self.resolve_path(file_path)
                        if os.path.exists(resolved_path):
                            # Read the updated content
                            updated_content = await self.read_file(resolved_path)
                            
                            # Find if file is already in context
                            file_in_context = False
                            for i, ctx in enumerate(self.context):
                                if ctx.get("type") == "file" and ctx.get("path") == file_path:
                                    # File exists in context, update it
                                    old_tokens = ctx.get("tokens", 0)
                                    tokens_in_file = self.count_tokens(updated_content)
                                    tokens_in_path = self.count_tokens(file_path)
                                    total_file_tokens = tokens_in_file + tokens_in_path
                                    
                                    # Update token count
                                    self.current_tokens = self.current_tokens - old_tokens + total_file_tokens
                                    
                                    # Update the context entry
                                    self.context[i] = {
                                        "type": "file",
                                        "path": file_path,
                                        "content": updated_content,
                                        "tokens": total_file_tokens,
                                    }
                                    file_in_context = True
                                    console.print(f"[green]Updated {file_path} in context[/green]")
                                    break
                            
                            # If file not in context, add it
                            if not file_in_context:
                                tokens_in_file = self.count_tokens(updated_content)
                                tokens_in_path = self.count_tokens(file_path)
                                total_file_tokens = tokens_in_file + tokens_in_path
                                
                                self.context.append({
                                    "type": "file",
                                    "path": file_path,
                                    "content": updated_content,
                                    "tokens": total_file_tokens,
                                })
                                self.current_tokens += total_file_tokens
                                console.print(f"[green]Added {file_path} to context[/green]")
                    except Exception as e:
                        console.print(f"[bold red]Error updating context for {file_path}:[/bold red] {str(e)}")
        
        except Exception as e:
            stop_thinking.set()
            await thinking_task
            raise e
            
    except anthropic.APIStatusError as e:
        error_message = str(e)
        if "overloaded_error" in error_message:
            console.print("\r[bold red]Error: Claude API is currently overloaded. Please try again in a few moments.[/bold red]")
        else:
            console.print(f"\r[bold red]API Error communicating with Claude:[/bold red] {str(e)}")
    except Exception as e:
        console.print(f"\r[bold red]Error communicating with Claude:[/bold red] {str(e)}")

# You may also need to update the handle_editor_tool method to properly handle the "text_editor_20250124" tool
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
        
        # Your existing implementation continues here...
        # ...

    except Exception as e:
        return f"Error in text editor tool: {str(e)}"
