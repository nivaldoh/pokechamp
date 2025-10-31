from openai import OpenAI
from time import sleep
from openai import RateLimitError
import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

class GPTPlayer():
    def __init__(self, api_key=""):
        if api_key == "":
            self.api_key = os.getenv('OPENAI_API_KEY')
        else:
            self.api_key = api_key
        self.completion_tokens = 0
        self.prompt_tokens = 0

        # Map common model names to official API names
        self.model_mapping = {
            # GPT-4o models
            'gpt-4o': 'gpt-4o',
            'gpt-4o-mini': 'gpt-4o-mini',

            # GPT-5 models
            'gpt-5': 'gpt-5',
            'gpt-5-mini': 'gpt-5-mini',
            'gpt-5-nano': 'gpt-5-nano',

            # Legacy GPT-4 models
            'gpt-4': 'gpt-4',
            'gpt-4-turbo': 'gpt-4-turbo',
            'gpt-4-turbo-preview': 'gpt-4-turbo-preview',

            # GPT-3.5 models
            'gpt-3.5-turbo': 'gpt-3.5-turbo',
        }

        # Models that use max_completion_tokens instead of max_tokens
        self.uses_max_completion_tokens = {
            'gpt-4o', 'gpt-4o-mini',
            'gpt-5', 'gpt-5-mini', 'gpt-5-nano'
        }

        # Models that only support temperature=1 (default)
        self.requires_default_temperature = {
            'gpt-5', 'gpt-5-mini', 'gpt-5-nano'
        }

        # Models that don't support the 'stop' parameter
        self.no_stop_support = {
            'gpt-5', 'gpt-5-mini', 'gpt-5-nano'
        }

        # Models that don't support 'response_format' parameter
        self.no_response_format_support = set()

        # Models that require the Responses API instead of Chat Completions
        self.responses_api_prefixes = (
            'gpt-5',
        )

        # Responses API tuning
        # Default budget for Responses API generations. Keep high enough to
        # capture reasoning traces but low enough to avoid long-running calls.
        self.responses_min_output_tokens = 10000

    def _call_with_timeout(self, func, timeout_seconds, *args, **kwargs):
        """
        Execute a function with a timeout using ThreadPoolExecutor.
        Returns None if timeout is exceeded.

        Args:
            func: Function to execute
            timeout_seconds: Maximum seconds to wait
            *args, **kwargs: Arguments to pass to func

        Returns:
            Function result or None on timeout
        """
        if timeout_seconds is None or timeout_seconds <= 0:
            # No timeout, call directly
            return func(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout_seconds)
                return result
            except FuturesTimeoutError:
                print(f'[GPT] API call timed out after {timeout_seconds} seconds')
                future.cancel()
                return None
            except Exception as e:
                print(f'[GPT] API call failed: {e}')
                raise

    def _uses_max_completion_tokens(self, model: str) -> bool:
        """Check if model uses max_completion_tokens parameter instead of max_tokens."""
        api_model_name = self.model_mapping.get(model, model)
        return any(api_model_name.startswith(prefix) for prefix in self.uses_max_completion_tokens)

    def _get_temperature(self, model: str, requested_temp: float) -> Optional[float]:
        """Get the appropriate temperature value for the model, or None to omit it."""
        api_model_name = self.model_mapping.get(model, model)
        if any(api_model_name.startswith(prefix) for prefix in self.requires_default_temperature):
            # These models only support temperature=1, so omit the parameter to use default
            return None
        return requested_temp

    def _supports_stop(self, model: str) -> bool:
        """Check if model supports the 'stop' parameter."""
        api_model_name = self.model_mapping.get(model, model)
        return not any(api_model_name.startswith(prefix) for prefix in self.no_stop_support)

    def _supports_response_format(self, model: str) -> bool:
        """Check if model supports the 'response_format' parameter for JSON mode."""
        api_model_name = self.model_mapping.get(model, model)
        return not any(api_model_name.startswith(prefix) for prefix in self.no_response_format_support)

    def _build_structured_response_format(self, actions=None):
        """Return a structured response format schema when actions are available."""

        if not actions:
            return {"type": "json_object"}

        def _clean(values) -> List[str]:
            cleaned: List[str] = []

            if isinstance(values, (list, tuple, set)):  # iterable of candidates
                candidates = values
            elif isinstance(values, str):
                candidates = [values]
            else:
                return []

            for value in candidates:
                if value is None:
                    continue
                if isinstance(value, str):
                    text = value.strip()
                else:
                    text = str(value).strip()
                if text:
                    cleaned.append(text)
            return sorted(set(cleaned))

        move_choices: List[str] = []
        switch_choices: List[str] = []
        choice_enum: List[str] = []

        if isinstance(actions, dict):
            move_choices = _clean(actions.get("moves", []))
            switch_choices = _clean(actions.get("switches", []))
            choice_enum = _clean(actions.get("choices", []))
        elif isinstance(actions, (list, tuple)) and not isinstance(actions, (str, bytes)):
            if actions and all(isinstance(item, str) for item in actions):
                choice_enum = _clean(actions)
            elif len(actions) >= 2:
                move_choices = _clean(actions[0])
                switch_choices = _clean(actions[1])
            elif len(actions) == 1:
                move_choices = _clean(actions[0])
        else:
            choice_enum = _clean([actions])

        schema_properties: Dict[str, Any] = {
            "move": {"type": "string"},
            "switch": {"type": "string"},
            "dynamax": {"type": "string"},
            "terastallize": {"type": "string"},
            "thought": {"type": "string"},
            "choice": {"type": "string"},
            "action": {"type": "string"},
            "target": {"type": "string"},
            "decision": {"type": "object"},
            "score": {"type": ["integer", "number", "string"]},
        }

        if move_choices:
            schema_properties["move"]["enum"] = move_choices
            schema_properties["dynamax"] = {"type": "string", "enum": move_choices}
            schema_properties["terastallize"] = {"type": "string", "enum": move_choices}
        if switch_choices:
            schema_properties["switch"]["enum"] = switch_choices
        if choice_enum:
            schema_properties["choice"]["enum"] = choice_enum

        schema = {
            "type": "object",
            "properties": schema_properties,
            "additionalProperties": True,
        }

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "battle_action",
                "schema": schema,
                "strict": False,
            },
            "schema_version": "1.1",
        }

    def _should_use_responses_api(self, api_model_name: str) -> bool:
        """Return True when the model needs to be queried via the Responses API."""
        return any(api_model_name.startswith(prefix) for prefix in self.responses_api_prefixes)

    def _build_responses_payload(self, system_prompt: str, user_prompt: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Construct `input` blocks and optional instructions for the Responses API."""

        input_blocks: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                ],
            }
        ]

        instructions = system_prompt.strip() if system_prompt else None
        return input_blocks, instructions

    @staticmethod
    def _extract_responses_output_text(response) -> str:
        """Extract concatenated text from a Responses API result."""
        # Prefer the convenience helper provided by the SDK when available
        direct_text = getattr(response, "output_text", None)
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text.strip()
        if isinstance(direct_text, (list, tuple)):
            joined = "\n".join(part for part in direct_text if part)
            if joined.strip():
                return joined.strip()

        texts: List[str] = []

        # Handle strongly-typed objects returned by the SDK
        output_items = getattr(response, "output", None)
        if output_items is not None:
            for item in output_items:
                item_type = getattr(item, "type", None)
                if item_type == "message":
                    for content in getattr(item, "content", []) or []:
                        text = getattr(content, "text", "")
                        if text:
                            texts.append(text)
                elif item_type == "reasoning":
                    for content in getattr(item, "content", []) or []:
                        text = getattr(content, "text", "")
                        if text:
                            texts.append(text)
                    for summary in getattr(item, "summary", []) or []:
                        text = getattr(summary, "text", "")
                        if text:
                            texts.append(text)
                else:
                    text = getattr(item, "text", "")
                    if text:
                        texts.append(text)

        if texts:
            return "\n".join(texts).strip()

        # Fall back to dictionary parsing if we weren't handed SDK models
        try:
            response_dict = response.model_dump()
        except AttributeError:
            response_dict = response  # assume mapping-like

        for item in response_dict.get("output", []) or []:
            item_type = item.get("type")
            if item_type == "message":
                for content in item.get("content", []) or []:
                    text = content.get("text")
                    if text:
                        texts.append(text)
            elif item_type == "reasoning":
                for content in item.get("content", []) or []:
                    text = content.get("text")
                    if text:
                        texts.append(text)
                for summary in item.get("summary", []) or []:
                    text = summary.get("text")
                    if text:
                        texts.append(text)
            else:
                text = item.get("text")
                if text:
                    texts.append(text)

        return "\n".join(texts).strip()

    @staticmethod
    def _get_usage_counts(usage_obj) -> Tuple[int, int]:
        """Return (prompt_tokens, completion_tokens) from either API usage schema."""
        if not usage_obj:
            return 0, 0

        prompt_tokens = 0
        completion_tokens = 0

        if isinstance(usage_obj, dict):
            prompt_tokens = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens") or 0
            completion_tokens = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens") or 0
            return prompt_tokens, completion_tokens

        prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
        if prompt_tokens is None:
            prompt_tokens = getattr(usage_obj, "input_tokens", 0)

        completion_tokens = getattr(usage_obj, "completion_tokens", None)
        if completion_tokens is None:
            completion_tokens = getattr(usage_obj, "output_tokens", 0)

        return prompt_tokens or 0, completion_tokens or 0

    @staticmethod
    def _get_incomplete_reason(response) -> Optional[str]:
        details = getattr(response, "incomplete_details", None)
        if not details:
            return None
        if isinstance(details, dict):
            return details.get("reason")
        return getattr(details, "reason", None)

    def _request_responses_completion(self, client, responses_kwargs: Dict[str, Any], model: str, fallback_max_tokens: int) -> Tuple[Any, int, int]:
        """Call Responses API with automatic retries when hitting the max token limit."""

        attempts = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        current_kwargs = dict(responses_kwargs)

        response = client.responses.create(**current_kwargs)
        prompt_tokens, completion_tokens = self._get_usage_counts(response.usage)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        status = getattr(response, "status", None)
        reason = self._get_incomplete_reason(response)

        if status == "incomplete" and reason == "max_output_tokens":
            current_max = current_kwargs.get("max_output_tokens")
            print(
                f"[GPT] Warning: Model {model} hit max_output_tokens={current_max}; "
                "returning incomplete response"
            )

        return response, total_prompt_tokens, total_completion_tokens

    def get_LLM_action(self, system_prompt, user_prompt, model='gpt-4o', temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=200, actions=None, timeout=None) -> str:
        """
        Get LLM action with optional timeout.

        Args:
            timeout: Maximum seconds to wait for API response. None means no timeout.
        """
        def _make_api_call():
            client = OpenAI(api_key=self.api_key)
            # client = AzureOpenAI()

            # Map model name to official API name
            api_model_name = self.model_mapping.get(model, model)

            # Determine if we should use the Responses API for this model
            if self._should_use_responses_api(api_model_name):
                temp_value = self._get_temperature(model, temperature)
                input_blocks, instructions = self._build_responses_payload(system_prompt, user_prompt)
                max_output_tokens = max(max_tokens, self.responses_min_output_tokens)
                responses_kwargs = {
                    "model": api_model_name,
                    "input": input_blocks,
                    "max_output_tokens": max_output_tokens,
                }

                if instructions:
                    responses_kwargs["instructions"] = instructions

                if temp_value is not None:
                    responses_kwargs["temperature"] = temp_value

                response, prompt_tokens, completion_tokens = self._request_responses_completion(
                    client,
                    responses_kwargs,
                    api_model_name,
                    responses_kwargs["max_output_tokens"],
                )

                outputs = self._extract_responses_output_text(response)
                if not outputs:
                    status = getattr(response, "status", None)
                    output_types = [
                        getattr(item, "type", None)
                        for item in getattr(response, "output", []) or []
                    ]
                    print(
                        f'[GPT] Warning: Model {model} returned empty response (status={status}, '
                        f'output_types={output_types})'
                    )
                    outputs = "{}"

                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens

                return outputs, bool(json_format)

            # Determine which token limit parameter to use
            token_param = "max_completion_tokens" if self._uses_max_completion_tokens(model) else "max_tokens"

            # Get appropriate temperature value
            temp_value = self._get_temperature(model, temperature)

            try:
                # Build common kwargs
                base_kwargs = {
                    "model": api_model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    token_param: max_tokens
                }

                # Only add temperature if model supports it
                if temp_value is not None:
                    base_kwargs["temperature"] = temp_value

                # Only add stop if model supports it
                if self._supports_stop(model) and stop:
                    base_kwargs["stop"] = stop

                # Only add response_format if model supports it
                if json_format and self._supports_response_format(model):
                    structured_format = self._build_structured_response_format(actions)
                    if structured_format and structured_format.get("type") == "json_schema":
                        base_kwargs["response_format"] = structured_format
                    else:
                        base_kwargs["response_format"] = {"type": "json_object"}

                response = client.chat.completions.create(**base_kwargs)
            except RateLimitError:
                # sleep 5 seconds and try again
                sleep(5)
                print('rate limit error')
                return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions, timeout)

            outputs = response.choices[0].message.content

            # Handle None/empty or multi-part responses from chat completions
            if isinstance(outputs, list):
                outputs = "".join(part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "") for part in outputs)

            if outputs is None or not str(outputs).strip():
                print(f'[GPT] Warning: Model {model} returned empty response')
                outputs = "{}"  # Return empty JSON object as fallback

            # log completion tokens
            self.completion_tokens += response.usage.completion_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            if json_format:
                return outputs, True
            return outputs, False

        # Execute with timeout if specified
        result = self._call_with_timeout(_make_api_call, timeout)

        # Handle timeout
        if result is None:
            print(f'[GPT] Timeout after {timeout}s, returning fallback')
            return "{}", bool(json_format)

        return result
    
    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.7, model='gpt-4o', json_format=False, seed=None, stop=[], max_tokens=200):
        client = OpenAI(api_key=self.api_key)
        # client = AzureOpenAI()

        # Map model name to official API name
        api_model_name = self.model_mapping.get(model, model)

        if self._should_use_responses_api(api_model_name):
            temp_value = self._get_temperature(model, temperature)
            input_blocks, instructions = self._build_responses_payload(
                system_prompt,
                user_prompt + ('\n{"' if json_format else ''),
            )
            max_output_tokens = max(max_tokens, self.responses_min_output_tokens)
            responses_kwargs = {
                "model": api_model_name,
                "input": input_blocks,
                "max_output_tokens": max_output_tokens,
            }

            if instructions:
                responses_kwargs["instructions"] = instructions

            if temp_value is not None:
                responses_kwargs["temperature"] = temp_value

            response, prompt_tokens, completion_tokens = self._request_responses_completion(
                client,
                responses_kwargs,
                api_model_name,
                responses_kwargs["max_output_tokens"],
            )
            message = self._extract_responses_output_text(response)

            if not message:
                status = getattr(response, "status", None)
                output_types = [
                    getattr(item, "type", None)
                    for item in getattr(response, "output", []) or []
                ]
                print(
                    f'[GPT] Warning: Model {model} returned empty response in get_LLM_query '
                    f'(status={status}, output_types={output_types})'
                )
                message = "{}"

            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

            if json_format:
                return message, True
            return message, False

        # Determine which token limit parameter to use
        token_param = "max_completion_tokens" if self._uses_max_completion_tokens(model) else "max_tokens"

        # Get appropriate temperature value
        temp_value = self._get_temperature(model, temperature)

        try:
            output_padding = ''
            if json_format:
                output_padding  = '\n{"'

            # Build kwargs
            query_kwargs = {
                "model": api_model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt+output_padding}
                ],
                "stream": False,
                token_param: max_tokens
            }

            # Only add temperature if model supports it
            if temp_value is not None:
                query_kwargs["temperature"] = temp_value

            # Only add stop if model supports it
            if self._supports_stop(model) and stop:
                query_kwargs["stop"] = stop

            if json_format and self._supports_response_format(model):
                structured_format = self._build_structured_response_format()
                if structured_format and structured_format.get("type") == "json_schema":
                    query_kwargs["response_format"] = structured_format
                else:
                    query_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**query_kwargs)
            message = response.choices[0].message.content
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)
            print('rate limit error1')
            return self.get_LLM_query(system_prompt, user_prompt, temperature, model, json_format, seed, stop, max_tokens)

        # Handle None/empty response
        if message is None or not message.strip():
            print(f'[GPT] Warning: Model {model} returned empty response in get_LLM_query')
            message = "{}"  # Return empty JSON object as fallback

        if json_format:
            json_start = 0
            json_end = message.find('}') + 1 # find the first "}
            message_json = '{"' + message[json_start:json_end]
            if len(message_json) > 0:
                return message_json, True
        return message, False

    async def get_LLM_action_with_mcp(
        self,
        system_prompt: str,
        user_prompt: str,
        mcp_client,
        model: str = 'gpt-4o',
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_tool_calls: int = 10,
        timeout: Optional[float] = None
    ) -> str:
        """
        Get LLM action using MCP tools (OpenAI function calling).

        This method implements a tool calling loop where the LLM can request
        to use MCP tools, and we execute them and return results.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: Initial user prompt
            mcp_client: Connected MCP client with available tools
            model: OpenAI model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            max_tool_calls: Maximum number of tool call iterations
            timeout: Maximum seconds to wait for entire MCP session. None means no timeout.

        Returns:
            Final LLM response (should be JSON with move/switch)
        """
        import time as time_module
        start_time = time_module.time() if timeout else None

        client = OpenAI(api_key=self.api_key)

        # Map model name to official API name
        api_model_name = self.model_mapping.get(model, model)

        # Convert MCP tools to OpenAI function format
        tools = []
        tools_list = mcp_client.get_tools_list()
        print(f"[GPT MCP] Starting MCP session with {len(tools_list)} available tools")

        for tool in tools_list:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            })

        print(f"[GPT MCP] Converted {len(tools)} tools to OpenAI function format")

        # Determine which token limit parameter to use
        token_param = "max_completion_tokens" if self._uses_max_completion_tokens(model) else "max_tokens"

        # Get appropriate temperature value
        temp_value = self._get_temperature(model, temperature)
        temp_display = temp_value if temp_value is not None else "default (1)"
        print(f"[GPT MCP] Using model: {api_model_name}, temperature: {temp_display}, max_tokens: {max_tokens}")

        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Tool calling loop
        tool_call_count = 0
        while tool_call_count < max_tool_calls:
            # Check timeout
            if start_time and (time_module.time() - start_time) > timeout:
                print(f"[GPT MCP] Timeout after {timeout}s during tool calling loop")
                return json.dumps({"error": "timeout exceeded"})

            try:
                print(f"[GPT MCP] Initiating API request (iteration {tool_call_count + 1}/{max_tool_calls})")

                # Build kwargs for API call
                api_kwargs = {
                    "model": api_model_name,
                    "messages": messages,
                    "tools": tools if tools else None,
                    token_param: max_tokens
                }

                # Only add temperature if model supports it
                if temp_value is not None:
                    api_kwargs["temperature"] = temp_value

                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    **api_kwargs
                )

                # Update token counts
                self.completion_tokens += response.usage.completion_tokens
                self.prompt_tokens += response.usage.prompt_tokens
                print(f"[GPT MCP] Response received - tokens: {response.usage.completion_tokens} completion, {response.usage.prompt_tokens} prompt")

                message = response.choices[0].message

                # Check if LLM wants to call tools
                if message.tool_calls:
                    print(f"[GPT MCP] LLM requested {len(message.tool_calls)} tool call(s)")
                    # Add assistant message to conversation
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in message.tool_calls
                        ]
                    })

                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                        print(f"[GPT MCP] Calling tool: {tool_name} with args: {arguments}")

                        # Call MCP server
                        try:
                            result = await mcp_client.call_tool(tool_name, arguments)
                            print(f"[GPT MCP] Tool result: {result}")
                        except Exception as e:
                            result = {"error": str(e)}
                            print(f"[GPT MCP] Tool error: {e}")

                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result)
                        })

                    tool_call_count += 1
                else:
                    # No more tool calls, return final response
                    print(f"[GPT MCP] LLM returned final decision (no more tool calls after {tool_call_count} iterations)")
                    return message.content if message.content else ""

            except RateLimitError:
                sleep(5)
                print('rate limit error in MCP mode')
                continue
            except Exception as e:
                print(f'Error in MCP tool calling: {e}')
                # Return what we have so far or error
                if messages:
                    return json.dumps({"error": str(e)})
                raise

        # Max tool calls reached, ask for final decision
        print(f"[GPT MCP] Max tool calls ({max_tool_calls}) reached, requesting final decision")

        # Check timeout before final call
        if start_time and (time_module.time() - start_time) > timeout:
            print(f"[GPT MCP] Timeout after {timeout}s before final decision")
            return json.dumps({"error": "timeout exceeded"})

        messages.append({
            "role": "user",
            "content": "You've used many tools. Please make your final decision now as JSON: {\"move\":\"<name>\"} or {\"switch\":\"<name>\"}"
        })

        try:
            print(f"[GPT MCP] Initiating final decision API request")

            # Build kwargs for final API call
            final_api_kwargs = {
                "model": api_model_name,
                "messages": messages,
                token_param: max_tokens
            }

            # Only add temperature if model supports it
            if temp_value is not None:
                final_api_kwargs["temperature"] = temp_value

            response = await asyncio.to_thread(
                client.chat.completions.create,
                **final_api_kwargs
            )
            self.completion_tokens += response.usage.completion_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            print(f"[GPT MCP] Final decision received - tokens: {response.usage.completion_tokens} completion, {response.usage.prompt_tokens} prompt")
            return response.choices[0].message.content
        except Exception as e:
            print(f"[GPT MCP] Error in final decision: {e}")
            return json.dumps({"error": "max tool calls reached without decision"})
