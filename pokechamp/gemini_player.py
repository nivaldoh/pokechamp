import asyncio
import os
import sys
import json
import logging
import random
from time import sleep
from typing import Any, Dict, List, Optional

from google import genai
try:
    from google.genai import errors as genai_errors
except Exception:
    genai_errors = None

class GeminiPlayer():
    def __init__(self, api_key=""):
        if api_key == "":
            self.api_key = os.getenv('GEMINI_API_KEY')
        else:
            self.api_key = api_key

        # Reduce noisy HTTP logging from the underlying transport
        logging.getLogger("google.auth.transport.requests").setLevel(logging.WARNING)
        logging.getLogger("google.auth").setLevel(logging.WARNING)

        # Configure the Gemini API client
        self.client = genai.Client(api_key=self.api_key)

        self.completion_tokens = 0
        self.prompt_tokens = 0
        
        # Map common model names to official API names
        self.model_mapping = {
            # Default to latest Gemini 2.5
            'gemini-flash': 'gemini-2.5-flash',
            'gemini-flash-2.5': 'gemini-2.5-flash',
            'gemini-pro': 'gemini-2.5-pro',
            'gemini-pro-2.5': 'gemini-2.5-pro',
            
            # Gemini 2.0 models
            'gemini-2.0-flash': 'gemini-2.0-flash',
            'gemini-2.0-flash-lite': 'gemini-2.0-flash-lite',
            'gemini-2.0-pro': 'gemini-2.0-pro-experimental',
            'gemini-2.0-pro-experimental': 'gemini-2.0-pro-experimental',
            'gemini-2.0-flash-thinking': 'gemini-2.0-flash-thinking-exp',
            
            # Gemini 1.5 models (legacy support)
            'gemini-1.5-flash': 'gemini-1.5-flash',
            'gemini-1.5-pro': 'gemini-1.5-pro',
        }

    def get_LLM_action(self, system_prompt, user_prompt, model='gemini-2.0-flash', temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=1000, actions=None) -> str:
        # Map model name to official API name
        api_model_name = self.model_mapping.get(model, model)
        # Combine system and user prompts for Gemini
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Single attempt; let callers handle errors/fallbacks
        response = self.client.models.generate_content(model=api_model_name, contents=combined_prompt)
        outputs = response.text
        # Token counters (approximate)
        self.completion_tokens += len(outputs.split()) * 1.3
        self.prompt_tokens += len(combined_prompt.split()) * 1.3

        if json_format:
            start_idx = outputs.find('{')
            end_idx = outputs.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_content = outputs[start_idx:end_idx + 1]
                try:
                    json.loads(json_content)
                    return json_content, True
                except json.JSONDecodeError:
                    return outputs, True
            else:
                return outputs, True

        return outputs, False
    
    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.7, model='gemini-2.0-flash', json_format=False, seed=None, stop=[], max_tokens=1000):
        api_model_name = self.model_mapping.get(model, model)
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Single attempt; let callers handle errors/fallbacks
        response = self.client.models.generate_content(model=api_model_name, contents=combined_prompt)
        message = response.text

        if json_format:
            json_start = 0
            json_end = message.find('}') + 1
            message_json = '{"' + message[json_start:json_end]
            if len(message_json) > 0:
                return message_json, True

        return message, False

    # ------------ Logging helpers ------------
    def _log_block(self, title: str, content: str) -> None:
        try:
            sep = "\n" + ("=" * 28) + f" {title} " + ("=" * 28) + "\n"
            print(sep + (content or "") + "\n" + ("-" * len(sep.strip())) + "\n")
        except Exception:
            pass

    def _schema_from_spec(self, spec: Dict[str, Any]):
        """Recursively convert a simple JSON-like spec into genai.types.Schema."""
        if spec is None:
            return None

        t = (spec.get("type") or "").lower()
        if t == "object":
            props = {}
            for k, v in (spec.get("properties") or {}).items():
                props[k] = self._schema_from_spec(v)
            return genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties=props or None,
            )
        if t == "array":
            item_spec = spec.get("items") or {"type": "string"}
            return genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=self._schema_from_spec(item_spec)
            )
        if t == "string":
            return genai.types.Schema(type=genai.types.Type.STRING)
        if t == "integer":
            return genai.types.Schema(type=genai.types.Type.INTEGER)
        if t == "number":
            return genai.types.Schema(type=genai.types.Type.NUMBER)
        if t == "boolean":
            return genai.types.Schema(type=genai.types.Type.BOOLEAN)
        # Default fallback
        return genai.types.Schema(type=genai.types.Type.STRING)

    async def run_mcp_task(
        self,
        system_prompt: str,
        user_prompt: str,
        mcp_client,
        *,
        model: str = 'gemini-2.5-flash',
        temperature: float = 0.3,
        max_tokens: int = 1000,
        max_tool_calls: int = 5,
        response_schema_spec: Optional[Dict[str, Any]] = None,
        log_title: Optional[str] = None,
    ) -> str:
        """Generic MCP task runner for assistants with a custom response schema."""
        api_model_name = self.model_mapping.get(model, model)

        # Build tool declarations
        tools_list = mcp_client.get_tools_list()
        fn_decls: List[genai.types.FunctionDeclaration] = []
        for tool in tools_list:
            schema = self._convert_schema_to_gemini(tool.get("input_schema", {}))
            fn_decls.append(
                genai.types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=schema,
                )
            )
        tools = [genai.types.Tool(functionDeclarations=fn_decls)] if fn_decls else None

        # Response schema for JSON output
        resp_schema = self._schema_from_spec(response_schema_spec) if response_schema_spec else None

        history: List[genai.types.Content] = [
            genai.types.Content(role="user", parts=[genai.types.Part(text=user_prompt)])
        ]

        # Log prompts
        self._log_block((log_title or "MCP Task") + " SYSTEM", system_prompt)
        self._log_block((log_title or "MCP Task") + " USER", user_prompt)

        tool_call_count = 0
        while tool_call_count < max_tool_calls:
            try:
                # To support both Gemini 2.0 Flash and 2.5 Flash, avoid
                # responseMimeType/responseSchema with tools and minimize
                # special configs. Rely on tool declarations and instructions.
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=api_model_name,
                    contents=history,
                    config=genai.types.GenerateContentConfig(
                        tools=tools,
                        temperature=temperature,
                        systemInstruction=system_prompt,
                        maxOutputTokens=max_tokens,
                    ),
                )

                response_text = ""
                try:
                    response_text = response.text or ""
                except Exception:
                    response_text = ""

                if getattr(response, "candidates", None):
                    parts = getattr(response.candidates[0].content, "parts", []) or []
                else:
                    parts = []

                function_calls: List[genai.types.FunctionCall] = []
                for part in parts:
                    fc = getattr(part, "function_call", None) or getattr(part, "functionCall", None)
                    if fc is not None:
                        function_calls.append(fc)

                if function_calls:
                    history.append(genai.types.Content(role="model", parts=parts))
                    for fc in function_calls:
                        tool_name = fc.name
                        fc_args = getattr(fc, "args", None) or getattr(fc, "arguments", None)
                        arguments = dict(fc_args) if fc_args is not None else {}
                        try:
                            result = await mcp_client.call_tool(tool_name, arguments)
                        except Exception as e:
                            result = {"error": str(e)}
                        history.append(
                            genai.types.Content(
                                role="user",
                                parts=[genai.types.Part.from_function_response(name=tool_name, response={"result": result})],
                            )
                        )
                    tool_call_count += 1
                    continue

                if response_text and response_text.strip():
                    self._log_block((log_title or "MCP Task") + " RESPONSE", response_text)
                    return response_text
                # If empty, loop again
                continue

            except Exception as e:
                # On error, try a best-effort final answer
                try:
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=api_model_name,
                        contents=history,
                        config=genai.types.GenerateContentConfig(
                            systemInstruction=system_prompt,
                            responseMimeType="application/json",
                            responseSchema=resp_schema,
                            maxOutputTokens=max_tokens,
                        ),
                    )
                    text = getattr(response, "text", None) or json.dumps({"error": str(e)})
                    self._log_block((log_title or "MCP Task") + " RESPONSE", text)
                    return text
                except Exception:
                    return json.dumps({"error": str(e)})

        # Max tool calls used: request final JSON
        try:
            history.append(
                genai.types.Content(role="user", parts=[genai.types.Part(text="Please output final JSON now.")])
            )
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=api_model_name,
                contents=history,
                config=genai.types.GenerateContentConfig(
                    systemInstruction=system_prompt,
                    maxOutputTokens=max_tokens,
                    responseMimeType="application/json",
                    responseSchema=resp_schema,
                ),
            )
            text = getattr(response, "text", None) or json.dumps({"error": "max tool calls reached"})
            self._log_block((log_title or "MCP Task") + " RESPONSE", text)
            return text
        except Exception as max_err:
            return json.dumps({"error": f"max tool calls reached: {max_err}"})

    def generate_json_decision(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str = 'gemini-2.5-flash',
        temperature: float = 0.3,
        max_tokens: int = 1000,
        log_title: Optional[str] = None,
    ) -> str:
        """Single non-tool JSON call for the main decider."""
        api_model_name = self.model_mapping.get(model, model)
        try:
            # Log prompts
            self._log_block((log_title or "Decider") + " SYSTEM", system_prompt)
            self._log_block((log_title or "Decider") + " USER", user_prompt)

            # Prefer JSON mode, but some models may reject strict schemas.
            try:
                response = self.client.models.generate_content(
                    model=api_model_name,
                    contents=[genai.types.Content(role="user", parts=[genai.types.Part(text=user_prompt)])],
                    config=genai.types.GenerateContentConfig(
                        temperature=temperature,
                        systemInstruction=system_prompt,
                        maxOutputTokens=max_tokens,
                        responseMimeType="application/json",
                    ),
                )
                text = getattr(response, "text", None)
                self._log_block((log_title or "Decider") + " RESPONSE", text or "")
                return text or "{}"
            except Exception:
                response = self.client.models.generate_content(
                    model=api_model_name,
                    contents=[genai.types.Content(role="user", parts=[genai.types.Part(text=user_prompt)])],
                    config=genai.types.GenerateContentConfig(
                        temperature=temperature,
                        systemInstruction=system_prompt,
                        maxOutputTokens=max_tokens,
                    ),
                )
                text = getattr(response, "text", None)
                self._log_block((log_title or "Decider") + " RESPONSE", text or "")
                return text or "{}"
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def get_LLM_action_with_mcp(
        self,
        system_prompt: str,
        user_prompt: str,
        mcp_client,
        model: str = 'gemini-2.0-flash',
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_tool_calls: int = 20
    ) -> str:
        """
        Get LLM action using MCP tools (Gemini function calling).

        This implements Gemini's function calling interface for MCP tools.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: Initial user prompt
            mcp_client: Connected MCP client with available tools
            model: Gemini model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            max_tool_calls: Maximum number of tool call iterations

        Returns:
            Final LLM response (should be JSON with move/switch)
        """
        # Map model name
        api_model_name = self.model_mapping.get(model, model)

        # Convert MCP tools to Gemini FunctionDeclarations
        tools_list = mcp_client.get_tools_list()
        fn_decls: List[genai.types.FunctionDeclaration] = []
        for tool in tools_list:
            schema = self._convert_schema_to_gemini(tool.get("input_schema", {}))
            fn_decls.append(
                genai.types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=schema,
                )
            )

        tools = [genai.types.Tool(functionDeclarations=fn_decls)] if fn_decls else None

        # Build conversation with proper roles/parts
        history: List[genai.types.Content] = [
            genai.types.Content(role="user", parts=[genai.types.Part(text=user_prompt)])
        ]

        tool_call_count = 0

        while tool_call_count < max_tool_calls:
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=api_model_name,
                    contents=history,
                    config=genai.types.GenerateContentConfig(
                        tools=tools,
                        temperature=temperature,
                        systemInstruction=system_prompt,
                        maxOutputTokens=max_tokens,
                        toolConfig=genai.types.ToolConfig(
                            functionCallingConfig=genai.types.FunctionCallingConfig(
                                mode=genai.types.FunctionCallingConfigMode.AUTO
                            )
                        ),
                        automaticFunctionCalling=genai.types.AutomaticFunctionCallingConfig(
                            maximumRemoteCalls=max_tool_calls
                        ),
                        responseMimeType="application/json",
                        responseSchema=genai.types.Schema(
                            type=genai.types.Type.OBJECT,
                            properties={
                                "move": genai.types.Schema(type=genai.types.Type.STRING),
                                "switch": genai.types.Schema(type=genai.types.Type.STRING),
                            },
                        ),
                    ),
                )

                # Extract response text for token estimation
                response_text = ""
                try:
                    response_text = response.text or ""
                except Exception:
                    response_text = ""

                if response_text:
                    self.completion_tokens += len(response_text.split()) * 1.3
                    # Approximate prompt tokens using last user+system
                    self.prompt_tokens += len((system_prompt + "\n\n" + user_prompt).split()) * 1.3

                # Ensure we have candidates
                if not getattr(response, "candidates", None):
                    if response_text.strip():
                        return response_text
                    continue

                candidate = response.candidates[0]
                parts = getattr(getattr(candidate, "content", None), "parts", None) or []

                # Gather all function calls in this response (handle casing)
                function_calls: List[genai.types.FunctionCall] = []
                for part in parts:
                    fc = getattr(part, "function_call", None)
                    if fc is None:
                        fc = getattr(part, "functionCall", None)
                    if fc is not None:
                        function_calls.append(fc)

                if function_calls:
                    # Add the model message (with function calls) to history
                    history.append(genai.types.Content(role="model", parts=parts))

                    # Execute each function call and add function responses
                    for fc in function_calls:
                        tool_name = fc.name
                        fc_args = getattr(fc, "args", None)
                        if fc_args is None:
                            fc_args = getattr(fc, "arguments", None)
                        arguments = dict(fc_args) if fc_args is not None else {}
                        try:
                            result = await mcp_client.call_tool(tool_name, arguments)
                        except Exception as e:
                            result = {"error": str(e)}
                            print(f"[MCP] Tool error: {e}")

                        # Add functionResponse part as a user message
                        history.append(
                            genai.types.Content(
                                role="user",
                                parts=[
                                    genai.types.Part.from_function_response(
                                        name=tool_name,
                                        response={"result": result},
                                    )
                                ],
                            )
                        )

                    tool_call_count += 1
                    continue

                # No tool calls: try to return the textual decision
                if response_text and response_text.strip():
                    return response_text
                # If no text, continue to next iteration
                continue

            except Exception as e:
                print(f"Gemini MCP error: {e}")
                await asyncio.sleep(1)
                # Try a final best-effort answer without tools
                try:
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=api_model_name,
                        contents=history,
                        config=genai.types.GenerateContentConfig(
                            systemInstruction=(
                                system_prompt
                                + "\n\nPlease provide your final decision as JSON: {\"move\":\"<name>\"} or {\"switch\":\"<name>\"}."
                            ),
                            maxOutputTokens=max_tokens,
                        ),
                    )
                    return response.text or json.dumps({"error": str(e)})
                except Exception:
                    return json.dumps({"error": str(e)})

        # Max tool calls reached: ask for final decision
        try:
            history.append(
                genai.types.Content(
                    role="user",
                    parts=[
                        genai.types.Part(
                            text=(
                                "You've used many tools. Please make your final decision now as JSON: "
                                '{"move":"<name>"} or {"switch":"<name>"}'
                            )
                        )
                    ],
                )
            )
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=api_model_name,
                contents=history,
                config=genai.types.GenerateContentConfig(
                    systemInstruction=system_prompt,
                    maxOutputTokens=max_tokens,
                ),
            )
            return response.text or json.dumps({"error": "max tool calls reached without decision"})
        except Exception as max_calls_error:
            print(f"Gemini max calls error: {max_calls_error}")
            return json.dumps({"error": "max tool calls reached without decision"})

    def _convert_schema_to_gemini(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JSON Schema to Gemini parameter format.

        Args:
            json_schema: JSON Schema definition

        Returns:
            Gemini-compatible parameter schema
        """
        # Gemini uses a similar but slightly different schema format
        # For now, we can mostly pass it through
        gemini_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        if "properties" in json_schema:
            gemini_schema["properties"] = json_schema["properties"]

        if "required" in json_schema:
            gemini_schema["required"] = json_schema["required"]

        return gemini_schema
