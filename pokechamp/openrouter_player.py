from openai import OpenAI
from time import sleep
from openai import RateLimitError
import os
import json

class OpenRouterPlayer():
    def __init__(self, api_key=""):
        if api_key == "":
            self.api_key = os.getenv('OPENROUTER_API_KEY')
        else:
            self.api_key = api_key
        self.completion_tokens = 0
        self.prompt_tokens = 0
        
        # Optional headers for OpenRouter leaderboards
        self.site_url = os.getenv('OPENROUTER_SITE_URL', 'https://github.com/pokechamp')
        self.site_name = os.getenv('OPENROUTER_SITE_NAME', 'PokeChamp')

    def get_LLM_action(self, system_prompt, user_prompt, model='openai/gpt-4o', temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=200, actions=None) -> str:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        try:
            if json_format:
                response = client.chat.completions.create(
                    response_format={"type": "json_object"},
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    stream=False,
                    stop=stop,
                    max_tokens=max_tokens,
                    extra_headers={
                        "HTTP-Referer": self.site_url,
                        "X-Title": self.site_name,
                    }
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    stream=False,
                    stop=stop,
                    max_tokens=max_tokens,
                    extra_headers={
                        "HTTP-Referer": self.site_url,
                        "X-Title": self.site_name,
                    }
                )
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)  
            print('rate limit error')
            return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions)
        except Exception as e:
            print(f'OpenRouter API error: {e}')
            # sleep 2 seconds and try again
            sleep(2)
            return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions)
            
        outputs = response.choices[0].message.content
        # log completion tokens
        self.completion_tokens += response.usage.completion_tokens
        self.prompt_tokens += response.usage.prompt_tokens
        
        if json_format:
            # Handle cases where the model adds extra text before the JSON
            # Look for the first { and last } to extract JSON
            start_idx = outputs.find('{')
            end_idx = outputs.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_content = outputs[start_idx:end_idx + 1]
                try:
                    # Validate JSON
                    json.loads(json_content)
                    return json_content, True
                except json.JSONDecodeError:
                    # If JSON is invalid, return the original output
                    return outputs, True
            else:
                # No JSON found, return original output
                return outputs, True
        
        return outputs, False
    
    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.7, model='openai/gpt-4o', json_format=False, seed=None, stop=[], max_tokens=200):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        try:
            output_padding = ''
            if json_format:
                output_padding  = '\n{"'
                
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt+output_padding}
                ],
                temperature=temperature,
                stream=False,
                stop=stop,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                }
            )
            message = response.choices[0].message.content
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)  
            print('rate limit error1')
            return self.get_LLM_query(system_prompt, user_prompt, temperature, model, json_format, seed, stop, max_tokens)
        except Exception as e:
            print(f'OpenRouter API error: {e}')
            # sleep 2 seconds and try again
            sleep(2)
            return self.get_LLM_query(system_prompt, user_prompt, temperature, model, json_format, seed, stop, max_tokens)
        
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
        model: str = 'openai/gpt-4o',
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_tool_calls: int = 10
    ) -> str:
        """Get LLM action via OpenRouter using MCP tools."""

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        # Convert MCP tools to OpenAI function format
        tools = []
        for tool in mcp_client.get_tools_list():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        tool_call_count = 0

        while tool_call_count < max_tool_calls:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools if tools else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers={
                        "HTTP-Referer": self.site_url,
                        "X-Title": self.site_name,
                    }
                )

                self.completion_tokens += response.usage.completion_tokens
                self.prompt_tokens += response.usage.prompt_tokens

                message = response.choices[0].message

                if message.tool_calls:
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

                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                        try:
                            result = await mcp_client.call_tool(tool_name, arguments)
                        except Exception as e:
                            result = {"error": str(e)}

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result)
                        })

                    tool_call_count += 1
                else:
                    return message.content if message.content else ""

            except RateLimitError:
                sleep(5)
                print('OpenRouter rate limit error in MCP mode')
                continue
            except Exception as e:
                print(f'OpenRouter MCP call failed: {e}')
                if messages:
                    return json.dumps({"error": str(e)})
                raise

        # After exhausting tool calls, request final answer
        messages.append({
            "role": "user",
            "content": "You've used many tools. Please provide the final decision as JSON: {\"move\":\"<name>\"} or {\"switch\":\"<name>\"}."
        })

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                }
            )
            self.completion_tokens += response.usage.completion_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": f"unable to obtain final decision: {e}"})
