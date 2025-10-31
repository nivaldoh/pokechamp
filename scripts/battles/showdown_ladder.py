import asyncio
from tqdm import tqdm
import argparse
import os
from common import *
# poke_env local files are incomplete and don't have everything metamon requires when integrating both codebases
# so the solution was to pip install poke-env separately but still keep the local modifications in a different folder
# to avoid shadowing.
# from poke_env.player.team_util import get_llm_player, get_metamon_teams, load_random_team, TeamSet
from pokechamp.poke_env_local.player.team_util import (
    get_llm_player, get_metamon_teams, load_random_team, TeamSet
)

parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--prompt_algo", default="minimax", choices=prompt_algos)
parser.add_argument("--battle_format", default="gen9ou", choices=["gen8randombattle", "gen8ou", "gen9ou", "gen9randombattle"])
parser.add_argument("--primary_model", type=str, default="gpt-5-mini", choices=[
    # OpenAI models
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    # Anthropic Claude models (direct via Anthropic API)
    "claude-sonnet-4-5", "claude-opus-4-1", "claude-sonnet-4", "claude-3-5-haiku",
    # Anthropic models (via OpenRouter)
    "anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus", "anthropic/claude-3-haiku",
    # Google models
    "google/gemini-pro", "gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
    # Meta models
    "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-8b-instruct",
    # Mistral models
    "mistralai/mistral-7b-instruct", "mistralai/mixtral-8x7b-instruct",
    # Cohere models
    "cohere/command-r-plus", "cohere/command-r",
    # Perplexity models
    "perplexity/llama-3.1-sonar-small-128k", "perplexity/llama-3.1-sonar-large-128k",
    # DeepSeek models
    "deepseek-ai/deepseek-coder-33b-instruct", "deepseek-ai/deepseek-llm-67b-chat",
    # Microsoft models
    "microsoft/wizardlm-2-8x22b", "microsoft/phi-3-medium-128k-instruct",
    # Ollama models
    "ollama/gpt-oss:20b", "ollama/llama3.1:8b", "ollama/qwen2.5:32b", "ollama/gemma3:4b", "ollama/gemma3:27b",
    # Local models (via OpenRouter)
    "llama", 'None'
])
parser.add_argument("--secondary_model", type=str, default="gemini-2.5-flash", choices=[
    # Google models
    "google/gemini-pro", "gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
    # Fallbacks (if someone wants assistants on other providers via OpenRouter)
    "gpt-4o-mini", "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.1-8b-instruct", "llama"
])
parser.add_argument("--log_dir", type=str, default="./logs")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default='pokechamp', choices=bot_choices)
parser.add_argument("--USERNAME", type=str, default='')
parser.add_argument("--PASSWORD", type=str, default='')
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--team_dir", type=str, default=None,
                   help="Path to directory containing local team files (*.{battle_format}_team). If not provided, uses HuggingFace teams.")
args = parser.parse_args()
    
async def main():
    player = get_llm_player(args, 
                            args.primary_model, 
                            args.prompt_algo, 
                            args.name, 
                            device=args.device,
                            battle_format=args.battle_format, 
                            online=True, 
                            USERNAME=args.USERNAME, 
                            PASSWORD=args.PASSWORD,
                            primary_model=args.primary_model,
                            secondary_model=args.secondary_model)
    
    if args.team_dir:
        teamloader = TeamSet(args.team_dir, args.battle_format)
    else:
        teamloader = get_metamon_teams(args.battle_format, "competitive")
    
    if not 'random' in args.battle_format:
        # Set teamloader on player for rejection recovery
        player.set_teamloader(teamloader)
        player.update_team(teamloader.yield_team())

    # Playing n_challenges games on the ladder
    n_challenges = args.N
    pbar = tqdm(total=n_challenges)
    wins = 0
    for i in range(n_challenges):
        print('starting ladder')

        # Track wins before this battle
        wins_before = player.n_won_battles

        await player.ladder(1)

        # Check if we won this specific battle
        winner = 'opponent'
        if player.n_won_battles > wins_before:
            winner = args.name
            wins += 1

        # Update team for next match
        if not 'random' in args.battle_format:
            player.update_team(teamloader.yield_team())

        pbar.set_description(f"{wins/(i+1)*100:.2f}%")
        pbar.update(1)
        print(winner)

    print(f'player 2 winrate: {wins/n_challenges*100}')

if __name__ == "__main__":
    asyncio.run(main())
