Forked from https://github.com/sALTaccount/PromptInspectorBot

# Prompt Inspector 🔎
Inspect prompts 🔎 from images uploaded to discord

> **Note:** This is the `no-privileged-intents` branch. It runs with **zero privileged
> gateway intents** and only supports the context-menu workflow described below. The
> `main` branch instead requires the Message Content and Server Members intents and
> additionally provides automatic scanning and click-🔎-to-DM.

## Functionality

This Discord bot inspects images for generation metadata from Automatic1111's WebUI, ComfyUI,
and NovelAI. It uses **no privileged gateway intents** — nothing needs to be enabled in the
Discord developer portal.

Two context menu apps are provided. Right-click (or long-press) a message → **Apps**:

- **View Prompt** — sends an ephemeral response in the channel with the image generation information.
- **View Prompt (Get a DM)** — sends the image generation information to your DMs.

When metadata is found, the bot leaves a 🔎 reaction on the post, so others can see it's
inspectable and run "View Prompt" themselves.

## Setup

1. Clone the repository
2. Enter the directory
3. Create a venv with `python3 -m venv ./venv`
4. Install the dependencies with `pip3 install -r requirements.txt`
5. Create a Discord bot and invite it to your server
6. Create a file named ".env" in the root directory of the project
7. Set `BOT_TOKEN=<your discord bot token>` in the .env file
8. Copy the `config.example.toml` to `config.toml`
9. Add the channel IDs for channels you want the bot to watch, and set the settings you want in the `config.toml` file
10. Run the bot with `python3 PromptInspector.py`

## Examples
![Example 1](images/2023-03-09_00-14.png)
![Example 2](images/2023-03-09_00-14_1.png)
