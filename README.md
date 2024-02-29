Forked from https://github.com/sALTaccount/PromptInspectorBot

# Prompt Inspector 🔎
Inspect prompts 🔎 from images uploaded to discord

## Functionality

This Discord bot reacts to any image with generation metadata from Automatic1111's WebUI and ComfyUI.
If generation metadata is detected, a magnifying glass react is added to the image. If the user
clicks the magnifying glass, they are sent a DM with the image generation information.

Two context menu apps are provided.
One sends an ephemeral response in the channel.
The other activates the emoji reaction (🔎) DM based response (useful for channels the bot isn't watching).

## Setup

1. Clone the repository
2. Enter the directory
3. Create a venv with `python3 -m venv ./venv`
4. Install the dependencies with `pip3 install -r requirements.txt`
5. Create a Discord bot and invite it to your server
6. Enable the `Message Content Intent` in the Discord developer portal
7. Enable the `Server Members Intent` in the Discord developer portal
8. Create a file named ".env" in the root directory of the project
9. Set `BOT_TOKEN=<your discord bot token>` in the .env file
10. Copy the `config.example.toml` to `config.toml`
11. Add the channel IDs for channels you want the bot to watch, and set the settings you want in the `config.toml` file
12. Run the bot with `python3 PromptInspector.py`

## Examples
![Example 1](images/2023-03-09_00-14.png)
![Example 2](images/2023-03-09_00-14_1.png)
