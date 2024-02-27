from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import toml
from discord import (
    ApplicationContext,
    Attachment,
    ButtonStyle,
    Embed,
    File,
    Intents,
    Message,
    RawReactionActionEvent,
)
from discord.ext import commands
from discord.ui import View, button
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


def load_config(filename="config.toml"):
    global \
        MONITORED_CHANNEL_IDS, \
        SCAN_LIMIT_BYTES, \
        A1111_IMPORTANT_FIELDS, \
        A1111_PROMPT_SIZE_LIMIT, \
        MESSAGE_EMBED_LIMIT, \
        ATTACH_FILE_SIZE_THRESHOLD
    config = toml.load(filename)
    MONITORED_CHANNEL_IDS = config.get("MONITORED_CHANNEL_IDS", [])
    SCAN_LIMIT_BYTES = config.get("SCAN_LIMIT_BYTES", 10 * 1024**2)  # Default 10 MB
    A1111_IMPORTANT_FIELDS = config.get("A1111_IMPORTANT_FIELDS", [])
    A1111_PROMPT_SIZE_LIMIT = config.get("A1111_PROMPT_SIZE_LIMIT", 1000)
    MESSAGE_EMBED_LIMIT = config.get("MESSAGE_EMBED_LIMIT", 25)
    ATTACH_FILE_SIZE_THRESHOLD = config.get("ATTACH_FILE_SIZE_THRESHOLD", 1980)
    return config


intents = Intents.default() | Intents.message_content | Intents.members
client = commands.Bot(intents=intents)


def get_params_from_string(param_str):
    max_prompt = A1111_PROMPT_SIZE_LIMIT
    output_dict = {}
    parts = param_str.split("Steps: ")
    prompts = parts[0]
    params = "Steps: " + parts[1]
    if "Negative prompt: " in prompts:
        output_dict["Prompt"] = prompts.split("Negative prompt: ")[0]
        output_dict["Negative Prompt"] = prompts.split("Negative prompt: ")[1]
        if len(output_dict["Negative Prompt"]) > max_prompt:
            output_dict["Negative Prompt"] = (
                output_dict["Negative Prompt"][:max_prompt] + "..."
            )
    else:
        output_dict["Prompt"] = prompts
    if len(output_dict["Prompt"]) > max_prompt:
        output_dict["Prompt"] = output_dict["Prompt"][:max_prompt] + "..."
    params = params.split(", ")
    for param in params:
        params = param.split(": ", 1)
        if len(params) == 2:
            output_dict[params[0]] = params[1]
    return output_dict


def get_embed(embed_dict, context: Message):
    embed_dict = embed_dict | {}
    embed = Embed(color=context.author.color)
    count = 0
    for key in A1111_IMPORTANT_FIELDS:
        if count >= MESSAGE_EMBED_LIMIT:
            break
        value = embed_dict.get(key)
        if value is None:
            continue
        embed.add_field(name=key, value=value, inline="Prompt" not in key)
        del embed_dict[key]
        count += 1
    for key, value in embed_dict.items():
        if count >= MESSAGE_EMBED_LIMIT:
            break
        embed.add_field(name=key, value=value, inline="Prompt" not in key)
        count += 1
    embed.set_footer(
        text=f"Posted by {context.author}",
        icon_url=context.author.display_avatar,
    )
    return embed


def read_info_from_image_stealth(image: Image.Image):
    # trying to read stealth pnginfo
    width, height = image.size
    pixels = image.load()

    has_alpha = image.mode == "RGBA"
    mode = None
    compressed = False
    binary_data = ""
    buffer_a = ""
    buffer_rgb = ""
    index_a = 0
    index_rgb = 0
    sig_confirmed = False
    confirming_signature = True
    reading_param_len = False
    reading_param = False
    read_end = False
    for x in range(width):
        for y in range(height):
            if has_alpha:
                r, g, b, a = pixels[x, y]
                buffer_a += str(a & 1)
                index_a += 1
            else:
                r, g, b = pixels[x, y]
            buffer_rgb += str(r & 1)
            buffer_rgb += str(g & 1)
            buffer_rgb += str(b & 1)
            index_rgb += 3
            if confirming_signature:
                if index_a == len("stealth_pnginfo") * 8:
                    decoded_sig = bytearray(
                        int(buffer_a[i : i + 8], 2) for i in range(0, len(buffer_a), 8)
                    ).decode("utf-8", errors="ignore")
                    if decoded_sig in {"stealth_pnginfo", "stealth_pngcomp"}:
                        confirming_signature = False
                        sig_confirmed = True
                        reading_param_len = True
                        mode = "alpha"
                        if decoded_sig == "stealth_pngcomp":
                            compressed = True
                        buffer_a = ""
                        index_a = 0
                    else:
                        read_end = True
                        break
                elif index_rgb == len("stealth_pnginfo") * 8:
                    decoded_sig = bytearray(
                        int(buffer_rgb[i : i + 8], 2)
                        for i in range(0, len(buffer_rgb), 8)
                    ).decode("utf-8", errors="ignore")
                    if decoded_sig in {"stealth_rgbinfo", "stealth_rgbcomp"}:
                        confirming_signature = False
                        sig_confirmed = True
                        reading_param_len = True
                        mode = "rgb"
                        if decoded_sig == "stealth_rgbcomp":
                            compressed = True
                        buffer_rgb = ""
                        index_rgb = 0
            elif reading_param_len:
                if mode == "alpha":
                    if index_a == 32:
                        param_len = int(buffer_a, 2)
                        reading_param_len = False
                        reading_param = True
                        buffer_a = ""
                        index_a = 0
                elif index_rgb == 33:
                    pop = buffer_rgb[-1]
                    buffer_rgb = buffer_rgb[:-1]
                    param_len = int(buffer_rgb, 2)
                    reading_param_len = False
                    reading_param = True
                    buffer_rgb = pop
                    index_rgb = 1
            elif reading_param:
                if mode == "alpha":
                    if index_a == param_len:
                        binary_data = buffer_a
                        read_end = True
                        break
                elif index_rgb >= param_len:
                    diff = param_len - index_rgb
                    if diff < 0:
                        buffer_rgb = buffer_rgb[:diff]
                    binary_data = buffer_rgb
                    read_end = True
                    break
            else:
                # impossible
                read_end = True
                break
        if read_end:
            break
    decoded_data = None
    if sig_confirmed and binary_data != "":
        # Convert binary string to UTF-8 encoded text
        byte_data = bytearray(
            int(binary_data[i : i + 8], 2) for i in range(0, len(binary_data), 8)
        )
        try:
            if compressed:
                decoded_data = gzip.decompress(bytes(byte_data)).decode("utf-8")
            else:
                decoded_data = byte_data.decode("utf-8", errors="ignore")
        except Exception as e:
            print(e)
    return decoded_data


@client.event
async def on_ready():
    print(f"Logged in as {client.user}!")


@client.event
async def on_message(message: Message):
    if message.channel.id in MONITORED_CHANNEL_IDS and message.attachments:
        attachments = [
            a
            for a in message.attachments
            if a.filename.lower().endswith(".png") and a.size < SCAN_LIMIT_BYTES
        ]
        for i, attachment in enumerate(
            attachments,
        ):  # download one at a time as usually the first image is already ai-generated
            metadata = OrderedDict()
            await read_attachment_metadata(i, attachment, metadata)
            if metadata:
                await message.add_reaction("🔎")
                return


class MyView(View):
    def __init__(self):
        super().__init__(timeout=3600, disable_on_timeout=True)
        self.metadata = None

    @button(label="Full Parameters", style=ButtonStyle.green)
    async def details(self, button, interaction):
        button.disabled = True
        await interaction.response.edit_message(view=self)
        if len(self.metadata) > ATTACH_FILE_SIZE_THRESHOLD:
            with io.StringIO() as f:
                f.write(self.metadata)
                f.seek(0)
                await interaction.followup.send(file=File(f, "parameters.yaml"))
        else:
            await interaction.followup.send(f"```yaml\n{self.metadata}```")


def populate_attachment_metadata(
    i: int,
    image_data: bytes,
    metadata: OrderedDict,
):
    with Image.open(io.BytesIO(image_data)) as img:
        # try:
        #     info = img.info['parameters']
        # except:
        #     info = read_info_from_image_stealth(img)
        if not img.info:
            return
        if "parameters" in img.info:
            info = img.info["parameters"]
        elif "prompt" in img.info:
            info = img.info["prompt"]
        elif img.info.get("Software") == "NovelAI":
            info = img.info["Description"] + img.info["Comment"]
        else:
            info = None
        # else:
        #    info = read_info_from_image_stealth(img)

        if info:
            metadata[i] = info


async def read_attachment_metadata(
    i: int,
    attachment: Attachment,
    metadata: OrderedDict,
):
    """Allows downloading in bulk"""
    try:
        image_data = await attachment.read()
        populate_attachment_metadata(i, image_data, metadata)
    except Exception as error:
        print(f"{type(error).__name__}: {error}")


@client.event
async def on_raw_reaction_add(ctx: RawReactionActionEvent):
    """Send image metadata in reacted post to user DMs"""
    if (
        ctx.emoji.name != "🔎"
        or ctx.channel_id not in MONITORED_CHANNEL_IDS
        or ctx.member.bot
    ):
        return
    channel = client.get_channel(ctx.channel_id)
    message = await channel.fetch_message(ctx.message_id)
    if not message:
        return
    attachments = [
        a for a in message.attachments if a.filename.lower().endswith(".png")
    ]
    if not attachments:
        return
    metadata = OrderedDict()
    tasks = [
        read_attachment_metadata(i, attachment, metadata)
        for i, attachment in enumerate(attachments)
    ]
    await asyncio.gather(*tasks)
    if not metadata:
        return
    user_dm = await client.get_user(ctx.user_id).create_dm()
    try:
        for attachment, data in [
            (attachments[i], data) for i, data in metadata.items()
        ]:
            if "Steps:" in data:
                params = get_params_from_string(data)
                embed = get_embed(params, message)
                embed.set_image(url=attachment.url)
                custom_view = MyView()
                custom_view.metadata = data
                await user_dm.send(view=custom_view, embed=embed, mention_author=False)
            else:
                img_type = "ComfyUI" if '"inputs"' in data else "NovelAI"
                embed = Embed(
                    title=img_type + " Parameters",
                    color=message.author.color,
                )
                embed.set_footer(
                    text=f"Posted by {message.author}",
                    icon_url=message.author.display_avatar,
                )
                embed.set_image(url=attachment.url)
                await user_dm.send(embed=embed, mention_author=False)
                with io.StringIO() as f:
                    f.write(data)
                    f.seek(0)
                    await user_dm.send(file=File(f, "parameters.yaml"))

    except Exception as error:
        print(f"{type(error).__name__}: {error}")


async def collect_attachments(
    ctx: ApplicationContext,
    message: Message,
):
    attachments = [
        a for a in message.attachments if a.filename.lower().endswith(".png")
    ]
    if not attachments:
        await ctx.respond("This post contains no matching images.", ephemeral=True)
        return None, None
    await ctx.defer(ephemeral=True)
    metadata = OrderedDict()
    tasks = [
        read_attachment_metadata(i, attachment, metadata)
        for i, attachment in enumerate(attachments)
    ]
    await asyncio.gather(*tasks)
    if not metadata:
        await ctx.respond(
            "This post contains no image generation data.",
            ephemeral=True,
        )
        return None, None
    return metadata, attachments


@client.message_command(name="View Prompt")
async def message_command_view_prompt(ctx: ApplicationContext, message: Message):
    """Get raw list of parameters for every image in this post."""
    metadata, _attachments = await collect_attachments(ctx, message)
    if not metadata:
        return
    response = "\n\n".join(metadata.values())
    if len(response) < ATTACH_FILE_SIZE_THRESHOLD:
        await ctx.respond(f"```yaml\n{response}```", ephemeral=True)
    else:
        with io.StringIO() as f:
            f.write(response)
            f.seek(0)
            await ctx.respond(file=File(f, "parameters.yaml"), ephemeral=True)


@client.message_command(name="View Prompt (Get a DM)")
async def message_command_view_prompt_dm(ctx: ApplicationContext, message: Message):
    """Send image metadata in selected post to user DM."""
    user_dm = await client.get_user(ctx.author.id).create_dm()
    metadata, attachments = await collect_attachments(ctx, message)
    if not metadata:
        return
    try:
        for attachment, data in [
            (attachments[i], data) for i, data in metadata.items()
        ]:
            if "Steps:" in data:
                params = get_params_from_string(data)
                embed = get_embed(params, message)
                embed.set_image(url=attachment.url)
                custom_view = MyView()
                custom_view.metadata = data
                await user_dm.send(view=custom_view, embed=embed, mention_author=False)
            else:
                img_type = "ComfyUI" if '"inputs"' in data else "NovelAI"
                embed = Embed(
                    title=img_type + " Parameters",
                    color=message.author.color,
                )
                embed.set_footer(
                    text=f"Posted by {message.author}",
                    icon_url=message.author.display_avatar,
                )
                embed.set_image(url=attachment.url)
                await user_dm.send(embed=embed, mention_author=False)
                with io.StringIO() as f:
                    f.write(data)
                    f.seek(0)
                    await user_dm.send(file=File(f, "parameters.yaml"))

    except Exception as error:
        print(f"{type(error).__name__}: {error}")
        await ctx.respond(
            "Something went wrong. Please tell @dogarrowtype I'm struggling to function out here in these mean streets.",
            ephemeral=True,
        )
        return

    await ctx.respond("DM sent!", ephemeral=True)


def set_comfy_input(result, node_class, node_id, key, inputs, typ=str):
    val = inputs.get(key)
    if val is None or not isinstance(val, typ):
        return
    k = f"{node_class}.{node_id}"
    vals = result.get(k)
    if vals is None:
        result[k] = {key: val}
    else:
        vals[key] = val


COMFY_HANDLERS = {
    "checkpointloadersimple": (("ckpt_name", str),),
    "vaeloader": (("vae_name", str),),
    "cliptextencode": (("text", str),),
    "text multiline": (("text", str),),
    "emptylatentimage": (("width", int), ("height", int)),
    "promptcontrolsimple": (("positive", str), ("negative", str)),
    "clipsetlastlayer": (("stop_at_clip_layer", int),),
}


def extract_comfy_metadata(data, result=None, dump_unhandled=False):
    if result is None:
        result = {}
    for k, v in data.items():
        inputs = v.get("inputs")
        typ = v.get("class_type", "").strip()
        handler = COMFY_HANDLERS.get(typ.lower())
        if not inputs or not handler:
            if dump_unhandled:
                print(f"Unhandled: {k} = {v!r}")
            continue
        for input_name, input_class in handler:
            set_comfy_input(result, typ, k, input_name, inputs, input_class)
    return result


def handle_check(filename: Path):
    with filename.open("rb") as fp:
        file_data = fp.read()
    metadata = OrderedDict()
    populate_attachment_metadata(0, file_data, metadata)
    if not metadata:
        print("* No metadata")
        return
    data = metadata[0]
    if "Steps:" in data:
        print("* Image appears to be A1111 format")
        params = get_params_from_string(data)
        for k, v in params.items():
            print(f"{k.strip():<28}: {v.strip()}")
        return
    try:
        json_data = json.loads(data)
    except json.decode.JSONDecodeError as err:
        print(f"! Cannot parse metadata or unknown metadata type: {err}")
        sys.exit(1)
    params = extract_comfy_metadata(json_data)
    for k, v in params.items():
        print(f"\n{k.strip()}:")
        for vk, vv in v.items():
            print(f"  {vk.strip()}: {str(vv).strip()}")


def main():
    parser = argparse.ArgumentParser(description="Prompt inspector bot")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.toml",
        help="Configuration file",
    )
    parser.add_argument(
        "-d",
        "--dump",
        type=Path,
        help="Check metadata for the specified file",
    )
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    load_config(args.config)
    if args.dump:
        handle_check(args.dump)
        return
    # Otherwise run the bot
    client.run(os.environ["BOT_TOKEN"])


if __name__ == "__main__":
    main()
