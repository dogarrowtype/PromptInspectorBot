from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType
from typing import Any, Sequence

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

log = None


class __f:  # noqa: N801
    def __init__(self, fmt, /, *args: Sequence[Any], **kwargs: dict[Any, Any]):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.fmt.format(*self.args, **self.kwargs)


class Config:
    class _EmptyValue:
        pass

    FIELDS = (
        ("monitored_channel_ids", set()),
        ("scan_limit_bytes", 10 * 1024**2),  # Default 10 MB
        (
            "a1111_important_fields",
            (
                "Prompt",
                "Negative Prompt",
                "Steps",
                "Sampler",
                "CFG scale",
                "Seed",
                "Size",
                "Model",
                "VAE",
                "Denoising strength",
                "Hires upscale",
                "Hires steps",
                "Hires upscaler",
                "Version",
            ),
        ),
        ("a1111_prompt_size_limit", 1000),
        ("comfyui_show_titles", True),
        ("comfyui_extract_widget_values", True),
        ("comfyui_prioritize_nodes_with_text", True),
        ("comfyui_replace_field_newlines", True),
        ("message_embed_limit", 25),
        ("attach_file_size_threshold", 1980),
        ("react_on_no_metadata", False),
        ("log_color", False),
        ("log_level", "INFO"),
    )

    def __init__(self):
        self.set_defaults()

    def set_defaults(self):
        for k, v in self.FIELDS:
            setattr(self, k, v)

    def load(self, filepath: Path | str = "config.toml"):
        empty = self._EmptyValue
        cfg = toml.load(filepath)
        for k, _v in self.FIELDS:
            cfgval = cfg.get(k.upper(), empty)
            if cfgval is not empty:
                if k == "monitored_channel_ids":
                    cfgval = set(cfgval)
                setattr(self, k, cfgval)


CFG = Config()

intents = Intents.default() | Intents.message_content | Intents.members
client = commands.Bot(intents=intents)


def reformat_json(string: str, indent: int = 1) -> None | str:
    try:
        data = json.loads(string)  # put JSON-data to a variable
        return json.dumps(data, indent=indent)
    except json.decoder.JSONDecodeError:
        # not a valid JSON
        return None


class InspectAttachmentView(View):
    TXTBLOCK_TYPES = MappingProxyType(
        {"txt": "plaintext", "json": "json", "yaml": "yaml"},
    )

    def __init__(
        self,
        timeout=3600,
        text_metadata: None | str = None,
        content_type="text/plain",
        content_extension="txt",
        **kwargs: dict[str, Any],
    ):
        super().__init__(timeout=timeout, disable_on_timeout=True)
        if text_metadata is not None:
            text_metadata = text_metadata.strip()
        self.text_metadata = text_metadata
        self.content_type = content_type
        self.content_extension = content_extension
        self.kwargs = kwargs

    @button(label="Full Parameters", style=ButtonStyle.green)
    async def details(self, button, interaction):
        button.disabled = True
        await interaction.response.edit_message(view=self)
        if not self.text_metadata:
            await interaction.followup.send("No metadata to send!", **self.kwargs)
            return
        json_metadata = reformat_json(self.text_metadata)
        if len(self.text_metadata) <= CFG.attach_file_size_threshold:
            if json_metadata is None:
                typ = self.TXTBLOCK_TYPES.get(self.content_extension, "plaintext")
                followup_text = f"```{typ}\n{self.text_metadata}```"
            else:
                followup_text = f"```json\n{json_metadata}```"
            await interaction.followup.send(followup_text, **self.kwargs)
            return
        with io.StringIO() as f:
            if json_metadata is None:
                f.write(self.text_metadata)
                followup_file_name = f"parameters.{self.content_extension}"
            else:
                f.write(json_metadata)
                followup_file_name = f"parameters.json"
            f.seek(0)
            await interaction.followup.send(file=File(f, followup_file_name), **self.kwargs)


class Metadata:
    NAME = "Unknown"
    ALLOW_INLINE_EMBEDS = True

    def __init__(self, s):
        self.text_metadata = s
        self.params = self.get_params_from_string(s)

    def get_params_from_string(*args: Sequence[Any], **kwargs: dict[Any, Any]):
        raise NotImplementedError

    def get_embed_view(self, msg_ctx: Message, attachment=None, ephemeral=False):
        embed = self.get_embed(msg_ctx, attachment=attachment)
        if ephemeral:
            view = InspectAttachmentView(
                text_metadata=self.text_metadata,
                ephemeral=True,
            )
        else:
            view = InspectAttachmentView(text_metadata=self.text_metadata)
        return embed, view

    def get_embed(
        self,
        msg_ctx: Message,
        attachment=None,
        prioritize_fields: tuple[str] = (),
    ):
        embed_dict = self.params | {}
        embed = Embed(title=f"{self.NAME} Parameters", color=msg_ctx.author.color)
        count = 0
        for key in prioritize_fields:
            if count >= CFG.message_embed_limit:
                break
            value = embed_dict.get(key)
            if value is None:
                continue
            embed.add_field(
                name=key,
                value=value,
                inline=self.ALLOW_INLINE_EMBEDS
                and "Prompt" not in key
                and len(value) < 32,
            )
            del embed_dict[key]
            count += 1
        for key, value in embed_dict.items():
            if count >= CFG.message_embed_limit:
                break
            embed.add_field(
                name=key,
                value=value,
                inline=self.ALLOW_INLINE_EMBEDS
                and "Prompt" not in key
                and len(value) < 32,
            )
            count += 1
        embed.set_footer(
            text=f"Posted by {msg_ctx.author}",
            icon_url=msg_ctx.author.display_avatar,
        )
        if attachment is not None:
            embed.set_image(url=attachment.url)
        return embed


class MetadataA1111(Metadata):
    NAME = "A1111"
    CONTENT_TYPE = "text/plain"
    EXTENSION = "txt"

    def get_embed(self, msg_ctx: Message, attachment=None):
        return super().get_embed(
            msg_ctx,
            attachment=attachment,
            prioritize_fields=CFG.a1111_important_fields,
        )

    def get_params_from_string(self, param_str: str) -> OrderedDict[str, str]:
        # TODO: try to adapt A1111's own metadata parsing code: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cf2772fab0af5573da775e7437e6acdca424f26e/modules/generation_parameters_copypaste.py#L211
        max_prompt = CFG.a1111_prompt_size_limit
        output_dict = OrderedDict()
        parts = param_str.split("Steps: ", 1)
        if len(parts) != 2:
            raise ValueError("Can't parse A1111 metadata: missing Steps key")
        prompts = parts[0]
        params = "Steps: " + parts[1]
        neg_parts = (
            prompts.split("Negative prompt: ", 1)
            if "Negative prompt: " in prompts
            else ()
        )
        if neg_parts:
            output_dict["Prompt"] = neg_parts[0].strip()
            output_dict["Negative Prompt"] = neg_parts[1].strip()
        else:
            output_dict["Prompt"] = prompts.strip()
        params = params.split(", ")
        for param in params:
            params = param.split(": ", 1)
            if len(params) == 2:
                output_dict[params[0].strip()] = params[1].strip()
        for k, v in output_dict.items():
            if len(v) > max_prompt:
                output_dict[k] = v[:max_prompt] + "..."
        return output_dict


class MetadataComfyUI(Metadata):
    NAME = "ComfyUI"
    CONTENT_TYPE = "application/json"
    EXTENSION = ".json"
    ALLOW_INLINE_EMBEDS = False

    # This is just a read-only dict.
    COMFY_HANDLERS = MappingProxyType(
        {
            # Handler format: input name, input type, [optional widget name]
            # When widget name is present and conforms to the expected type its value will
            # replace the node input value.
            "checkpointloadersimple": (("ckpt_name", str),),
            "vaeloader": (("vae_name", str),),
            "cliptextencode": (("text", str),),
            "clipsetlastlayer": (("stop_at_clip_layer", int),),
            "cliptextencodesdxl": (("text_l", str), ("text_g", str)),
            "cliptextencodeperpweight": (("text", str),),
            "bnk_cliptextencodeadvanced": (("text", str),),
            "bnk_cliptextencodesdxladvanced": (("text", str),),
            "editableclipencode": (("text", str),),
            "text multiline": (("text", str),),
            "emptylatentimage": (("width", int), ("height", int)),
            "promptcontrolsimple": (("positive", str), ("negative", str)),
            "ksampler": (
                ("seed", int),
                ("steps", int),
                ("cfg", float),
                ("sampler_name", str),
                ("scheduler", str),
            ),
            "ksampleradvanced": (
                ("noise_seed", int),
                ("steps", int),
                ("start_at_step", int),
                ("end_at_step", int),
                ("cfg", float),
                ("sampler_name", str),
                ("scheduler", str),
            ),
            "samplercustom": (
                ("noise_seed", int),
                ("cfg", float),
            ),
            "ksampler with restarts (simple)": (
                ("seed", int),
                ("steps", int),
                ("cfg", float),
                ("sampler_name", str),
                ("scheduler", str),
            ),
            "ksampler with restarts": (
                ("seed", int),
                ("steps", int),
                ("cfg", float),
                ("sampler_name", str),
                ("scheduler", str),
                ("restart_scheduler", str),
            ),
            "ksampler with restarts (advanced)": (
                ("noise_seed", int),
                ("steps", int),
                ("cfg", float),
                ("sampler_name", str),
                ("scheduler", str),
                ("restart_scheduler", str),
            ),
            "ksampler with restarts (custom)": (
                ("noise_seed", int),
                ("steps", int),
                ("cfg", float),
                ("scheduler", str),
                ("restart_scheduler", str),
            ),
            "efficient_loader": (
                ("ckpt_name", str),
                ("vae_name", str),
                ("clip_skip", int),
                ("clip_positive", str),
                ("clip_negative", str),
                ("empty_latent_width", int),
                ("empty_latent_height", int),
            ),
            "checkpointloader|pysssss": (("ckpt_name", str),),
            "checkpoint loader (simple)": (("ckpt_name", str),),  # WAS Node Suite
            "ttn pipeloader": (
                ("ckpt_name", str),
                ("vae_name", str),
                ("clip_skip", int),
                ("positive", str),
                ("negative", str),
                ("empty_latent_width", int),
                ("empty_latent_height", int),
                ("seed", int),
            ),
            "ttn pipeloadersdxl": (
                ("ckpt_name", str),
                ("vae_name", str),
                ("clip_skip", int),
                ("positive", str),
                ("negative", str),
                ("empty_latent_width", int),
                ("empty_latent_height", int),
                ("seed", int),
            ),
            "showtext|pysssss": (("text", str, "text"),),
        },
    )

    def __init__(self, prompt: str, workflow: None | str):
        self.text_metadata = prompt
        self.params = self.get_params_from_string(prompt, workflow)

    def get_embed(self, msg_ctx: Message, attachment=None):
        return super().get_embed(
            msg_ctx,
            attachment=attachment,
            prioritize_fields=tuple(k for k, v in self.params.items() if "text" in v)
            if CFG.comfyui_prioritize_nodes_with_text
            else (),
        )

    def get_params_from_string(
        self,
        param_str: str,
        workflow_str: None | str,
    ) -> OrderedDict[str, str]:
        promptdata = json.loads(param_str)
        workflowdata = {}
        if workflow_str:
            with contextlib.suppress(Exception):
                workflowdata = json.loads(workflow_str)
        comfymeta = self.extract_comfy_metadata(promptdata, workflowdata)
        params = OrderedDict()
        nl = "\n"
        for k, v in comfymeta.items():
            vs = ((ik, str(iv)) for ik, iv in v.items())
            params[k] = "\n".join(
                f"[{ik}]:{f' {iv}' if len(iv) < 32 else f'{nl}{iv}{nl}'}"
                for ik, iv in vs
            ).strip()
        return params

    @staticmethod
    def set_comfy_input(result, name, key, inputs, typ=str) -> None:
        val = inputs.get(key)
        if val is None or not isinstance(val, typ):
            return
        if typ is str:
            if CFG.comfyui_replace_field_newlines:
                val = val.replace("\r", " ").replace("\n", " ")
            val = val.strip()
        vals = result.get(name)
        if vals is None:
            result[name] = {key: val}
        else:
            vals[key] = val

    @classmethod
    def set_widget_value(
        cls,
        workflowdata,
        inputs,
        node_id,
        input_name,
        required_type,
        widget_name,
    ):
        wf_node = workflowdata.get(node_id, {})
        widget_idx = -1
        for idx, wf_input in enumerate(wf_node.get("inputs", ())):
            if wf_input.get("name") != input_name:
                continue
            cur_widget_name = wf_input.get("widget", {}).get("name")
            if cur_widget_name != widget_name:
                continue
            widget_idx = idx
            break
        widget_values = wf_node.get("widgets_values", ())
        if widget_idx != -1 and widget_idx < len(widget_values):
            widget_value = None
            with contextlib.suppress(IndexError):
                widget_value = widget_values[widget_idx][0]
            if isinstance(widget_value, required_type):
                inputs[input_name] = widget_value

    @classmethod
    def extract_comfy_metadata(cls, promptdata, workflowdata, result=None):
        workflowdata = {str(v["id"]): v for v in workflowdata.get("nodes", ())}
        handlers = cls.COMFY_HANDLERS
        if result is None:
            result = OrderedDict()
        for k, v in promptdata.items():
            inputs = v.get("inputs").copy()
            typ = v.get("class_type", "").strip()
            handler = handlers.get(typ.lower())
            if not inputs or not handler:
                continue
            for input_name, required_type, *rest in handler:
                if rest and CFG.comfyui_extract_widget_values:
                    cls.set_widget_value(
                        workflowdata,
                        inputs,
                        k,
                        input_name,
                        required_type,
                        rest[0],
                    )
                if CFG.comfyui_show_titles:
                    title = v.get("_meta", {}).get("title", None)
                    if isinstance(title, str):
                        title = title.strip()
                    if title == typ:
                        title = None
                else:
                    title = None
                name = (
                    f"{typ}.{k} - {title.strip()}"
                    if isinstance(title, str)
                    else f"{typ}.{k}"
                )
                cls.set_comfy_input(result, name, input_name, inputs, required_type)
        return result


def populate_attachment_metadata(
    i: int,
    image_data: bytes,
    metadata: OrderedDict,
):
    with Image.open(io.BytesIO(image_data)) as img:
        if not img.info:
            return
        ii = img.info
        if "Steps:" in ii.get("parameters", ""):
            # Has Steps in paramaters field, looks like A1111 format
            metadata[i] = MetadataA1111(ii["parameters"])
        elif ii.get("prompt", "").lstrip().startswith('{"'):
            # (Apparent) JSON data in prompt field, looks like ComfyUI format
            metadata[i] = MetadataComfyUI(ii["prompt"], ii.get("workflow"))
        #
        # NovelAI NYI
        # elif img.info.get("Software") == "NovelAI" and "Description" in img.info:
        #     info = img.info["Description"] + img.info.get("Comment", "")


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
        errname = type(error).__name__
        log.exception(__f("Error: {errname}", errname=errname), exc_info=error)


async def collect_attachments(
    ctx: ApplicationContext,
    message: Message,
    respond=True,
):
    if respond:
        await ctx.defer(ephemeral=True)
    attachments = [
        a for a in message.attachments if a.filename.lower().endswith(".png")
    ]
    if not attachments:
        if respond:
            await ctx.respond("Sorry, this post has no images, or none of the images have prompts.", ephemeral=True)
        return None, None
    metadata = OrderedDict()
    tasks = [
        read_attachment_metadata(i, attachment, metadata)
        for i, attachment in enumerate(attachments)
    ]
    await asyncio.gather(*tasks)
    if not metadata:
        if respond:
            await ctx.respond(
                "Sorry, none of the images in this post have prompts.",
                ephemeral=True,
            )
        return None, None
    return metadata, attachments


async def update_reactions(message: Message, count: int):
    if count > 0:
        await message.add_reaction("🔎")
    elif CFG.react_on_no_metadata:
        await message.add_reaction("⛔")


@client.event
async def on_ready():
    log.info(__f("Logged in as {user}!", user=client.user))


@client.event
async def on_message(message: Message):
    if (
        not (message.channel.id in CFG.monitored_channel_ids and message.attachments)
        or message.author.bot
    ):
        return
    attachments = [
        a
        for a in message.attachments
        if a.filename.lower().endswith(".png") and a.size < CFG.scan_limit_bytes
    ]
    if not attachments:
        return
    log.info(__f("MESSAGE: {0!r}", message))
    count = 0
    for i, attachment in enumerate(
        attachments,
    ):  # download one at a time as usually the first image is already ai-generated
        metadata = OrderedDict()
        await read_attachment_metadata(i, attachment, metadata)
        if metadata:
            count += 1
            break
    await update_reactions(message, count)


@client.event
async def on_raw_reaction_add(ctx: RawReactionActionEvent):
    """Send image metadata in reacted post to user DMs"""
    if (
        ctx.emoji.name != "🔎"
        or ctx.channel_id not in CFG.monitored_channel_ids
        or ctx.member.bot
    ):
        return
    channel = client.get_channel(ctx.channel_id)
    message = await channel.fetch_message(ctx.message_id)
    if not message:
        return
    log.info(__f("REACTION: {0!r}", ctx))
    metadata, attachments = await collect_attachments(ctx, message, respond=False)
    count = 0
    if metadata:
        user_dm = await client.get_user(ctx.user_id).create_dm()
        try:
            for attachment, md in ((attachments[i], data) for i, data in metadata.items()):
                embed, view = md.get_embed_view(message, attachment)
                await user_dm.send(embed=embed, view=view, mention_author=False)
                count += 1
        except Exception as error:
            errname = type(error).__name__
            log.exception(__f("Error: {errname}", errname=errname), exc_info=error)
            await user_dm.send(content="Something went wrong when sending the prompt, sorry! The prompt for the image you have requested may be too long to send.", mention_author=False)
             
    await update_reactions(message, count)


@client.message_command(name="View Prompt")
async def message_command_view_prompt(ctx: ApplicationContext, message: Message):
    """Get raw list of parameters for every image in this post."""
    log.info(
        __f("APP: View: ctx={ctx!r}, message={message!r}", ctx=ctx, message=message),
    )
    metadata, attachments = await collect_attachments(ctx, message)
    if not metadata:
        return
    extraargs = {}
    for idx, (attachment, md) in enumerate(
        (attachments[i], data) for i, data in metadata.items()
    ):
        embed, view = md.get_embed_view(message, attachment, ephemeral=True)
        try:
            await ctx.respond(embed=embed, view=view, **extraargs)
            if idx == 0:
                extraargs["ephemeral"] = True
        except Exception as error:
            errname = type(error).__name__
            log.exception(__f("Error: {errname}", errname=errname), exc_info=error)
            await ctx.respond(
                "Couldn't DM. The prompt may be too long. Also, please check that your DMs from non-friends are enabled for this server.",
                ephemeral=True,
                delete_after=60,
            )
            return


@client.message_command(name="View Prompt (Get a DM)")
async def message_command_view_prompt_dm(ctx: ApplicationContext, message: Message):
    """Get raw list of parameters for every image in this post."""
    log.info(
        __f("APP: ViewDM: ctx={ctx!r}, message={message!r}", ctx=ctx, message=message),
    )
    metadata, attachments = await collect_attachments(ctx, message)
    if not metadata:
        return
    user_dm = await client.get_user(ctx.author.id).create_dm()
    for attachment, md in ((attachments[i], data) for i, data in metadata.items()):
        embed, view = md.get_embed_view(message, attachment)
        try:
            await user_dm.send(embed=embed, view=view, mention_author=False)
        except Exception as error:
            errname = type(error).__name__
            log.exception(__f("Error: {errname}", errname=errname), exc_info=error)
            await ctx.respond(
                "Couldn't DM. The prompt may be too long. Also, please check that your DMs from non-friends are enabled for this server.",
                ephemeral=True,
                delete_after=60,
            )
            return
    await ctx.respond("DM sent!", ephemeral=True, delete_after=60)


def handle_check(filename: Path):
    with filename.open("rb") as fp:
        file_data = fp.read()
    metadata = OrderedDict()
    populate_attachment_metadata(0, file_data, metadata)
    if not metadata:
        print("* No metadata")
        return
    md = metadata[0]
    print(f"* Dumping {md.NAME} parameters")
    for k, v in md.params.items():
        print(f"\n{k.strip()}:\n{v.strip()}")


class ColorLogFormatter(logging.Formatter):
    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {  # noqa: RUF012
        logging.DEBUG: GREY,
        logging.INFO: GREY,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def __init__(
        self,
        use_color: bool,
        fmt="{asctime} {levelname:>8}: {message}",
        datefmt="%Y%m%d.%H%M%S",
    ):
        super().__init__(
            fmt=fmt,
            datefmt="%Y%m%d.%H%M%S",
            style="{",
        )
        if use_color:
            self.formatters = {
                ll: ColorExceptionLogFormatter(
                    fmt=f"{self.LEVEL_COLORS[ll]}{fmt}{self.RESET}",
                    datefmt=datefmt,
                    style="{",
                )
                for ll in (
                    logging.DEBUG,
                    logging.INFO,
                    logging.WARNING,
                    logging.ERROR,
                    logging.CRITICAL,
                )
            }
        self.use_color = use_color

    def format(self, record):
        if not self.use_color:
            return super().format(record)
        return self.formatters[record.levelno].format(record)


class ColorExceptionLogFormatter(logging.Formatter):
    def formatException(self, exc_info):
        result = super().formatException(exc_info)
        return f"{ColorLogFormatter.BOLD_RED}{result}{ColorLogFormatter.RESET}"

    def formatStack(self, stack_info):
        return f"{ColorLogFormatter.BOLD_RED}{stack_info}{ColorLogFormatter.RESET}"


def setup_logging():
    global log  # noqa: PLW0603
    log_level = getattr(
        logging,
        CFG.log_level.upper(),
        None,
    )
    if log_level is None:
        raise ValueError("Invalid log_level in configuration")
    log = logging.getLogger("PromptInspector")
    log.setLevel(log_level)
    ch = logging.StreamHandler()  # Simple console logging
    ch.setLevel(log_level)
    ch.setFormatter(ColorLogFormatter(CFG.log_color))
    log.addHandler(ch)


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
    args = parser.parse_args()
    CFG.load(args.config)
    if args.dump:
        handle_check(args.dump)
        return
    # Otherwise run the bot
    setup_logging()
    if not CFG.monitored_channel_ids:
        log.error("No channels to monitor!")
        sys.exit(1)
    bot_token = os.environ.get("BOT_TOKEN")
    if bot_token is None:
        log.error("BOT_TOKEN environment variable missing!")
        sys.exit(1)
    client.run(bot_token)


if __name__ == "__main__":
    main()
