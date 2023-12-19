import os
import time
import torch
import random
import gradio as gr

from sys import platform
from PIL import Image
from contextlib import nullcontext
from diffusers.utils import load_image
from diffusers import AutoPipelineForImage2Image, LCMScheduler
from modules import script_callbacks

cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
is_mac = platform == "darwin"


def should_use_fp16():
    if is_mac:
        return True

    gpu_props = torch.cuda.get_device_properties("cuda")
    if gpu_props.major < 6:
        return False

    nvidia_16_series = ["1660", "1650", "1630"]
    for x in nvidia_16_series:
        if x in gpu_props.name:
            return False

    return True


class timer:
    def __init__(self, method_name="timed process"):
        self.method = method_name

    def __enter__(self):
        self.start = time.time()
        print(f"{self.method} starts")

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        print(f"{self.method} took {str(round(end - self.start, 2))}s")


def load_models(model_id="Lykon/dreamshaper-7"):
    if not is_mac:
        torch.backends.cuda.matmul.allow_tf32 = True

    use_fp16 = should_use_fp16()
    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

    if use_fp16:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id, cache_dir=cache_path, torch_dtype=torch.float16, variant="fp16", safety_checker=None)
    else:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id, cache_dir=cache_path, safety_checker=None)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm_lora_id)
    pipe.fuse_lora()

    device = "mps" if is_mac else "cuda"
    pipe.to(device=device)
    generator = torch.Generator()

    def inference(prompt,
                  img,
                  steps=4,
                  cfg=1,
                  sketch_strength=0.9,
                  seed=random.randrange(0, 2 ** 63)):
        with torch.inference_mode():
            with torch.autocast("cuda") if device == "cuda" else nullcontext():
                with timer("inference"):
                    return pipe(
                        prompt=prompt,
                        image=load_image(img),
                        generator=generator.manual_seed(seed),
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        strength=sketch_strength
                    ).images[0]

    return inference


def on_ui_tabs():
    canvas_size = 512
    inference = load_models()

    with gr.Blocks(analytics_enabled=False) as realtime_lcm_canvas:
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    steps = gr.Slider(label="steps", minimum=4, maximum=8, step=1, value=4, interactive=True)
                    cfg = gr.Slider(label="cfg", minimum=0.1, maximum=3, step=0.1, value=1, interactive=True)
                    sketch_strength = gr.Slider(label="sketch strength", minimum=0.1, maximum=0.9, step=0.1, value=0.9,
                                                interactive=True)
                with gr.Column():
                    model_id = gr.Text(
                        label="Model Hugging Face id (after changing this wait until the model downloads in the console)",
                        value="Lykon/dreamshaper-7", interactive=True)
                    prompt = gr.Text(label="Prompt",
                                     value="Scary warewolf, 8K, realistic, colorful, long sharp teeth, splash art",
                                     interactive=True)
                    seed = gr.Number(label="seed", value=1337, interactive=True)

            with gr.Row(equal_height=True):
                img = gr.Image(source="canvas", tool="color-sketch", shape=(canvas_size, canvas_size),
                               width=canvas_size,
                               height=canvas_size, type="pil")
                output = gr.Image(width=canvas_size, height=canvas_size)

                def process_image(prompt, img, steps, cfg, sketch_strength, seed):
                    if not img:
                        return Image.new("RGB", (canvas_size, canvas_size))
                    return inference(prompt, img, steps, cfg, sketch_strength, seed=int(seed))

                reactive_controls = [prompt, img, steps, cfg, sketch_strength, seed]
                for control in reactive_controls:
                    control.change(fn=process_image, inputs=reactive_controls, outputs=output)

                def update_model(model_name):
                    global inference
                    inference = load_models(model_name)

                model_id.change(fn=update_model, inputs=model_id)

    return [(realtime_lcm_canvas, "Realtime_LCM_Canvas", "Realtime_LCM_Canvas")]


script_callbacks.on_ui_tabs(on_ui_tabs)
