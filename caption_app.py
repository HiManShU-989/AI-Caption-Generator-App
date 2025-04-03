from PIL import Image
import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaption:
    def _init_(self):
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate(self, img):
        if isinstance(img, str):
            img = Image.open(img)  # Open image if path is provided

        img = img.convert("RGB")  # Ensure the image is in RGB format

        # Process the image correctly
        inputs = self.processor(images=img, return_tensors="pt")  
        pixel_values = inputs["pixel_values"]  # Extract pixel values
        output = self.model.generate(pixel_values=pixel_values)  # Generate caption

        caption = self.processor.decode(output[0], skip_special_tokens=True)  # Decode output
        return caption

# Custom CSS to enhance appearance
custom_css = """
body {
    background: linear-gradient(to right, #1f4037, #99f2c8); /* Green Gradient Background */
    font-family: 'Arial', sans-serif;
    color: white;
    text-align: center;
}
#title {
    font-size: 2.5em;
    font-weight: bold;
    padding: 20px;
}
button {
    background-color: #ffcc00 !important; /* Yellow Button */
    color: black !important;
    font-size: 16px;
    font-weight: bold;
    border-radius: 10px;
}
"""

# Create an instance of the class
ic = ImageCaption()

# Improved UI with Gradio Blocks
with gr.Blocks(css=custom_css) as app:
    gr.Markdown("# 🖼 Image Caption Generator**", elem_id="title")
    gr.Markdown(
        "📌 *Upload an image and get an AI-generated caption.*\n"
       
    )

    with gr.Row():
        img = gr.Image(type="pil", label="Upload Your Image Here")
        output_text = gr.Textbox(label="Generated Caption", interactive=False)

    generate_btn = gr.Button("🔍 Generate Caption")
    
    generate_btn.click(ic.generate, inputs=img, outputs=output_text)

# Launch the improved UI
app.launch(share=True)
