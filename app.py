from flask import Flask, request, jsonify
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# model_name = "nlpconnect/vit-gpt2-image-captioning"
model_name = "Salesforce/blip-image-captioning-base"

@app.route("/health-check", methods=["GET"])
def healthCheck():
    return jsonify({"messsage": "server is running"})

# Derfine the API endpoint
@app.route("/generate-caption", methods=["POST"])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Load and preprocess the image
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Generate the caption
    output_ids = model.generate(pixel_values, max_length=50, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Return the caption
    return jsonify({"caption": caption})

# Run the Flask app
if __name__ == '__main__':
    # Load the pre-trained ViT-GPT2 model
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    app.run(port=5000, debug=True)
