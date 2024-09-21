import os
import base64
from flask import Flask, request, render_template
import openai
import pandas as pd
from dotenv import load_dotenv
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables and set the API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
logging.debug(f"OpenAI API Key Loaded: {openai.api_key is not None}")

app = Flask(__name__)

def analyze_image(image_path):
    logging.debug(f"Analyzing image at path: {image_path}")
    # Encode the image to base64
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    img_type = 'image/jpeg'  # Adjust based on your image type
    base64_image = base64.b64encode(image_data).decode('utf-8')
    data_url = f"data:{img_type};base64,{base64_image}"
    logging.debug("Image encoded to base64")

    # Create a more detailed prompt
    prompt = """
Please analyze the following image and perform the tasks below:

1. Extract all the text content present in the image.
2. Identify any brand names or logos present in the image.

Provide your response in the following JSON format:

{
  "extracted_text": "Extracted text here",
  "brand": "Identified brand here (if any)"
}
"""
    logging.debug(f"Prompt: {prompt}")

    # Create the messages payload using the vision API method
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Use the GPT-4 Vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        logging.debug("OpenAI API call successful")
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API Error: {e}")
        return None

    # Extract the response
    result = response.choices[0].message.content
    logging.debug(f"OpenAI response: {result}")

    return result

def parse_extracted_info(result):
    logging.debug("Parsing extracted information")
    extracted_text = ""
    brand_found = "Unknown"

    if result is None:
        logging.error("No result to parse")
        return extracted_text, brand_found

    # Try to parse the response as JSON
    try:
        # Find the JSON object in the response
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = result[json_start:json_end]
            data = json.loads(json_str)
            extracted_text = data.get('extracted_text', '').strip()
            brand_found = data.get('brand', 'Unknown').strip()
            logging.debug(f"Extracted Text: {extracted_text}")
            logging.debug(f"Brand Found: {brand_found}")
        else:
            logging.error("JSON format not found in response")
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        logging.debug("Attempting to parse response as plain text")
        # Fallback to plain text parsing
        extracted_text = result.strip()
        logging.debug(f"Extracted Text: {extracted_text}")

    return extracted_text, brand_found

def evaluate_text_and_brand(extracted_text, brand_found):
    evaluation_prompt = f"""
Please evaluate the following text extracted from an image for the brand "{brand_found}" based on the following criteria:

1. **Intent & Purpose**
   - What is the message conveying?
   - What is the intent of the messenger?

2. **Propriety & Accuracy**
   - Is the message adhering to Ethical Standards?
   - Is it fact-based?

3. **Competence & Transparency**
   - What is the level of source clarity?
   - Is there a level of bias acknowledgement?

4. **Reliability**
   - Track record of the messenger?
   - What is the consistency of the messenger over time?

Text:
\"\"\"
{extracted_text}
\"\"\"

Provide the evaluation in a pure JSON format like this without ANY other aspects:

[
  {{
    "Criteria": "Intent & Purpose",
    "Evaluation": {{
      "Message Conveyed": "Your evaluation here.",
      "Intent of Messenger": "Your evaluation here."
    }}
  }},
  {{
    "Criteria": "Propriety & Accuracy",
    "Evaluation": {{
      "Adherence to Ethical Standards": "Your evaluation here.",
      "Fact-Based": "Your evaluation here."
    }}
  }},
  {{
    "Criteria": "Competence & Transparency",
    "Evaluation": {{
      "Source Clarity": "Your evaluation here.",
      "Bias Acknowledgement": "Your evaluation here."
    }}
  }},
  {{
    "Criteria": "Reliability",
    "Evaluation": {{
      "Messenger's Track Record": "Your evaluation here.",
      "Consistency Over Time": "Your evaluation here."
    }}
  }}
]
"""

    logging.debug(f"Evaluation Prompt: {evaluation_prompt}")

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": evaluation_prompt}
            ],
        )
        logging.debug("OpenAI API call for evaluation successful")
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API Error during evaluation: {e}")
        return None

    evaluation = response.choices[0].message.content
    logging.debug(f"Evaluation Response: {evaluation}")
    return evaluation

def create_evaluation_table(evaluation):
    logging.debug("Creating evaluation table")
    try:
        evaluation_data = json.loads(evaluation)
        df = pd.DataFrame(evaluation_data)
        logging.debug("Evaluation data parsed into DataFrame")
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        # Handle parsing errors
        df = pd.DataFrame(columns=["Criteria", "Evaluation"])
    return df

def generate_summary(evaluation, brand_found):
    logging.debug("Generating summary")
    summary_prompt = f"""
Based on the following evaluations of the text for the brand "{brand_found}", provide a concise summary overview and give a total score between 0 and 100 where 100 is true, 0 is a lie.

Evaluations:
{evaluation}

Summary:
"""
    logging.debug(f"Summary Prompt: {summary_prompt}")

    try:
        summary_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": summary_prompt}
            ],
        )
        logging.debug("OpenAI API call for summary successful")
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API Error during summary generation: {e}")
        return "Summary generation failed."

    summary = summary_response.choices[0].message.content
    logging.debug(f"Summary Response: {summary}")
    return summary

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        logging.debug("Image upload request received")
        # Ensure 'static' directory exists
        if not os.path.exists('static'):
            os.makedirs('static')
            logging.debug("Created 'static' directory")

        # Save the uploaded image
        image = request.files['image']
        image_path = os.path.join('static', image.filename)
        image.save(image_path)
        logging.debug(f"Image saved at {image_path}")

        # Analyze the image
        result = analyze_image(image_path)

        # Parse the extracted information
        extracted_text, brand_found = parse_extracted_info(result)

        # Evaluate the text and brand
        evaluation = evaluate_text_and_brand(extracted_text, brand_found)

        # Create evaluation table
        df = create_evaluation_table(evaluation)

        # Generate summary
        summary = generate_summary(evaluation, brand_found)

        return render_template('results.html', tables=[df.to_html(classes='data')], summary=summary)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
