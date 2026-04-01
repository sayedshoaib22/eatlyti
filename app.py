from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# API configuration
SITE_URL = os.getenv("SITE_URL", "http://localhost:5000")
SITE_NAME = os.getenv("SITE_NAME", "Eatlytic")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://api.openrouter.ai/api/v1/chat/completions"

# Removed test code and OpenRouter config

# Language mapping
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean"
}


@app.route("/api/analyze", methods=["POST"])
def analyze_food():
    # Validate API key upfront
    api_key = OPENROUTER_API_KEY
    if not api_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in environment")
        return jsonify({
            "success": False,
            "error": "API key not configured on server"
        }), 400

    try:
        # Parse request JSON
        data = request.get_json()
        if not data:
            print("❌ ERROR: No JSON data provided")
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # Extract and validate image data
        image_data = data.get("imageDataUrl", "").strip()
        if not image_data:
            print("❌ ERROR: No image data URL provided")
            return jsonify({
                "success": False,
                "error": "Image not provided"
            }), 400
        
        # Validate image data format
        if not image_data.startswith("data:image/"):
            print(f"❌ ERROR: Invalid image format. Got: {image_data[:50]}...")
            return jsonify({
                "success": False,
                "error": "Invalid image format. Must be a base64 data URL (data:image/...)"
            }), 400

        # Get language setting
        language_code = data.get("language", "en").strip()
        if not language_code or language_code not in LANGUAGE_NAMES:
            language_code = "en"
        
        language_name = LANGUAGE_NAMES.get(language_code, "English")
        print(f"📝 Analyzing food image in {language_name}")

        # Prepare OpenRouter API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME,
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are Eatlytic AI, a nutrition expert.

Analyze the food image and provide:

1. **Estimated Calories**: Total calories
2. **Macronutrients**: Protein (g), Fat (g), Carbs (g)
3. **Fiber**: Dietary fiber if applicable
4. **Heart Health Impact**: Positive/Negative factors
5. **Muscle Impact**: Protein quality and amino acids
6. **Energy Impact**: Quick vs sustained energy
7. **Smart Eating Tips**: 2-3 actionable suggestions

Format: Use **bold** for headers and bullet points for details.
Respond in {language_name} (or English if translation not available)."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this food item in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        # Make API request to OpenRouter
        print(f"🔗 Calling OpenRouter API...")
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        print(f"📊 OpenRouter Response Status: {response.status_code}")

        # Handle authentication errors
        if response.status_code == 401:
            print("❌ ERROR: Invalid or expired OpenRouter API key")
            return jsonify({
                "success": False,
                "error": "Authentication failed - Invalid API key"
            }), 401

        # Handle other API errors
        if response.status_code == 429:
            print("❌ ERROR: Rate limited by OpenRouter")
            return jsonify({
                "success": False,
                "error": "Too many requests - Please try again later"
            }), 429

        if not response.ok:
            error_response = response.text
            print(f"❌ OpenRouter API Error ({response.status_code}): {error_response}")
            
            try:
                error_json = response.json()
                error_message = error_json.get("error", {}).get("message", error_response)
            except:
                error_message = error_response

            return jsonify({
                "success": False,
                "error": "API processing failed",
                "details": error_message,
                "status": response.status_code
            }), 502

        # Parse successful response
        try:
            result = response.json()
        except Exception as e:
            print(f"❌ ERROR: Failed to parse OpenRouter response: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Invalid response format from API"
            }), 502

        # Validate response structure
        if "choices" not in result or len(result["choices"]) == 0:
            print(f"❌ ERROR: Unexpected API response structure: {result}")
            return jsonify({
                "success": False,
                "error": "API returned empty response"
            }), 502

        # Extract analysis
        try:
            analysis = result["choices"][0]["message"]["content"]
            if not analysis or not analysis.strip():
                return jsonify({
                    "success": False,
                    "error": "API returned empty analysis"
                }), 502
        except (KeyError, IndexError, TypeError) as e:
            print(f"❌ ERROR: Could not extract content from response: {str(e)}")
            return jsonify({
                "success": False,
                "error": "Malformed API response"
            }), 502

        print("✅ Analysis completed successfully")
        return jsonify({
            "success": True,
            "analysis": analysis
        }), 200

    except requests.exceptions.Timeout:
        print("❌ ERROR: OpenRouter API request timed out")
        return jsonify({
            "success": False,
            "error": "Request timeout - API took too long to respond"
        }), 504

    except requests.exceptions.ConnectionError as e:
        print(f"❌ ERROR: Connection failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to connect to API service"
        }), 503

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ UNEXPECTED ERROR: {error_trace}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e),
            "type": type(e).__name__
        }), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "Eatlytic AI backend running"
    })


@app.route("/")
def home():
    return jsonify({
        "name": "Eatlytic API",
        "version": "2.0",
        "status": "running"
    })


if __name__ == "__main__":

    if OPENROUTER_API_KEY:
        print("✅ API key loaded")
    else:
        print("⚠️ WARNING: api_key missing")

    print("🚀 Server running on http://localhost:5000")

    app.run(host="0.0.0.0", port=5000, debug=True)