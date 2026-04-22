"""
app.py

Flask web UI for Portland Zoning Query Tool.
Allows querying single addresses or multiple properties by ZIP code.
"""

from flask import Flask, render_template, request, jsonify
import logging
import json
import os
import requests
from dotenv import load_dotenv
# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from light_rag_impl import query_rag, ingest_docs
from zoning import query_property_by_address, query_properties_by_zip

# Ingest docs at startup (or you could trigger it via a route)
# try:
#     ingest_docs()
# except Exception as e:
#     logger.error(f"Failed to ingest docs at startup: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    properties = None
    error = None
    address = ""
    zip_code = ""
    num = 10
    base_zones = []
    min_sqft = ""
    query_type = "address"
    rag_answer = None
    property_question = ""
    selected_property_index = 0

    if request.method == "POST":
        action = request.form.get("action", "search")
        if action == "rag":
            property_question = request.form.get("property_question", "").strip()
            selected_property_index = int(request.form.get("property_index", "0") or 0)
            properties_json = request.form.get("properties_json", "").strip()
            query_type = request.form.get("query_type", "address")

            if not properties_json:
                error = "Missing property data. Please run a property search first."
            elif not property_question:
                error = "Please enter a question for the selected property."
            else:
                try:
                    properties = json.loads(properties_json)
                    if selected_property_index < 0 or selected_property_index >= len(properties):
                        raise IndexError("Invalid property selection.")
                    selected_property = properties[selected_property_index]
                    
                    # Contextualize the question with property data
                    context_q = (
                        f"Property Info: {json.dumps(selected_property)}.\n"
                        f"Question: {property_question}\n\n"
                        "INSTRUCTIONS:\n"
                        "- Provide a highly detailed and comprehensive answer.\n"
                        "- ALWAYS cite the exact laws, code sections, subsections, and provisions from the retrieved documents.\n"
                        "- Explicitly list out specific regulations, conditions, limitations (e.g., building area, operating hours), and specific operational requirements.\n"
                        "- Do not vaguely refer to 'zoning laws' or 'certain conditions'. Extract and provide the actual details and rules from the text."
                    )
                    rag_answer = query_rag(context_q)
                    
                except Exception as e:
                    error = f"Error querying RAG: {str(e)}"

            return render_template(
                "index.html",
                properties=properties,
                error=error,
                address=address,
                zip=zip_code,
                num=num,
                base_zones=base_zones,
                min_sqft=min_sqft,
                query_type=query_type,
                rag_answer=rag_answer,
                property_question=property_question,
                selected_property_index=selected_property_index,
            )

        query_type = request.form.get("query_type", "address")
        if query_type == "address":
            address = request.form.get("address", "").strip()
            if not address:
                error = "Please enter an address."
            else:
                try:
                    properties = [query_property_by_address(address)]
                except Exception as e:
                    error = f"Error querying address: {str(e)}"
        else:
            zip_code = request.form.get("zip", "").strip()
            try:
                num = int(request.form.get("num", 10))
            except ValueError:
                num = 10
            base_zones = request.form.getlist("base_zones")
            min_sqft_str = request.form.get("min_sqft", "").strip()
            try:
                min_sqft = int(min_sqft_str) if min_sqft_str else None
            except ValueError:
                min_sqft = None
            if not zip_code:
                error = "Please enter a ZIP code."
            else:
                try:
                    properties = query_properties_by_zip(
                        zip_code,
                        num,
                        base_zones=base_zones,
                        min_sqft=min_sqft,
                    )
                    if not properties:
                        error = "No properties found for this ZIP code (or none match the filters)."
                except Exception as e:
                    error = f"Error: {str(e)}"

    return render_template(
        "index.html",
        properties=properties,
        error=error,
        address=address,
        zip=zip_code,
        num=num,
        base_zones=base_zones,
        min_sqft=min_sqft,
        query_type=query_type,
        rag_answer=rag_answer,
        property_question=property_question,
        selected_property_index=selected_property_index,
    )

@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle generic RAG chat queries without zoning context."""
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400
    
    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400
        
    try:
        rag_answer = query_rag(question)
        return jsonify({"answer": rag_answer})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": "Failed to process query."}), 500

if __name__ == "__main__":
    # Default to listening on all interfaces to avoid host header issues locally.
    # Keep debug on for local development, but bind to 0.0.0.0 so requests from
    # different hostnames (e.g., localhost vs 127.0.0.1) don't get blocked.
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
