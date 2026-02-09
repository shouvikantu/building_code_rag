"""
app.py

Flask web UI for Portland Zoning Query Tool.
Allows querying single addresses or multiple properties by ZIP code.
"""

from flask import Flask, render_template, request
import logging
import json
from dotenv import load_dotenv
from light_rag_impl import start_background_init

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

start_background_init()


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

                    from light_rag_impl import query_property
                    rag_answer = query_property(property_question, selected_property)
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
                    from zoning import query_property_by_address
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
                    from zoning import query_properties_by_zip
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

if __name__ == "__main__":
    # Default to listening on all interfaces to avoid host header issues locally.
    # Keep debug on for local development, but bind to 0.0.0.0 so requests from
    # different hostnames (e.g., localhost vs 127.0.0.1) don't get blocked.
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
