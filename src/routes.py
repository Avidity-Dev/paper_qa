import hashlib
import mimetypes
import os
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.adapters.cloudobjectstore import CloudStorageFactory

app = Flask(__name__)
CORS(app)

storage_config = {
    "provider": "azure",
    "config": {},
}

storage_adapter = CloudStorageFactory.create_storage_adapter(
    storage_config["provider"], storage_config["config"]
)


@app.route("/api/v0.1/upload", methods=["POST"])
def upload_files():
    """Handle file uploads and store them in cloud storage."""
    try:
        if "file0" not in request.files:
            return jsonify({"error": "No files provided"}), 400

        uploaded_files = []
        file_hashes = set()

        # Process each uploaded file
        for key in request.files:
            file = request.files[key]

            if file.filename == "":
                continue

            if file:
                # Secure the filename
                filename = secure_filename(file.filename)

                # Read file data and calculate hash
                file_data = file.read()
                file_hash = hashlib.sha256(file_data).hexdigest()

                # Check for duplicates
                if file_hash in file_hashes:
                    continue

                file_hashes.add(file_hash)

                # Create a new filename using the hash
                ext = os.path.splitext(filename)[1]
                cloud_filename = f"{file_hash}{ext}"

                # Get content type
                content_type = mimetypes.guess_type(filename)[0]

                # Upload to cloud storage
                file_obj = BytesIO(file_data)
                upload_result = storage_adapter.upload_file(
                    file_obj, cloud_filename, content_type
                )

                uploaded_files.append(
                    {
                        "original_filename": filename,
                        "cloud_filename": cloud_filename,
                        "hash": file_hash,
                        "storage_details": upload_result,
                    }
                )

        if not uploaded_files:
            return jsonify({"error": "No valid files were uploaded"}), 400

        return (
            jsonify(
                {
                    "message": "Files uploaded successfully",
                    "files": uploaded_files,
                    "documentIds": [f["hash"] for f in uploaded_files],
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": "Internal server error during upload"}), 500
