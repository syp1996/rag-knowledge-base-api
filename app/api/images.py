"""
Image upload API endpoints
"""
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()

# Allowed file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Upload directory
UPLOAD_DIR = Path("uploads/images")


def is_allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename to avoid conflicts"""
    import time
    file_ext = Path(original_filename).suffix.lower()
    unique_id = str(uuid.uuid4())
    timestamp = str(int(time.time() * 1000))  # Current timestamp
    return f"{timestamp}-{unique_id}{file_ext}"


@router.post("/upload")
async def upload_image(
    image: UploadFile = File(...),
    documentId: Optional[str] = Form(None)
):
    """
    Upload an image file
    
    Args:
        image: The image file to upload
        documentId: Optional document ID for association
    
    Returns:
        JSON response with the image URL
    """
    try:
        # Validate file
        if not image.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        if not is_allowed_file(image.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read file content
        content = await image.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Ensure upload directory exists
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        unique_filename = generate_unique_filename(image.filename)
        file_path = UPLOAD_DIR / unique_filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Generate URL (assuming the uploads directory will be served statically)
        # You may need to adjust this based on your server configuration
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        image_url = f"{base_url}/uploads/images/{unique_filename}"
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "url": image_url
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/health")
async def images_health():
    """Health check for images service"""
    upload_dir_exists = UPLOAD_DIR.exists()
    upload_dir_writable = os.access(UPLOAD_DIR, os.W_OK) if upload_dir_exists else False
    
    return {
        "status": "healthy" if upload_dir_exists and upload_dir_writable else "unhealthy",
        "upload_directory": str(UPLOAD_DIR),
        "directory_exists": upload_dir_exists,
        "directory_writable": upload_dir_writable
    }