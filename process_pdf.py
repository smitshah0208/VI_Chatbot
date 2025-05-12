import os
import uuid
import pandas as pd
from unstructured.partition.pdf import partition_pdf
from PIL import Image
import io
import base64
import fitz  # PyMuPDF - ENSURE THIS IS WORKING (NO module 'fitz' has no attribute 'open' ERROR)
import shutil
import traceback

from dotenv import load_dotenv
load_dotenv()
TF_ENABLE_ONEDNN_OPTS= os.getenv("TF_ENABLE_ONEDNN_OPTS")

def handle_image_element(element, index, pdf_doc, image_output_path):
    """
    Handle image extraction with multiple fallback methods.
    Args:
        element: The image element from unstructured.
        index: Index of the element for naming.
        pdf_doc: Open PyMuPDF document, or None. (WILL BE NONE if fitz.open fails)
        image_output_path: Directory to save extracted images.
    Returns:
        Markdown string for the image.
    """
    debug_index = 1 # Keep or remove this debug section as needed
    if index == debug_index:
        print(f"\n--- DEBUGGING Image {index} ---")
        print(f"Element type: {type(element)}")
        print(f"Element text (if any): {getattr(element, 'text', 'N/A')}")
        if hasattr(element, 'image') and element.image is not None:
            print(f"element.image type: {type(element.image)}")
            try:
                print(f"element.image format: {element.image.format}")
                print(f"element.image size: {element.image.size}")
                print(f"element.image mode: {element.image.mode}")
            except Exception as e_pil:
                print(f"element.image - PIL details error: {e_pil}")
        else:
            print(f"element.image: Not present or None")
        if hasattr(element, 'metadata'):
            print("Element Metadata:")
            try:
                metadata_dict = element.metadata.to_dict() if hasattr(element.metadata, 'to_dict') else vars(element.metadata)
                for k, v in metadata_dict.items():
                    print(f"  metadata.{k}: {v}")
            except Exception as e_meta:
                print(f"  Error printing metadata: {e_meta}")
                try:
                    metadata_vars = vars(element.metadata)
                    for k,v in metadata_vars.items():
                         print(f"  metadata.{k} (vars): {v}")
                except Exception as e_vars:
                    print(f"  Could not print metadata using vars(): {e_vars}")
        else:
            print("Element Metadata: Not present")
        print(f"--- END DEBUGGING Image {index} ---\n")

    img_filename_base = f"image_{index}_{uuid.uuid4().hex[:8]}"
    img_filename = f"{img_filename_base}.png"
    img_path = os.path.join(image_output_path, img_filename)
    md_img_path = f"images/{img_filename}"

    # Method 0: Base64
    if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64') and element.metadata.image_base64:
        if hasattr(element.metadata, 'image_mime_type') and element.metadata.image_mime_type:
            ext_parts = element.metadata.image_mime_type.split('/')
            if len(ext_parts) > 1:
                ext = ext_parts[-1]
                if ext in ['jpeg', 'png', 'gif', 'bmp', 'tiff', 'jpg']:
                    img_filename = f"{img_filename_base}.{ext}"
                    img_path = os.path.join(image_output_path, img_filename)
                    md_img_path = f"images/{img_filename}"
        try:
            img_data = base64.b64decode(element.metadata.image_base64)
            image = Image.open(io.BytesIO(img_data))
            image.save(img_path)
            print(f"Method 0 (metadata.image_base64) successful for Image {index}")
            return f"\n![Image {index} via Base64]({md_img_path})\n"
        except Exception as e0:
            if index == debug_index: print(f"DEBUG Image {index}: Method 0 (metadata.image_base64) failed: {e0}")

    # Method 1: element.image
    if hasattr(element, 'image') and element.image is not None:
        try:
            element.image.save(img_path)
            print(f"Method 1 (element.image) successful for Image {index}")
            return f"\n![Image {index} Direct]({md_img_path})\n"
        except Exception as e1:
            if index == debug_index: print(f"DEBUG Image {index}: Method 1 (element.image) failed: {e1}")


    # Method 2: Metadata image_path (Corrected Logic)
    image_path_from_metadata = None
    if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_path') and element.metadata.image_path:
        # unstructured (with extract_image_block_output_dir set) should provide a usable path directly.
        # This path is expected to be relative to the project root or an absolute path.
        image_path_from_metadata = element.metadata.image_path
        if index == debug_index: print(f"DEBUG Image {index}: Method 2 attempting with metadata.image_path: '{image_path_from_metadata}'")


    if image_path_from_metadata:
        try:
            if os.path.isfile(image_path_from_metadata):
                _, ext = os.path.splitext(image_path_from_metadata)
                img_filename_ext = img_filename # default to .png if no ext
                md_img_path_ext = md_img_path  # default
                if ext and ext.lower() != '.': # Ensure ext is not empty or just "."
                    img_filename_ext = f"{img_filename_base}{ext}"
                    md_img_path_ext = f"images/{img_filename_ext}"

                final_img_save_path = os.path.join(image_output_path, os.path.basename(img_filename_ext))

                shutil.copy(image_path_from_metadata, final_img_save_path)
                print(f"Method 2 (metadata.image_path) successful for Image {index} from '{image_path_from_metadata}' to '{final_img_save_path}'")
                return f"\n![Image {index} Metadata Path]({os.path.join('images', os.path.basename(img_filename_ext))})\n" # Use basename for markdown
            else:
                if index == debug_index: print(f"DEBUG Image {index}: Method 2 (metadata.image_path): File not found at '{image_path_from_metadata}'")
        except Exception as e2:
            if index == debug_index: print(f"DEBUG Image {index}: Method 2 (metadata.image_path) failed for '{image_path_from_metadata}': {e2}\n{traceback.format_exc()}")

    # PyMuPDF Fallbacks (Methods 3 and 4)
    # These will only work if pdf_doc was successfully opened (i.e., fitz.open worked)
    if pdf_doc:
        page_number = None
        coordinates = None
        if hasattr(element, 'metadata'):
            if hasattr(element.metadata, 'page_number'):
                page_number = element.metadata.page_number
            if hasattr(element.metadata, 'coordinates') and element.metadata.coordinates:
                coords_data = element.metadata.coordinates
                if isinstance(coords_data, dict) and 'points' in coords_data and isinstance(coords_data['points'], tuple):
                     # Handling {'points': ((x1,y1), (x2,y2), ...), 'system': 'PixelSpace', ...}
                    coords_tuple = coords_data['points']
                    if len(coords_tuple) > 1 and all(isinstance(pt, tuple) and len(pt) == 2 for pt in coords_tuple):
                        all_x = [pt[0] for pt in coords_tuple]
                        all_y = [pt[1] for pt in coords_tuple]
                        # Ensure coordinates are sensible numbers before min/max
                        all_x_valid = [x for x in all_x if isinstance(x, (int, float))]
                        all_y_valid = [y for y in all_y if isinstance(y, (int, float))]
                        if all_x_valid and all_y_valid:
                            coordinates = (min(all_x_valid), min(all_y_valid), max(all_x_valid), max(all_y_valid))
                        else:
                            if index == debug_index: print(f"DEBUG Image {index}: Invalid coordinate numbers in points: {coords_tuple}")
                elif isinstance(coords_data, tuple) and len(coords_data) == 4 and \
                    all(isinstance(c, (int, float)) for c in coords_data):
                    coordinates = coords_data


        # Method 3: PyMuPDF with coordinates
        if page_number is not None and coordinates:
            try:
                page_index = int(page_number) - 1
                if 0 <= page_index < len(pdf_doc):
                    page = pdf_doc[page_index]
                    x0, y0, x1, y1 = coordinates

                    if x1 > x0 and y1 > y0:
                        rect = fitz.Rect(x0, y0, x1, y1)
                        if index == debug_index: print(f"DEBUG Image {index}: Method 3 attempting with rect: {rect} on page {page_number}")

                        pix = page.get_pixmap(clip=rect, dpi=150)
                        if pix.width > 0 and pix.height > 0:
                            pix.save(img_path)
                            print(f"Method 3 (PyMuPDF get_pixmap with coords) successful for Image {index}")
                            return f"\n![Image {index} Coords Fallback]({md_img_path})\n"
                        else:
                            if index == debug_index: print(f"DEBUG Image {index}: Method 3: get_pixmap resulted in empty image for rect {rect}.")
                    else:
                        if index == debug_index: print(f"DEBUG Image {index}: Method 3: Invalid coordinates {coordinates}")
            except Exception as e3:
                if index == debug_index: print(f"DEBUG Image {index}: Method 3 (PyMuPDF get_pixmap with coords) failed: {e3}\n{traceback.format_exc()}")


        # Method 4: PyMuPDF page.get_images()
        if page_number is not None:
            try:
                page_index = int(page_number) - 1
                if 0 <= page_index < len(pdf_doc):
                    page = pdf_doc[page_index]
                    image_list = page.get_images(full=True)

                    if not image_list:
                        if index == debug_index: print(f"DEBUG Image {index}: Method 4: No images found on page {page_number} via page.get_images().")
                    else:
                        extracted_img_info = None
                        if coordinates:
                            el_rect = fitz.Rect(coordinates)
                            if index == debug_index: print(f"DEBUG Image {index}: Method 4 attempting to match element rect {el_rect} with PDF image objects.")
                            best_iou = 0.1 # Minimum IoU threshold

                            for i_idx, img_info in enumerate(image_list):
                                # img_info is (xref, smask, width, height, bpc, colorspace, altcolorspace, name, filter, #uses)
                                # Need to find bboxes for this image object on the page
                                img_bboxes = page.get_image_bboxes(img_info, transform=False)
                                if not img_bboxes: continue
                                for pdf_img_rect in img_bboxes:
                                    # Calculate IoU
                                    intersect_rect = fitz.Rect(el_rect)
                                    intersect_rect.intersect(pdf_img_rect)

                                    intersection_area = intersect_rect.width * intersect_rect.height
                                    if intersection_area <= 0: continue

                                    union_area = (el_rect.width * el_rect.height) + \
                                                 (pdf_img_rect.width * pdf_img_rect.height) - \
                                                 intersection_area
                                    if union_area <=0: continue

                                    iou = intersection_area / union_area
                                    if index == debug_index: print(f"DEBUG Image {index}: Method 4: PDF Img obj {i_idx} rect {pdf_img_rect}, IoU: {iou:.2f}")


                                    if iou > best_iou:
                                        best_iou = iou
                                        extracted_img_info = img_info
                                        if index == debug_index: print(f"DEBUG Image {index}: Method 4: Found new best PDF image object match (xref: {img_info[0]}) with IoU: {iou:.2f}")

                        if not extracted_img_info and image_list:
                            # Fallback if no specific overlap found or no coordinates:
                            # Try to find the largest image on the page, or just take the first valid one.
                            if index == debug_index: print(f"DEBUG Image {index}: Method 4: No specific overlap found or no coords, falling back to largest/first image on page {page_number}.")
                            largest_area = 0
                            candidate_img_info = None
                            for img_info in image_list:
                                try:
                                    # Extract basic info without full image data first if possible
                                    # PyMuPDF's get_images gives info, but dimensions might need extract_image
                                    # A safer way to get dimensions without full extraction might not be direct,
                                    # so we'll try a lightweight extraction or use info[2] * info[3] if reliable.
                                    # Let's try getting dimensions from the image info tuple directly first.
                                    width, height = img_info[2], img_info[3]
                                    if width > 0 and height > 0:
                                         area = width * height
                                         if area > largest_area:
                                             largest_area = area
                                             candidate_img_info = img_info
                                except Exception:
                                    # Fallback to actually extracting if dimensions aren't directly reliable
                                     try:
                                        temp_base_image = pdf_doc.extract_image(img_info[0])
                                        if temp_base_image and temp_base_image.get("image"):
                                            area = temp_base_image.get("width",0) * temp_base_image.get("height",0)
                                            if area > largest_area:
                                                largest_area = area
                                                candidate_img_info = img_info
                                     except Exception:
                                        continue # Skip images that fail extraction

                            if candidate_img_info:
                                extracted_img_info = candidate_img_info
                            elif image_list:
                                # If no largest found, just take the first one that can be extracted
                                for img_info_first in image_list:
                                    try:
                                        base_image_first = pdf_doc.extract_image(img_info_first[0])
                                        if base_image_first and base_image_first.get("image"):
                                            extracted_img_info = img_info_first
                                            break
                                    except:
                                        continue # Skip if first image extraction fails


                        if extracted_img_info:
                            xref = extracted_img_info[0]
                            base_image = pdf_doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]

                            img_filename_ext = f"{img_filename_base}.{image_ext}"
                            final_img_save_path = os.path.join(image_output_path, img_filename_ext)
                            md_img_path_ext = f"images/{img_filename_ext}"

                            with open(final_img_save_path, "wb") as img_file:
                                img_file.write(image_bytes)
                            print(f"Method 4 (PyMuPDF page.get_images) successful for Image {index} (xref: {xref})")
                            return f"\n![Image {index} Page Img Fallback ({image_ext})]({md_img_path_ext})\n"
            except Exception as e4:
                if index == debug_index: print(f"DEBUG Image {index}: Method 4 (PyMuPDF page.get_images) failed: {e4}\n{traceback.format_exc()}")
    else: # This else corresponds to "if pdf_doc:"
        if index == debug_index:
            print(f"DEBUG Image {index}: PyMuPDF methods (3 & 4) skipped because pdf_doc is not available (fitz.open likely failed).")


    # Final "Could not extract" message logic
    final_fail_message = f"\n[Image {index} - Could not extract]\n"
    if index == debug_index:
        print(f"All extraction methods conclusively failed for DEBUG Image {index}.")
    else:
         # Print a failure message if none of the methods worked.
         print(f"All attempted extraction methods failed for Image {index}. Could not extract.")

    return final_fail_message

def elements_to_markdown_improved(elements, pdf_path=None, output_dir="md_output", save_images=True):
    """
    Convert Unstructured elements to Markdown with improved image handling

    Args:
        elements: List of elements from partition_pdf
        pdf_path: Original PDF path (for fallback image extraction)
        output_dir: Directory to save markdown and images
        save_images: Whether to save images as files

    Returns:
        Full markdown string representation of the document
    """
    if save_images:
        os.makedirs(output_dir, exist_ok=True)
        image_output_path = os.path.join(output_dir, "images")
        os.makedirs(image_output_path, exist_ok=True)
    else:
        image_output_path = None

    markdown_content = []
    pdf_doc = None
    # CRITICAL: The following 'fitz.open' must work.
    # If you see "module 'fitz' has no attribute 'open'", PyMuPDF is not set up correctly.
    if pdf_path and os.path.exists(pdf_path):
        try:
            pdf_doc = fitz.open(pdf_path)
        except AttributeError as ae:
            print(f"CRITICAL PyMuPDF Error: {ae}. 'fitz.open' is not available. Please check your PyMuPDF (fitz) installation and ensure no local file is named fitz.py.")
            pdf_doc = None
        except Exception as e:
            print(f"Warning: Could not open PDF for fallback image extraction: {pdf_path}. Error: {e}")
            pdf_doc = None

    last_title_level = 1

    for i, element in enumerate(elements):
        element_type = type(element).__name__

        if element_type == "Title":
            # Simple logic for title hierarchy - may need adjustment for complex docs
            # Check if the previous element was also a Title
            if i > 0 and type(elements[i-1]).__name__ == "Title":
                # Increase level for consecutive titles, but don't exceed h6
                last_title_level = min(last_title_level + 1, 6)
            else:
                # Reset to h1 for a new title sequence
                last_title_level = 1

            if hasattr(element, 'text'):
                markdown_content.append(f"{'#' * last_title_level} {element.text.strip()}")
        elif element_type == "NarrativeText":
            if hasattr(element, 'text'):
                markdown_content.append(f"\n{element.text.strip()}\n")
        elif element_type == "ListItem":
            if hasattr(element, 'text'):
                markdown_content.append(f"- {element.text.strip()}")
        elif element_type == "Table":
            try:
                # Prefer pandas markdown if available, otherwise use HTML or raw text
                if hasattr(element, 'to_pandas'):
                    df = element.to_pandas()
                    markdown_content.append(df.to_markdown(index=False))
                elif hasattr(element, 'metadata') and hasattr(element.metadata, 'text_as_html') and element.metadata.text_as_html:
                    markdown_content.append(f"\n\n**Table (HTML):**\n\n{element.metadata.text_as_html}\n\n")
                elif hasattr(element, 'text'):
                    markdown_content.append(f"\n\n**Table Content:**\n\n{element.text.strip()}\n\n")
                else:
                    markdown_content.append("\n\n**[Table - No pandas, HTML, or text representation found]**\n\n")
            except Exception as e:
                print(f"Error processing Table element {i}: {e}")
                if hasattr(element, 'text'):
                     markdown_content.append(f"\n\n**Table Content (Error processing: {str(e)}):**\n\n{element.text.strip()}\n\n")
                else:
                    markdown_content.append(f"\n\n**[Table - Error processing: {str(e)}]**\n\n")
        elif element_type == "Image":
            if save_images and image_output_path:
                # Pass pdf_doc for PyMuPDF fallback
                image_md = handle_image_element(element, i, pdf_doc, image_output_path)
                markdown_content.append(image_md)
            else:
                markdown_content.append(f"\n[Image {i} - Image saving disabled]\n")
        elif element_type == "FigureCaption":
            if hasattr(element, 'text'):
                markdown_content.append(f"*{element.text.strip()}*\n")
        elif element_type == "Footer":
            if hasattr(element, 'text'):
                markdown_content.append(f"\n---\n*Footer: {element.text.strip()}*\n")
        elif element_type == "Text": # Generic Text element
            if hasattr(element, 'text'):
                 # Add a blank line before if the previous wasn't text/narrative/list for separation
                 if markdown_content and not markdown_content[-1].endswith('\n\n'):
                     markdown_content.append('\n')
                 markdown_content.append(f"{element.text.strip()}")

        # Add a blank line after most elements for separation, unless it's the last element
        if i < len(elements) - 1:
             next_element_type = type(elements[i+1]).__name__
             # Avoid double blank lines if the next element type also adds one (like NarrativeText)
             if element_type not in ["NarrativeText", "Table", "Footer"] and next_element_type not in ["NarrativeText", "Table", "Footer", "ListItem", "Text"]:
                  markdown_content.append("\n\n")
             elif element_type == "Text" and next_element_type == "Text":
                  markdown_content.append(" ") # Just a space for consecutive text blocks? Or new line? Depends on desired output. Let's add a newline.
                  markdown_content.append("\n")
             elif element_type == "Text" and next_element_type not in ["Text", "NarrativeText", "ListItem"]:
                 markdown_content.append("\n\n")
             elif element_type in ["Title", "ListItem", "FigureCaption"] and next_element_type not in ["NarrativeText", "Table", "Footer"]:
                 markdown_content.append("\n")


    if pdf_doc:
        pdf_doc.close()

    # Clean up empty lines potentially introduced by the logic above
    # This is a simple cleanup, more sophisticated logic might be needed
    markdown_string = "\n".join([line for line in "\n".join(markdown_content).splitlines() if line.strip() or line.startswith('#') or line.startswith('- ') or line.startswith('![')])
    markdown_string = markdown_string.replace('\n\n\n', '\n\n') # Replace triple newlines with double


    if output_dir:
        markdown_filepath = os.path.join(output_dir, "document.md")
        try:
            with open(markdown_filepath, "w", encoding="utf-8") as f:
                f.write(markdown_string)
            print(f"Markdown output saved to {markdown_filepath}")
        except Exception as e:
            print(f"Error saving markdown to {markdown_filepath}: {e}")

    return markdown_string


# --- Modified process_pdf function signature ---
def get_md_text(pdf_path, output_dir_base="md_output", poppler_path=None):
    """
    Process a PDF with improved image handling.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir_base: Base directory for output (markdown and images).
        poppler_path: Optional. The path to the poppler 'bin' directory.
                      Required if poppler is not in your system's PATH
                      and image extraction is needed.

    Returns:
        Full markdown string representation of the document, or None if processing fails.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return None

    print(f"Processing PDF: {pdf_path}")
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(output_dir_base)
    os.makedirs(output_dir, exist_ok=True)

    unstructured_image_temp_dir = os.path.join(output_dir, "unstructured_temp_images")
    os.makedirs(unstructured_image_temp_dir, exist_ok=True)
    print(f"Using temporary image extraction directory for unstructured: {unstructured_image_temp_dir}")

    # Prepare pdf_image_args dictionary
    image_extraction_args = {}
    if poppler_path:
        image_extraction_args["poppler_path"] = poppler_path
        print(f"Using poppler_path: {poppler_path} for image extraction.")
    elif os.name == 'nt': # On Windows, if no path is given, warn the user
         print("Warning: poppler_path not provided. Image extraction may fail if poppler is not in your system's PATH.")


    elements = None
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_output_dir=unstructured_image_temp_dir,
            max_characters=3000000,
            # --- Use the prepared dictionary here ---
            pdf_image_args=image_extraction_args,
        )
        print(f"Extracted {len(elements)} elements using 'hi_res' strategy.")
    except Exception as e:
        print(f"Error during partition_pdf with 'hi_res' strategy: {e}")
        traceback.print_exc()
        try:
            print("Trying 'fast' strategy as fallback...")
            elements = partition_pdf(
                filename=pdf_path,
                strategy="fast",
                extract_images_in_pdf=True,
                extract_image_block_output_dir=unstructured_image_temp_dir,
                 # --- Use the same dictionary here ---
                pdf_image_args=image_extraction_args,
            )
            print(f"Extracted {len(elements)} elements using 'fast' strategy.")
        except Exception as e_fast:
            print(f"Error during partition_pdf with 'fast' strategy: {e_fast}")
            traceback.print_exc()
            return None

    if not elements:
        print("No elements extracted from PDF. Exiting.")
        return None

    # Pass the pdf_path for PyMuPDF, and output_dir for saving markdown/images
    markdown = elements_to_markdown_improved(
        elements,
        pdf_path=pdf_path, # For PyMuPDF fallback
        output_dir=output_dir, # For saving final markdown and images subdir
        save_images=True
    )

    print(f"Conversion complete. Output saved to {os.path.join(output_dir, 'document.md')}")

    # Clean up the temporary directory created by unstructured
    if os.path.exists(unstructured_image_temp_dir):
        print(f"Cleaning up temporary image directory: {unstructured_image_temp_dir}")
        shutil.rmtree(unstructured_image_temp_dir, ignore_errors=True)

    return markdown

