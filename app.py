# pip install opencv-python pytesseract pdf2image scikit-learn pandas pillow numpy
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import pandas as pd
import cv2, pytesseract, re, os, time, sys
from pytesseract import Output
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import logging
import time

# Set up logging
# Create and configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler("my_log_file.log")  # Log file path
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Starting Logging...")

# If on Windows, set tesseract path:
def resource_path(relative_path):
    # Works both in Python and in PyInstaller exe
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

tesseract_exe = resource_path(os.path.join("tess", "tesseract.exe"))
pytesseract.pytesseract.tesseract_cmd = tesseract_exe

#pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

LANG = "guj+eng"         # Gujarati and English
DPI  = 800
TESS_CONFIG = "--oem 1 --psm 6"   # LSTM, assume uniform block per cell
COL_BOUNDS = [(0, 250), (250, 605), (605, 1280), (1280, 1490), (1490, 1950), (1950, 2120), (2120, 2371), (2371, 2830)]
COL_RENAMES = {
    0: "SR_No",
    1: "House_No",
    2: "Name_of_Elector",
    3: "Relation",
    4: "Relative_name",
    5: "Gender",
    6: "Age",
    7: "Voter_ID"
}

def process_first_page(img: Image.Image) -> dict:
    """
    Iterate over multiple section of an image and extract following details from each.
    LS_Number, Assembly Number, Part Number and Polling Station Number.
    Args:
        img: PIL Image of the page to process.
    Returns:
        A dictionary mapping section names to extracted text.
    """
    ls_number_area = img.crop((370, 500, 2600, 800))
    assembly_number_area = img.crop((370, 350, 2600, 500))
    part_number_area = img.crop((2620, 350, 3160, 500))
    polling_region_area = img.crop((450, 1253, 1935, 1920))
    polling_station_number_area = img.crop((450, 2290, 1935, 2770))

    ls_number = pytesseract.image_to_string(ls_number_area, lang=LANG, config=TESS_CONFIG).strip()
    # Consider text only after ":" if present and remove any spaces
    if ":" in ls_number:
        ls_number = ls_number.split(":", 1)[1].strip().replace(" ", "")
    assembly_number = pytesseract.image_to_string(assembly_number_area, lang=LANG, config=TESS_CONFIG).strip()
    if ":" in assembly_number:
        assembly_number = assembly_number.split(":", 1)[1].strip().replace(" ", "")
    part_number = pytesseract.image_to_string(part_number_area, lang=LANG, config=TESS_CONFIG).strip()
    if ":" in part_number:
        part_number = part_number.split(":", 1)[1].strip().replace(" ", "")
    polling_region = pytesseract.image_to_string(polling_region_area, lang=LANG, config=TESS_CONFIG).strip()
    if ":" in polling_region:
        polling_region = polling_region.split(":", 1)[1].strip()
    polling_region = re.sub(r'\s+', ' ', polling_region)
    polling_station = pytesseract.image_to_string(polling_station_number_area, lang=LANG, config=TESS_CONFIG).strip().split(":")

    if len(polling_station) > 1:
        # Search anything before first newline as polling station number
        polling_station_name = polling_station[1].strip().split("\n", 1)[0].strip()
        if len(polling_station) > 2:
            polling_station_address = polling_station[2].strip().replace("\n", " ")
        else:
            polling_station_address = ':'.join(polling_station).strip().replace("\n", " ")
    else:
        polling_station_name = ':'.join(polling_station)
        polling_station_address = ':'.join(polling_station)
    return {
        "LS_Number": ls_number,
        "Assembly_Number": assembly_number,
        "Part_Number": part_number,
        "Polling_Region": polling_region,
        "Polling_Station_Name": polling_station_name,
        "Polling_Station_Address": polling_station_address
    }

def preprocess_table_image(img: Image.Image) -> Image.Image:
    """
    Perform Preprocessing on input image
    1. Convert to greyscale
    2. Apply adaptive thresholding
    3. Light Denoising
    4. Mild Morphological Closing
    5. Final Image Normalization
    Args:
        img: PIL Image of the table area.
    Returns:
        Preprocessed image as PIL Image.
    """
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11)
    denoised = cv2.medianBlur(th, 3)
    kernel = np.ones((1,1), np.uint8)
    closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    final = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    final = Image.fromarray(final)
    return final

def process_data_page(img: Image.Image) -> pd.DataFrame:
    """
    Process a data page image and extract the borderless table as a DataFrame.
    Args:
        img: PIL Image of the data page.
    Returns:
        DataFrame containing the extracted table data.
    """
    # Preprocess the image to enhance table features
    preprocessed = preprocess_table_image(img)

    # Extract the table from the preprocessed image
    ocr_df = pytesseract.image_to_data(
        preprocessed,                   # use preprocessed image
        lang=LANG,                      # use both Gujarati and English
        config=TESS_CONFIG,             # LSTM, assume uniform block of text
        output_type=Output.DATAFRAME    # output as DataFrame
    )

    # Clean up the OCR DataFrame
    ocr_df = ocr_df[ocr_df.text.notna() & (ocr_df.text.str.strip() != "") & (ocr_df.conf.astype(float) >= 0)].copy()
    if ocr_df.empty:
        return pd.DataFrame()
    
    # Calculate the x_center of each word for clustering
    ocr_df["x_center"] = ocr_df["left"] + ocr_df["width"] / 2.0
    ocr_df["y_center"] = ocr_df["top"] + ocr_df["height"] / 2.0
    # Assign line numbers based on y_center proximity
    ocr_df = ocr_df.sort_values(by=["y_center", "x_center"]).reset_index(drop=True)
    line_num = 0
    last_y = None
    line_nums = []
    for _, row in ocr_df.iterrows():
        y = row["y_center"]
        if last_y is None or abs(y - last_y) > 50:  # 12 pixels tolerance
            line_num += 1
        line_nums.append(line_num)
        last_y = y
    ocr_df['line_num'] = line_nums

    # Assign columns based on predefined column bounds
    ocr_df['Column ID'] = pd.cut(
        ocr_df['x_center'],
        bins=[b[0] for b in COL_BOUNDS] + [COL_BOUNDS[-1][1]],
        labels=list(range(len(COL_BOUNDS)))
        )

    # Pivot the DataFrame to create a structured table
    ocr_df = ocr_df.pivot_table(
        index='line_num',
        columns='Column ID',
        values='text',
        aggfunc=lambda x: ' '.join(x),
        observed=False
    )
    ocr_df.rename(columns=COL_RENAMES, inplace=True)
    return ocr_df

def process_pdf(pdf_path: str) -> pd.DataFrame:
    """
    Process the entire PDF and extract structured data from all data pages into a single DataFrame and save it as an Excel file.
    Args:
        pdf_path: Path to the PDF file.
    Returns:
        DataFrame containing the combined extracted data from all pages.
    """
    start_time = time.time()
    logger.info(f"[PID : {os.getpid()}] Processing PDF: {pdf_path}")
    try:
        logger.info(f"[PID : {os.getpid()}] Converting PDF to images...")
        pages = convert_from_path(pdf_path, dpi=400)
        logger.info(f"[PID : {os.getpid()}] Total pages converted: {len(pages)}")

        logger.info(f"[PID : {os.getpid()}] Processing first page for metadata extraction...")

        x, y = pages[0].size
        y_shrink = 0
        y_expand = 0
        if y < 4512:
            y_shrink = 4512 - y
        elif y > 4512:
            y_expand = y - 4512
            y_shrink = 4512 - y
        metadata = process_first_page(pages[0].crop((0, 0-y_shrink, 3400, 4512+y_expand)))
        logger.info(f"[PID : {os.getpid()}] Metadata extracted: {metadata}")

        df = pd.DataFrame(columns=["SR_No", "House_No", "Name_of_Elector", "Relation", "Relative_name", "Gender", "Age", "Voter_ID"])

        logger.info(f"[PID : {os.getpid()}] Processing data pages...")
        for page in pages[1:-1]:
            logger.info(f"[PID : {os.getpid()}] Processing page {pages.index(page) + 1} of {len(pages)}")
            data = process_data_page(page.crop((250, 310-y_shrink, 3080, 4130+y_expand-y_shrink)))
            logger.info(f"[PID : {os.getpid()}] Page {pages.index(page) + 1} processed. Data extracted: {data.shape}")
            df = pd.concat([df, data], ignore_index=True)
        logger.info(f"[PID : {os.getpid()}] All data pages processed. Total records: {df.shape[0]}")

        logger.info(f"[PID : {os.getpid()}] Adding metadata to DataFrame") 
        df['LS_Number'] = metadata['LS_Number']
        df['Assembly_Number'] = metadata['Assembly_Number']
        df['Part_Number'] = metadata['Part_Number']
        df['Polling_Region'] = metadata['Polling_Region']
        df['Polling_Station_Name'] = metadata['Polling_Station_Name']
        df['Polling_Station_Address'] = metadata['Polling_Station_Address']

        logger.info(f"[PID : {os.getpid()}] Saving extracted data to Excel file...")
        df[['LS_Number', 'Assembly_Number', 'Part_Number', 'SR_No', 'House_No', 'Name_of_Elector', 'Relation',
       'Relative_name', 'Gender', 'Age', 'Voter_ID',  'Polling_Region',
       'Polling_Station_Name', 'Polling_Station_Address']].to_excel(pdf_path[:-4] + ".xlsx", index=False)

        end_time = time.time()
        logger.info(f"[PID : {os.getpid()}] PDF processing completed in {end_time - start_time:.2f} seconds for {pdf_path}.")
        return df
    except Exception as e:
        logger.error(f"[PID : {os.getpid()}] Error processing PDF {pdf_path}: {e}")
        return pd.DataFrame()

def process_pdf_folder_parallel(folder_path: str) -> None:
    """
    Process all PDF files parallelly in a folder and extract structured data.
    Save each extracted DataFrame to a Excel file and merge all excels into a single DataFrame.
    Args:
        folder_path: Path to the folder containing PDF files.
    Returns:
        None
    """
    pdf_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    logger.info(f"[PID : {os.getpid()}] Found {len(pdf_paths)} PDF files to process.")
    logger.info(f"[PID : {os.getpid()}] PDF files: {pdf_paths}")
    all_data = []

    with ProcessPoolExecutor() as executor:
        for data in executor.map(process_pdf, pdf_paths):
            try:
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                logger.error(f"[PID : {os.getpid()}] Error processing data: {e}")

    logger.info(f"[PID : {os.getpid()}] Combining all extracted data into a single DataFrame...")
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df[['LS_Number', 'Assembly_Number', 'Part_Number', 'SR_No', 'House_No', 'Name_of_Elector', 'Relation',
       'Relative_name', 'Gender', 'Age', 'Voter_ID',  'Polling_Region',
       'Polling_Station_Name', 'Polling_Station_Address']].to_excel(os.path.join(folder_path, f"{os.path.basename(folder_path)}.xlsx"))
    logger.info(f"[PID : {os.getpid()}] Combined data saved to {os.path.join(folder_path, f'{os.path.basename(folder_path)}.xlsx')}")

if __name__ == "__main__":
    # multiprocessing.freeze_support()
    # logger.info(f"[PID : {os.getpid()}] Entering main() before input()")

    # directory_path = input("Enter the directory path containing PDF files: ").strip()

    paths = ["P143(Nasavadi)"]
    for directory_path in paths:
        logger.info(f"[PID : {os.getpid()}] Got directory path: {directory_path}")
        process_pdf_folder_parallel(directory_path)

    logger.info(f"[PID : {os.getpid()}] Finished processing. Exiting.")