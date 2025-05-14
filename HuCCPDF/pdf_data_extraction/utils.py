import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import re
import requests
import pymupdf4llm
import markdown

# Define the output di

# Function to convert PDF page to image and save it
def pdf_page_to_image(pdf_path, page_number, index, output_dir=None):
    
    if os.path.exists(pdf_path):
        document = fitz.open(pdf_path)
    else:
        res = requests.get(pdf_path)
        document = fitz.open(stream=res.content, filetype="pdf")
    page = document.load_page(page_number - 1)  # Pages are 0-indexed in PyMuPDF
    pix = page.get_pixmap()
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    
    # Save image as PNG file
    if output_dir is not None:
        image_filename = f'page_{index}.png'
        image_path = os.path.join(output_dir, image_filename)
        
        img.save(image_path)
        
        return image_filename
    else: 
        return img

def column_boxes(page, footer_margin=50, header_margin=50, no_image_text=True):
    """Determine bboxes which wrap a column."""
    paths = page.get_drawings()
    bboxes = []

    # path rectangles
    path_rects = []

    # image bboxes
    img_bboxes = []

    # bboxes of non-horizontal text
    # avoid when expanding horizontal text boxes
    vert_bboxes = []

    # compute relevant page area
    clip = +page.rect
    clip.y1 -= footer_margin  # Remove footer area
    clip.y0 += header_margin  # Remove header area

    def can_extend(temp, bb, bboxlist):
        """Determines whether rectangle 'temp' can be extended by 'bb'
        without intersecting any of the rectangles contained in 'bboxlist'.

        Items of bboxlist may be None if they have been removed.

        Returns:
            True if 'temp' has no intersections with items of 'bboxlist'.
        """
        for b in bboxlist:
            if not intersects_bboxes(temp, vert_bboxes) and (
                b == None or b == bb or (temp & b).is_empty
            ):
                continue
            return False

        return True

    def in_bbox(bb, bboxes):
        """Return 1-based number if a bbox contains bb, else return 0."""
        for i, bbox in enumerate(bboxes):
            if bb in bbox:
                return i + 1
        return 0

    def intersects_bboxes(bb, bboxes):
        """Return True if a bbox intersects bb, else return False."""
        for bbox in bboxes:
            if not (bb & bbox).is_empty:
                return True
        return False

    def extend_right(bboxes, width, path_bboxes, vert_bboxes, img_bboxes):
        """Extend a bbox to the right page border.

        Whenever there is no text to the right of a bbox, enlarge it up
        to the right page border.

        Args:
            bboxes: (list[IRect]) bboxes to check
            width: (int) page width
            path_bboxes: (list[IRect]) bboxes with a background color
            vert_bboxes: (list[IRect]) bboxes with vertical text
            img_bboxes: (list[IRect]) bboxes of images
        Returns:
            Potentially modified bboxes.
        """
        for i, bb in enumerate(bboxes):
            # do not extend text with background color
            if in_bbox(bb, path_bboxes):
                continue

            # do not extend text in images
            if in_bbox(bb, img_bboxes):
                continue

            # temp extends bb to the right page border
            temp = +bb
            temp.x1 = width

            # do not cut through colored background or images
            if intersects_bboxes(temp, path_bboxes + vert_bboxes + img_bboxes):
                continue

            # also, do not intersect other text bboxes
            check = can_extend(temp, bb, bboxes)
            if check:
                bboxes[i] = temp  # replace with enlarged bbox

        return [b for b in bboxes if b != None]

    def clean_nblocks(nblocks):
        """Do some elementary cleaning."""

        # 1. remove any duplicate blocks.
        blen = len(nblocks)
        if blen < 2:
            return nblocks
        start = blen - 1
        for i in range(start, -1, -1):
            bb1 = nblocks[i]
            bb0 = nblocks[i - 1]
            if bb0 == bb1:
                del nblocks[i]

        # 2. repair sequence in special cases:
        # consecutive bboxes with almost same bottom value are sorted ascending
        # by x-coordinate.
        y1 = nblocks[0].y1  # first bottom coordinate
        i0 = 0  # its index
        i1 = -1  # index of last bbox with same bottom

        # Iterate over bboxes, identifying segments with approx. same bottom value.
        # Replace every segment by its sorted version.
        for i in range(1, len(nblocks)):
            b1 = nblocks[i]
            if abs(b1.y1 - y1) > 10:  # different bottom
                if i1 > i0:  # segment length > 1? Sort it!
                    nblocks[i0 : i1 + 1] = sorted(
                        nblocks[i0 : i1 + 1], key=lambda b: b.x0
                    )
                y1 = b1.y1  # store new bottom value
                i0 = i  # store its start index
            i1 = i  # store current index
        if i1 > i0:  # segment waiting to be sorted
            nblocks[i0 : i1 + 1] = sorted(nblocks[i0 : i1 + 1], key=lambda b: b.x0)
        return nblocks

    # extract vector graphics
    for p in paths:
        path_rects.append(p["rect"].irect)
    path_bboxes = path_rects

    # sort path bboxes by ascending top, then left coordinates
    path_bboxes.sort(key=lambda b: (b.y0, b.x0))

    # bboxes of images on page, no need to sort them
    for item in page.get_images():
        img_bboxes.extend(page.get_image_rects(item[0]))

    # blocks of text on page
    blocks = page.get_text(
        "dict",
        flags=fitz.TEXTFLAGS_TEXT,
        clip=clip,
    )["blocks"]

    # Make block rectangles, ignoring non-horizontal text
    for b in blocks:
        bbox = fitz.IRect(b["bbox"])  # bbox of the block

        # ignore text written upon images
        if no_image_text and in_bbox(bbox, img_bboxes):
            continue

        # confirm first line to be horizontal
        line0 = b["lines"][0]  # get first line
        if line0["dir"] != (1, 0):  # only accept horizontal text
            vert_bboxes.append(bbox)
            continue

        srect = fitz.EMPTY_IRECT()
        for line in b["lines"]:
            lbbox = fitz.IRect(line["bbox"])
            text = "".join([s["text"].strip() for s in line["spans"]])
            if len(text) > 1:
                srect |= lbbox
        bbox = +srect

        if not bbox.is_empty:
            bboxes.append(bbox)

    # Sort text bboxes by ascending background, top, then left coordinates
    bboxes.sort(key=lambda k: (in_bbox(k, path_bboxes), k.y0, k.x0))

    # Extend bboxes to the right where possible
    bboxes = extend_right(
        bboxes, int(page.rect.width), path_bboxes, vert_bboxes, img_bboxes
    )

    # immediately return of no text found
    if bboxes == []:
        return []

    # --------------------------------------------------------------------
    # Join bboxes to establish some column structure
    # --------------------------------------------------------------------
    # the final block bboxes on page
    nblocks = [bboxes[0]]  # pre-fill with first bbox
    bboxes = bboxes[1:]  # remaining old bboxes

    for i, bb in enumerate(bboxes):  # iterate old bboxes
        check = False  # indicates unwanted joins

        # check if bb can extend one of the new blocks
        for j in range(len(nblocks)):
            nbb = nblocks[j]  # a new block

            # never join across columns
            if bb == None or nbb.x1 < bb.x0 or bb.x1 < nbb.x0:
                continue

            # never join across different background colors
            if in_bbox(nbb, path_bboxes) != in_bbox(bb, path_bboxes):
                continue

            temp = bb | nbb  # temporary extension of new block
            check = can_extend(temp, nbb, nblocks)
            if check == True:
                break

        if not check:  # bb cannot be used to extend any of the new bboxes
            nblocks.append(bb)  # so add it to the list
            j = len(nblocks) - 1  # index of it
            temp = nblocks[j]  # new bbox added

        # check if some remaining bbox is contained in temp
        check = can_extend(temp, bb, bboxes)
        if check == False:
            nblocks.append(bb)
        else:
            nblocks[j] = temp
        bboxes[i] = None

    # do some elementary cleaning
    nblocks = clean_nblocks(nblocks)

    # return identified text bboxes
    return nblocks

def pdf_page_to_markdown(pdf_path, page_number):
    md_text = pymupdf4llm.to_markdown(pdf_path, pages=[page_number - 1])
    return md_text

def pdf_page_to_html(pdf_path, page_number):
    if os.path.exists(pdf_path):
        document = fitz.open(pdf_path)
    else:
        res = requests.get(pdf_path)
        document = fitz.open(stream=res.content, filetype="pdf")
    page = document.load_page(page_number - 1)  # Pages are 0-indexed in PyMuPDF
    text = page.get_text("html")
    return text


def pdf_page_to_text(pdf_path, page_number):
    if os.path.exists(pdf_path):
        document = fitz.open(pdf_path)
    else:
        res = requests.get(pdf_path)
        document = fitz.open(stream=res.content, filetype="pdf")
    page = document.load_page(page_number - 1)  # Pages are 0-indexed in PyMuPDF
    page.clean_contents()
    bboxes = column_boxes(page, no_image_text=True)

    pdf_data = {}
    # bboxes is a list of fitz.IRect objects, that are sort ascending by their y0,
    # then x0 coordinates. Their text content can be extracted by all PyMuPDF
    # get_text() variants, like for instance the following:
    text_blocks = []

    for rect in bboxes:
        text_blocks.extend(page.get_text("blocks", clip=rect, sort=True, flags=fitz.TEXTFLAGS_BLOCKS & ~fitz.TEXT_PRESERVE_IMAGES))
        
    # if there are no text blocks, skip the page
    if not text_blocks:
        return ""
    
    pdf_data['texts'] = []
    for text_block in text_blocks:
        string_block = text_block[4].strip("\n").replace("-\n", "")
        string_block = re.sub(r'(\w)\n(\w)', r'\1 \2', string_block)

        text_block_coords = (text_block[0], text_block[1], text_block[2], text_block[3])
        
        if pdf_data["texts"] and isinstance(pdf_data["texts"][-1], str):
            pdf_data["texts"][-1] += string_block if pdf_data["texts"][-1].endswith(" ") else " " + string_block
        else:
            pdf_data["texts"].append(string_block)


    return "\n\n".join(pdf_data['texts'])