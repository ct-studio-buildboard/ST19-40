import docx
from docx.shared import Pt
from docx import Document
import sys
from pdf2image import convert_from_path
import os
import comtypes.client
from docx.enum.text import WD_ALIGN_PARAGRAPH

GLOBAL_PATH_TO_NOTEBOOK = "//Users//Michael//Desktop//Project//"


alphabets = {'english' : 'anikvopuxw',
             'greek' :   'αηικνορυχω'}


fonts = ['Calibri', 'Algerian', 'Arial Black', 'Arial Narrow', 'Bahnschrift',
         'Bahnschrift Condensed', 'Bahnschrift Light', 'Cambria', 'Bodoni MT Poster Compressed', 'Chiller',
         'Consolas', 'Calibri Light', 'Century', 'Century Gothic', 'Courier New',
         'Franklin Gothic Book', 'Yu Gothic UI Semilight', 'Verdana', 'Trebuchet MS', 'Sitka Text']


def alphabet_to_docs(alphabet_key, selected_font):
    local_path = alphabet_key.replace(" ", "_") + '//' + selected_font.replace(" ", "_")
    os.makedirs(GLOBAL_PATH_TO_NOTEBOOK + local_path)
    for num, letter in enumerate(alphabets[alphabet_key]):
        document = Document()
        style = document.styles['Normal']
        font = style.font
        font.name = selected_font
        font.size = docx.shared.Pt(600)
        paragraph = document.add_paragraph(letter)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        document.save(local_path + '//' + str(num) + '.docx')


def doc_to_pdf(local_directory):
    wdFormatPDF = 17
    word = comtypes.client.CreateObject('Word.Application')
    filenames = os.listdir(GLOBAL_PATH_TO_NOTEBOOK + local_directory)
    for filename in filenames:
        in_file = GLOBAL_PATH_TO_NOTEBOOK + local_directory + filename
        out_file = GLOBAL_PATH_TO_NOTEBOOK + local_directory  + filename.replace(".docx", "")
        doc = word.Documents.Open(in_file)
        doc.SaveAs(out_file, FileFormat=wdFormatPDF)
        doc.Close()
        os.remove(GLOBAL_PATH_TO_NOTEBOOK + local_directory + filename)
    word.Quit()


def pdf_to_jpeg(local_directory):
    filenames = os.listdir(GLOBAL_PATH_TO_NOTEBOOK + local_directory)
    for filename in filenames:
        pages = convert_from_path(local_directory + filename, 500)
        for page in pages:
            filename = filename.replace(".pdf", "")
            page.save(GLOBAL_PATH_TO_NOTEBOOK + local_directory + local_directory.split('//')[0] + '_' + local_directory.split('//')[1] + '_' + filename + '.jpg', 'JPEG')
        os.remove(GLOBAL_PATH_TO_NOTEBOOK + local_directory + filename + '.pdf')


for alphabet_key in alphabets:
    for selected_font in fonts:
        alphabet_to_docs(alphabet_key, selected_font)
        local_directory = alphabet_key + '//' + selected_font.replace(" ", "_") + '//'
        doc_to_pdf(local_directory)
        pdf_to_jpeg(local_directory)