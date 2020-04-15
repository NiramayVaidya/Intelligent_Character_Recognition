import preprocess
import recognize
import printed_icr
from pdf2image import convert_from_path

def intel_char_recog(filename, texttype, filetype):
    output = 'default - empty string'
    if texttype.lower() == 'print':
        if filetype.lower() == 'pdf':
            pages = convert_from_path(filename, 500)
            # for page in pages:
            page[0].save('image.jpg', 'JPEG')
            output = printed_icr.printed_intel_char_recog('image.jpg')            
        elif filetype.lower() == 'image':
            output = printed_icr.printed_intel_char_recog(filename)
    elif texttype.lower() == 'handwritten':
        if filetype.lower() == 'image':
            folder = preprocess.save_words_and_lines(filename)
            output = recognize.recog(folder)
    return output
            
def main():
    print(intel_char_recog('sample_segmentation_data/png1.jpg', 'handwritten', 'image'))
    
if __name__ == '__main__':
    main()
