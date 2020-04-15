import cv2
import pytesseract

def printed_intel_char_recog(image_file):
    # Load the required image
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    '''
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    image = cv2.medianBlur(image, 3)
    '''
    
    output = pytesseract.image_to_string(image)
    with open('output.txt', 'w') as f:
        f.write(output)
    return output
     