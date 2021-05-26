import easyocr
import cv2
import mysql.connector
import smtplib

# print(h, w, c)
def imgtext():
    img = cv2.imread("crop.jpg")
    h, w, c = img.shape
    l = []
    with open("result.txt",'r+') as f:
        for i in f.read().splitlines():
            for j in i.split():
                l.append(float(j))
    f.close()
    # print(l)
    c_x = int(l[1]*w)
    c_y = int(l[2]*h)
    wi = int(l[3]*w)
    hi = int(l[4]*h)
    x = int(c_x - wi/2)
    y = int(c_y - hi/2)

    # print(x, y, wi, hi)

    img1 = img[y:y+hi,x:x+wi]

    reader = easyocr.Reader(['en'])
    read = reader.readtext(img1)
    a = read[0][1]
    license = ''.join(a.split()).upper()
    print(license)
    # cv2.imshow('o',img1)
    # cv2.waitKey(0)

    mydb = mysql.connector.connect(host = 'us-cdbr-east-03.cleardb.com', user='', passwd = '',database = 'heroku_22c8bc2b477dc6c')

    cursor = mydb.cursor()
    cursor.execute(f"SELECT MailAddress FROM license_number where Number = '{license}'")
    result = cursor.fetchall()

    smtp_server = 'smtp.gmail.com'
    sender_email = 'abhi.chaudhary43@gmail.com'
    receiver_email = str(result[0][0])
    sender_pass = 'Motog@234'

    message = 'Defaulter ' + license

    with smtplib.SMTP(smtp_server, 587) as server:
        server.starttls()
        server.login(sender_email,sender_pass)
        server.sendmail(sender_email,receiver_email,message)
        print('Success')

if __name__ =='__main__':
    imgtext()