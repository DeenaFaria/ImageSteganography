import math

from PIL import Image
from cv2 import imread,imwrite
import zlib

import numpy as np
from numpy import array
from base64 import urlsafe_b64encode
from hashlib import md5
from cryptography.fernet import Fernet
from custom_exceptions import *
from statistics import mean
from math import log10, sqrt
import cv2

#Function for compressing image
def con(im):
    with open(im, 'rb') as f:
         image_data = f.read()

# Compress image data
    compressed_data = zlib.compress(image_data)

# Write compressed data to file
    with open('compressed_image.zlib', 'wb') as f:
         f.write(compressed_data)

    return 'compressed_image.zlib'

#Function for decompressing image
def decom(im):
    with open('compressed_image.zlib', 'rb') as f:
         compressed_data = f.read()

# Decompress data
    image_data = zlib.decompress(compressed_data)

# Write decompressed data to file
    with open('decompressed_image.png', 'wb') as f:
         f.write(image_data)

    return 'decompressed_image.png'

#Function for calculating the average of the stego key
def avg(stego):
    asc=[]
    sss=stego
    for i in sss:
       aaa=ord(i)
    asc.append(aaa)

    #print (mean(asc))

    return mean(asc)

#Function for calculating PSNR
def PSNR(original, encoded):
    mse = np.mean((original - encoded) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

#Returns binary representation of a string
def str2bin(string):
    return ''.join((bin(ord(i))[2:]).zfill(7) for i in string)

#Returns text representation of a binary string
def bin2str(string):
    return ''.join(chr(int(string[i:i+7],2)) for i in range(len(string))[::7])

#Returns the encrypted/decrypted form of string depending upon mode input
def encrypt_decrypt(string,password,mode='enc'):
    _hash = md5(password.encode()).hexdigest()
    cipher_key = urlsafe_b64encode(_hash.encode())
    cipher = Fernet(cipher_key)
    if mode == 'enc':
        return cipher.encrypt(string.encode()).decode()
    else:
        return cipher.decrypt(string.encode()).decode()


#Encodes secret data in image
def encode(input_filepath,text,output_filepath,avg):

    data = text
    lm=len(data) #length of the message
    print("Data length:"+str(lm))

    data_length = bin(lm)[2:].zfill(32) #Converts the length to binary and inserts zeros to ensure that first 32 bits contains the data length 
    
    bin_data = iter(str2bin(data))

    img = imread(input_filepath,1)
    if img is None:
        raise FileError("The image file '{}' is inaccessible".format(input_filepath))
    
    height,width = img.shape[0],img.shape[1]
    encoding_capacity = height*width #working on single channel
    total_bits = 32+len(data)*7
    if total_bits > encoding_capacity:
        raise DataError("The data size is too big to fit in this image!")
    completed = False
    modified_bits = 0
    progress = 0
    progress_fraction = 1/total_bits
    
    k1=avg #user provided stego key

    lp=encoding_capacity-avg #Number of available pixels for encoding
    print("Old lp="+str(lp))
    
    # Update the starting point (i, j) based on K2
    if lm > lp:  # If the length of the data is larger than the number of available pixels
        k2 = encoding_capacity % lm  # Define a new stego key
        i = int(k2 / width)
        j = (k2 - i * width)%width
        i += 1  # Increment i by 1
        lp = encoding_capacity - k2

    else:
        # Retain the original stego key
        i = int(k1 / width)
        j = (k1 - i * width)%width
        i+=1
        lp = encoding_capacity - k1

    print("New lp="+str(lp))


    print("width"+str(width))    
    pixel_jump = int(lp / lm)  # Calculate pixel_jump based on available pixels and data length
    print(pixel_jump)

    

    ijump = int(pixel_jump / width)
    jjump = (pixel_jump - ijump * width)%width
    ijump+=1

    print("i,j,ijump,jjump"+str(":")+str(i)+","+str(j)+","+str(ijump)+","+str(jjump))
    
    counter=0
    new_counter=0
    res=0


    # Encoding process
    for a in range(i, height,ijump):
        b=j
        while b < width:
            pixel = img[a, b, 0]  # Seed pixel on red channel
            try:
                x = next(bin_data)
            except StopIteration:
                completed = True
                break

            if x == '0' and pixel % 2 == 1:
                pixel -= 1
                modified_bits += 1
            elif x == '1' and pixel % 2 == 0:
                pixel += 1
                modified_bits += 1

            img[a, b, 0] = pixel

            
            b += jjump  # Skip by pixel_jump
            counter+=1
            

            if completed:
                break


        j=0   
     

        if completed:
            break

    
    print("\nEncoding completed.")


    written = imwrite(output_filepath,img)
    if not written:
        raise FileError("Failed to write image '{}'".format(output_filepath))
    loss_percentage = (modified_bits/encoding_capacity)*100
    return written

#Extracts secret data from input image
def decode(lm,avg,input_filepath):
    result,extracted_bits,completed,number_of_bits = '',0,False,None    

    img = imread(input_filepath)
    if img is None:
        raise FileError("The image file '{}' is inaccessible".format(input_filepath))
    height,width = img.shape[0],img.shape[1]
    encoding_capacity = height*width
 
    k1=avg #user provided stego key

    lp=encoding_capacity-avg #Number of available pixels for encoding
    
    # Update the starting point (i, j) based on K2
    if lm > lp:  # If the length of the data is larger than the number of available pixels
        k2 = encoding_capacity % lm  # Define a new stego key
        i = int(k2 / width)
        j = k2 - i * width
        i += 1  # Increment i by 1
        lp = encoding_capacity - k2
 
    else:
        # Retain the original stego key
        i = int(k1 / width)
        j = k1 - i * width
        i+=1
        lp = encoding_capacity - k1
   


    data_length=lm
   
    pixel_jump=int(lp/lm)

    ijump = int(pixel_jump / width)
    jjump = (pixel_jump - ijump * width)%width
    ijump+=1
    
    nb=data_length*7
    
    counter=0
    new_counter=0
    res=0
    
    
    for a in range(i,height,ijump):
        b=j
        while b < width:
            pixel=img[a,b,0]
           
            result += str(pixel%2)
            
            extracted_bits += 1

            b=b+jjump
            counter+=1
                
            if extracted_bits == nb: #If the number of the extracted bits matches the data length of binary bits
                completed = True
                break

        j=0  

   
        if completed:
            break
        

        
   
    return bin2str(result)


if __name__ == "__main__":

    ch = int(input('Choose an option!\n\n1.Encrypt\n2.Decrypt\n\nInput(1/2): '))
    if ch == 1:
        ip_file = input('\nEnter cover image name(path)(with extension): ')
       
        text = input('Enter secret data: ')
        
        stego = input('Enter Stego key: ')
        aveg=avg(stego)
        op_file = input('Enter output image name(path)(with extension): ')

        
        try:
            original = cv2.imread(ip_file)
        
            Encoded = encode(ip_file,text,op_file,aveg)
            #print(Encoded)
            stegoImage=cv2.imread(op_file)
            
            

        except FileError as fe:
            print("Error: {}".format(fe))
        except DataError as de:
            print("Error: {}".format(de))
        else:
   
            print("Encoded Successfully!\n")
            value = PSNR(original, Encoded)
            print(f"PSNR value is {value} dB")
            Encoded = cv2.imread(op_file)
            cv2.imshow("original image", original)
            cv2.imshow("Encoded image", Encoded)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
       
    elif ch == 2:
        ip_file = input('Enter image path: ')
        
        
        stego = input('Enter Stego key: ')
        aveg=avg(stego)
        lm = int(input('Enter data length: '))

        try:
            data = decode(lm,aveg,ip_file)
        except FileError as fe:
            print("Error: {}".format(fe))
        except PasswordError as pe:
            print('Error: {}'.format(pe))
        else:
            print('Decrypted data:',data)
    else:
        print('Wrong Choice!')
