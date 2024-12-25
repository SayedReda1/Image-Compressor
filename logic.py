"""
This class contains all the logic for the vector quantization
encode and decode algorithms.

Algorithm for encoding image:
1. Read image using PIL

2. Compute best n vectors to represent the image (codebook, size)
- average(vectors) -> codebook = dict(0, Vector)
- while True:
    - if codebook.size < size: dict1 = split() TODO
    - foreach vector: selectNearestVector(dict1)
    - codebook = average(vectors) TODO
    - if codebook == dict1 and codebook.size == size: break
- return codebook

3. create a matrix of labels which best represents the image (compressed image)
"""

import pickle
import numpy as np
from PIL import Image
from math import ceil, floor, log2


#########################################
############ Input / Ouput ##############
#########################################
def readImage(imagePath: str) -> np.array:
    """ Reads grayscale image and returns matrix of vectors """
    # Read image
    image = Image.open(imagePath).convert('L')
    imageMatrix = np.array(image)

    return imageMatrix


def writeImage(imageMatrix: np.array, imagePath: str):
    """ Converts matrix of vectors to image and writes it to disk """
    image = Image.fromarray(imageMatrix)
    image.save(imagePath)

def readFromCompressedFile(filePath: str) -> tuple:
    """ Reads the compressed file and returns labels and codebook """
    with open(filePath, "rb") as file:
        data = pickle.load(file)

    return data["labels"], data["codebook"]

def writeToCompressedFile(labels: list[list[int]], codebook: dict, filePath: str):
    """ Writes the compressed matrix and codebook to a file """
    data = {
        "labels": labels,
        "codebook": codebook
    }
    with open(filePath, "wb") as file:
        pickle.dump(data, file)

#########################################
########### Helper Functions ############
#########################################
def partition(imageMatrix: np.array, blockSize: tuple) -> np.array:
    """ Partitions the image matrix into blocks """
    bh, bw = blockSize
    img_h, img_w = imageMatrix.shape
    # add padding if needed
    if img_h % bh != 0:
        img_h += bh - img_h % bh
    if img_w % bw != 0:
        img_w += bw - img_w % bw
    blocks = np.zeros((img_h//bh, img_w//bw, bh, bw), dtype='B')

    for i in range(0, img_h, bh):
        for j in range(0, img_w, bw):
            blocks[i//bh][j//bw] = imageMatrix[i:i+bh, j:j+bw]
    
    return blocks

def split(codebook: dict[np.array]) -> dict:
    """ Splits the each codeword into two vectors """
    newCodebook = dict()
    for i, vector in codebook.items():
        vector1 = np.array(vector)
        vector2 = np.array(vector)

        # Take pixels lower bound
        vector1 = np.vectorize(lambda x: ceil(x)-1)(vector1).astype('f')
        # Take pixels upper bound
        vector2 = np.vectorize(lambda x: floor(x)+1)(vector2).astype('f')

        newCodebook[i*2] = vector1
        newCodebook[i*2+1] = vector2

    return newCodebook

def average(blocks: np.array,
             nearestBlockMatrix: np.array,) -> dict:
    """ Calculates the average of vectors that have the same nearest vector """
    result = dict()
    freq = dict()
    height, width, bh, bw = blocks.shape

    # Sum related vectors
    for i in range(0, height):
        for j in range(0, width):
            vector = blocks[i][j]
            nearest = nearestBlockMatrix[i][j]
            if nearest not in result:
                result[nearest] = np.zeros(blockSize, dtype='f')
                freq[nearest] = 0
            result[nearest] += vector
            freq[nearest] += 1
    
    # Divide each vector by frequency
    for i in result.keys():
        result[i] = np.round(result[i] / freq[i], 1).astype('f')
    
    return result

def selectNearestVector(vector: np.array, codebook: dict) -> int:
    """ Selects the nearest vector to the given vector """
    minDistance = float('inf')
    nearestVector = -1

    for i, codeVector in codebook.items():
        distance = np.abs(vector - codeVector).sum()
        if distance < minDistance:
            minDistance = distance
            nearestVector = i
    
    return nearestVector

def distributeBlocks(blocks: np.array, nearestBlockMatrix: np.array,
                      codebook: dict):
    """ Distributes blocks to their nearest vector """
    height, width, bh, bw = blocks.shape

    # Nearest codeword for each vector
    for i in range(0, height):
        for j in range(0, width):
            block = blocks[i][j]
            nearestBlockMatrix[i][j] = selectNearestVector(block, codebook)

def equal(codebook1: dict, codebook2: dict) -> bool:
    """ Compares two codebooks """
    if len(codebook1) != len(codebook2):
        return False
    
    for i in codebook1.keys():
        if np.all(np.abs(codebook1[i] - codebook2[i]) > 1):
            return False
    
    return True

def generateCodebook(blocks: np.array,
                     nearestBlockMatrix: np.array,
                     codebookSize: int) -> dict:
    """ Generates best codebook of required size to represent the image """
    # Change required size to the nearest power of 2
    codebookSize = 2**ceil(log2(codebookSize))
    codebook = average(blocks, nearestBlockMatrix)

    # Loop until the condition is met or the max iterations is reached
    while True:
        # Split codebook if required
        if len(codebook) < codebookSize:
            codebook = split(codebook)
        
        # Distribute vectors to each's nearest vector
        # nearest vector is manipulated by reference
        distributeBlocks(blocks, nearestBlockMatrix, codebook)
        
        # Calculate new codebook
        newCodebook = average(blocks, nearestBlockMatrix)
        if len(newCodebook) == codebookSize and equal(codebook, newCodebook):
            break
        codebook = newCodebook

    # Round the values in codebook
    for i in codebook.keys():
        codebook[i] = np.round(codebook[i]).astype('B')
    
    return codebook
    
#########################################
############ Main functions #############
#########################################
def compress(imagePath: str, codebookSize: int, blockSize: list):
    """ Compresses the image using vector quantization """
    # Read image
    imageMatrix = readImage(imagePath)
    blocks = partition(imageMatrix, blockSize)

    # Generate codebook & select nearest block for each vector
    # nearest block matrix is the compressed matrix
    nearestBlockMatrix = np.zeros((blocks.shape[0], blocks.shape[1]), dtype='i')
    codebook = generateCodebook(blocks, nearestBlockMatrix, codebookSize)

    # Return the compressed image matrix and codebook
    return nearestBlockMatrix, codebook

def decompress(labels, codebook) -> np.array:
    """ Decompresses the compressed matrix using the codebook """
    bh, bw = codebook[0].shape
    img_h, img_w = labels.shape[0]*bh, labels.shape[1]*bw
    decompressed = np.zeros((img_h, img_w), dtype='B')

    # Fill decompressed matrix
    for i in range(0, img_h, bh):
        for j in range(0, img_w, bw):
            vector = codebook[labels[i//bh][j//bw]]
            for x in range(bh):
                for y in range(bw):
                    decompressed[i+x][j+y] = vector[x][y]
    
    return decompressed

if __name__ == "__main__":
    imagePath = "tests/house.bmp"
    blockSize = (2, 2)
    codebookSize = 32

    imageVectors = readImage(imagePath)
    labels, codebook = compress(imagePath, codebookSize, blockSize)
    writeToCompressedFile(labels, codebook, "tests/compressed.bin")

    print("Compressed")

    # Decompress
    labels, codebook = readFromCompressedFile("tests/compressed.bin")
    decompressed = decompress(labels, codebook)
    writeImage(decompressed, "tests/decompressed.bmp")

    print("Decompressed")

    # image = np.array([
    #     [1, 2, 7, 9, 4, 11],
    #     [3, 4, 6, 6, 12, 12],
    #     [4, 9, 15, 14, 9, 9],
    #     [10, 10, 20, 18, 8, 8],
    #     [4, 3, 17, 16, 1, 4],
    #     [4, 5, 18, 18, 5, 6],
    # ])
    
    # blocks = partition(image, (2, 2))
    # nearestBlockMatrix = np.zeros((3, 3), dtype='i')
    # codebook = generateCodebook(blocks, nearestBlockMatrix, 4)
    # print(codebook)
    # print(nearestBlockMatrix)