"""
This class contains all the logic for the vector quantization
encode and decode algorithms.

Algorithm for encoding image:
1. Read image using PIL
2. Partition the image into vector objects of required size

3. Compute best n vectors to represent the image (codebook, size)
- average(vectors) -> codebook = dict(0, Vector)
- while True:
    - if codebook.size < size: dict1 = split() TODO
    - foreach vector: selectNearestVector(dict1)
    - codebook = average(vectors) TODO
    - if codebook == dict1 and codebook.size == size: break
- return codebook

4. create a matrix of labels which best represents the image (compressed image)
"""

import pickle
import numpy as np
from PIL import Image
from math import ceil, floor, log2

class Vector:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels = [[0 for _ in range(width)] for _ in range(height)]
        self.nearestVector = 0
    
    def calculateDistance(self, vector):
        distance = 0
        for i in range(self.height):
            for j in range(self.width):
                distance += abs(self.pixels[i][j] - vector.pixels[i][j])
        return distance
    
    def selectNearestVector(self, vectors: dict):
        minDistance = float('inf')
        for i, vector in vectors.items():
            distance = self.calculateDistance(vector)
            if distance < minDistance:
                minDistance = distance
                self.nearestVector = i

#########################################
############ Input / Ouput ##############
#########################################
def readFromImage(imagePath: str, blockSize: list) -> list[list[Vector]]:
    """ Reads grayscale image and returns matrix of vectors """
    # Read image
    image = Image.open(imagePath).convert('L')
    imageMatrix = np.array(image).tolist()

    # Create matrix of vectors
    blockWidth, blockHeight = blockSize
    imageWidth, imageHeight = image.size
    vectors = []
    for i in range(0, imageHeight, blockHeight):
        row = []
        for j in range(0, imageWidth, blockWidth):
            vector = Vector(blockWidth, blockHeight)
            # Fill vector with pixels
            for x in range(blockHeight):
                for y in range(blockWidth):
                    vector.pixels[x][y] = imageMatrix[i+x][j+y]
            row.append(vector)
        vectors.append(row)

    return vectors


def writeToImage(imageMatrix: list[list[Vector]], imagePath: str):
    """ Converts matrix of vectors to image and writes it to disk """
    # Write image
    matrix = np.array(imageMatrix)
    image = Image.fromarray(np.array(matrix).astype(np.uint8))
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
def split(codebook: dict[Vector]) -> dict:
    """ Splits the each codeword into two vectors """
    newCodebook = dict()
    for i, vector in codebook.items():
        vector1 = Vector(vector.width, vector.height)
        vector2 = Vector(vector.width, vector.height)

        # Take pixels lower bound
        vector1.pixels = [[ceil(x)-1 for x in row] for row in vector.pixels]
        # Take pixels upper bound
        vector2.pixels = [[floor(x)+1 for x in row] for row in vector.pixels]

        newCodebook[i*2] = vector1
        newCodebook[i*2+1] = vector2

    return newCodebook

def average(imageVectors: list[list[Vector]]) -> dict:
    """ Calculates the average of vectors that have the same nearest vector """
    result = dict()
    freq = dict()

    # Sum related vectors
    for vectors in imageVectors:
        for vector in vectors:
            if vector.nearestVector not in result:
                result[vector.nearestVector] = Vector(vector.width, vector.height)
                freq[vector.nearestVector] = 0
            # Sum each pixel
            for i in range(vector.height):
                for j in range(vector.width):
                    result[vector.nearestVector].pixels[i][j] += vector.pixels[i][j]
            # Increment frequency
            freq[vector.nearestVector] += 1
    
    # Divide by number of vectors
    for i, vector in result.items():
        # Divide each pixel in vector
        for x in range(vector.height):
            for y in range(vector.width):
                vector.pixels[x][y] = round(vector.pixels[x][y] / freq[i], 2)
    
    return result

def generateCodebook(imageVectors: list[list[Vector]], requiredSize: int) -> dict:
    """ Generates best codebook of required size to represent the image """
    # Change required size to the nearest power of 2
    requiredSize = 2**ceil(log2(requiredSize))
    maxIterations = log2(requiredSize) + 10
    codebook = average(imageVectors)

    # Loop until the condition is met or the max iterations is reached
    for _ in range(int(maxIterations)):
        # Split codebook if required
        if len(codebook) < requiredSize:
            codebook = split(codebook)
        
        # Distribute vectors to nearest each's vector
        for vectors in imageVectors:
            for vector in vectors:
                vector.selectNearestVector(codebook)
        
        # Calculate new codebook
        newCodebook = average(imageVectors)
        if newCodebook == codebook and len(newCodebook) == requiredSize:
            break
        codebook = newCodebook

    # Round the values in codebook
    for i, vector in codebook.items():
        for x in range(vector.height):
            for y in range(vector.width):
                vector.pixels[x][y] = round(vector.pixels[x][y])
    
    return codebook
    
def generateLabels(imageVectors: list[list[Vector]]) -> list[list[int]]:
    """ Generates the compressed matrix of labels """
    labels = []
    # Create matrix of labels
    for vectors in imageVectors:
        row = []
        for vector in vectors:
            row.append(vector.nearestVector)
        labels.append(row)

    return labels

#########################################
############ Main functions #############
#########################################
def compress(imagePath: str, codebookSize: int, blockSize: list):
    """ Compresses the image using vector quantization """
    # Read image
    imageVectors = readFromImage(imagePath, blockSize)

    # Generate codebook & select nearest block for each vector
    codebook = generateCodebook(imageVectors, codebookSize)

    compressed_matrix = []
    # Create matrix of labels
    for vectors in imageVectors:
        row = []
        for vector in vectors:
            row.append(vector.nearestVector)
        compressed_matrix.append(row)

    # Return the compressed image matrix and codebook
    return compressed_matrix, codebook

def decompress(labels, codebook) -> list[list[int]]:
    """ Decompresses the compressed matrix using the codebook """
    blockWidth, blockHeight = codebook[0].width, codebook[0].height
    imageWidth, imageHeight = codebook[0].width * len(labels[0]), codebook[0].height * len(labels)
    decompressed = [[0 for _ in range(imageWidth)] for _ in range(imageHeight)]

    # Fill decompressed matrix
    for i in range(0, imageHeight, blockHeight):
        for j in range(0, imageWidth, blockWidth):
            vector = codebook[labels[i//blockHeight][j//blockWidth]]
            for x in range(blockHeight):
                for y in range(blockWidth):
                    decompressed[i+x][j+y] = vector.pixels[x][y]
    
    return decompressed

if __name__ == "__main__":
    image = readFromImage("tests/house.bmp", [4, 4])
    codebook = generateCodebook(image, 8)
    labels = generateLabels(image)

    decompressed = decompress(labels, codebook)

    writeToCompressedFile(labels, codebook, "tests/compressed.bin")

    labels2, codebook2 = readFromCompressedFile("tests/compressed.bin")

    print(labels == labels2)
    
    # Print each codebook
    # for i, vector in codebook.items():
    #     print(f"Vector {i}:")
    #     for row in vector.pixels:
    #         print(row)
    #     print()
    
    # Print each codebook
    for i, vector in codebook2.items():
        print(f"Vector {i}:")
        for row in vector.pixels:
            print(row)
        print()