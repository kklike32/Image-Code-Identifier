#!/usr/bin/python

#find the position of "#####" marker in grader file
#delete the ##### with the code from input file
#compile or pass to grader

#usage notes:
#"#####" code will be searched for consecutively, if it is not consecutive we won't find it
#input file cannot have any comments, we don't want to deal with them right now, although we could by having an array keep track of the lines and understanding that every character after the // in the same line is a comment
#only "#####" will be deleted/replaced from the graderFile
#it's up to the user to pass in the right files and make sure the languages match

#
#Finds the "#####" marker in the grader file
#@param _Lines: lines from the file
#@param _position: position tracker which will hold the position of the first '#' in the '#####'
#@return _position has the position of the first # after return
#
import getopt
import sys


def findGraderCode(_Lines, _position):
    counter = 0
    for line in _Lines:
        for char in line:
            if char == '#':
                counter += 1  # inc counter
                if counter == 4:  # when we see 5 #'s in a row, leave loop
                    _position -= 4 #get position of the first #?
                    return
            else:
                counter = 0  # reset counter
            _position += 1  # keep track of position
    print("Didn't find code \"#####\" in grader file") #stop program if we don't find the code
    sys.exit(0)

#
#Creates the new file which will be used for grading
#@param _Lines: lines of the grader file
#@param _inputLines: lines of the input file
#@param _position: position of the first # in the marker
#@param _newFile: file to be written to
#@returns nothing
#
def makeNewFile(_Lines, _inputLines, _position, _newFile):
    # put input code into grader file
    position2 = 0
    copied = False
    for lines in _Lines:
        for char in lines:
            if position2 < _position and position2 > _position + 4:  # as long as we aren't at any characters of #####
                _newFile.write(char)  # copy the code over one char at a time
            elif copied != True:  # if we have not copied yet
                copied = True  # make sure we don't copy again
                for lineInput in _inputLines:  # for each line in the input file
                    _newFile.write(lineInput)  # put the input code in the file
            position2 += 1  # inc position


#main code:
def main(argv):
    print("yay")
    argList = sys.argv[1:] #list of cmd line args past name of prog

    #vars for paths we will get from cmd line
    inputPath = ''
    graderPath = ''
    resultPath = ''

    try:
        args, values = getopt.getopt(argList, "h:igr:", ["Help", "Input", "Grader", "Result"]) #get args
        for curArg, curVal in args:
            if curArg in ("-h", "--Help"):
                print ("Usage: textToCompile.py -i <inputFilePath> -g <graderFilePath -r <resultFilePath>")
            elif curArg in ("-i", "--Input"):
                inputPath = curVal
            elif curArg in ("-g", "--Grader"):
                graderPath = curVal
            elif curArg in ("-r", "--Result"):
                resultPath = curVal

    except getopt.GetoptError:
        print("Usage: textToCompile.py -i <inputFilePath> -g <graderFilePath -r <resultFilePath>")
        sys.exit(0)

    graderFile = open(graderPath, 'r')
    Lines = graderFile.readLines() #get grader lines

    position = 0
    findGraderCode(Lines, position) #get the position of the first #

    inputFile = open(inputPath, 'r') #open input file (code to be graded)
    inputLines = inputFile.readLines() #get input lines

    newFile = open(resultPath, 'w') #open the grader file to write to it instead

    makeNewFile(Lines, inputLines, position, newFile) #copy input and grader into the new file

    #run the new file through a compiler/grader

