import subprocess
from os import remove,system

directory = "postaggers/"
jar_path = "ark-tweet-nlp-0.3.2/"
jar_name = "ark-tweet-nlp-0.3.2.jar"
file_name = "tweet.txt"

command = "java -Xmx500m -jar "+directory+jar_path+jar_name+" "+directory+file_name

def messageToFile(message):
    text_file = open(directory+file_name, "w")
    text_file.write(message)
    text_file.close()

def listToFile(messages):
    text_file = open(directory+file_name,"w")
    for m in messages:
        #sta dika sas minimata i allgi grammis mporei na min xreiazetai, tsekare to
        text_file.write(m+"\n")

    text_file.close()

def fileToList(path):
    tags = []
    data = open(path,"r")

    for line in data:
        #line=unicode(line,"utf-8")
        tags.append(line.split("\t")[1].split(" "))

    return tags    

#call java library in order to find the pos tags of the message
#pos tag for only one message
def pos_tag_message(message):

    #create temporary text file with the message
    messageToFile(message)
        
    #use the text file as input to the pos tagger and get output
    result = subprocess.check_output(command,shell=True)

    print (type(result))

    #get tokens
    tokens = result.split("\t")[1].split(" ")

    #delete temporary text file
    remove(directory+file_name)

    return tokens

#pos tag for a list of messages
def pos_tag_list(messages):

    #create temporary text file with the messages
    listToFile(messages)

    #use the text file as input to the pos tagger and get output
    system(command+" >"+directory+"out.txt")

    #read file to list
    results = fileToList(directory+"out.txt")

    #delete temporary text files
    remove(directory+file_name)
    remove(directory+"out.txt")

    return results
    

    
    
